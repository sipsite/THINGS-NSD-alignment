import os
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import multiprocessing
import psutil

from select_subj import load_subj_img_index

import datetime
import time
import gc

def get_current_time_info():
    now = datetime.datetime.now()
    standard_format = now.strftime("%m-%d_%H-%M-%S")
    return standard_format


# ================= 配置区域 =================
# 指定使用那张 48GB 显存的显卡 (根据你的 nvidia-smi，GPU 4 是 48GB 空闲卡)
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

# 全局变量：用于多进程共享内存 (Linux Fork 模式下零拷贝)
global_things_img = None
global_nsd_img = None
co1 = np.array([10, 0.1, 0.1]) ## __adjustable__
co1 /= co1.sum()

# ================= 多进程 Worker 函数 =================
def calculate_fine_metrics_worker(args):
    """
    单个查询的精细计算任务
    args: (t_idx, candidate_indices, coarse_scores, top_r)
    """
    t_idx, candidate_indices, coarse_scores, top_r = args
    
    # 直接访问全局内存，无需拷贝
    img_t = global_things_img[t_idx]       # (H, W, 3)
    imgs_n = global_nsd_img[candidate_indices] # (K, H, W, 3)
    
    fine_results = []
    
    # 预处理 Things 图片用于 PixCorr (展平 & Center)
    flat_t = img_t.flatten().astype(np.float32)
    flat_t -= flat_t.mean()
    norm_t = np.linalg.norm(flat_t) + 1e-8
    
    for k_idx, img_n in enumerate(imgs_n):
        # --- 1. PixCorr 计算 ---
        flat_n = img_n.flatten().astype(np.float32)
        flat_n -= flat_n.mean()
        norm_n = np.linalg.norm(flat_n) + 1e-8
        
        # Pearson Correlation = Cosine of centered vectors
        pix_corr = np.dot(flat_t, flat_n) / (norm_t * norm_n)
        
        # --- 2. SSIM 计算 (最耗时) ---
        ssim_val = ssim(
            img_t, img_n, 
            channel_axis=2, 
            data_range=255,   # 假设输入是 uint8 0-255
            win_size=11, 
            gaussian_weights=True, 
            sigma=1.5
        )
        
        # --- 3. 最终分数融合 (Fusion) ---
        # 这里的权重你可以根据偏好调整
        # coarse_score 是归一化后的 Cosine Sim (通常 0.x - 0.9)
        # ssim_val 是 0 - 1
        # pix_corr 是 -1 - 1
        c_score = coarse_scores[k_idx]
        
        final_score = co1[0] * c_score + co1[1] * ssim_val + co1[2] * pix_corr
        
        fine_results.append((candidate_indices[k_idx], final_score, c_score, ssim_val, pix_corr))
    
    # 按最终分数排序，取 Top-R
    fine_results.sort(key=lambda x: x[1], reverse=True)
    top_r_res = fine_results[:top_r]
    
    # 返回精简结果: [things_idx, nsd_idx, final, coarse, ssim, pixcorr]
    return [[t_idx, int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])] for r in top_r_res]

# ================= 主系统类 =================
class RetrievalSystem:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 System initialized. Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
            
        # 动态计算 CPU 核数
        self.num_workers = self.get_optimal_worker_count(reserve_cores=4)
        print(f"   CPU Workers: {self.num_workers} (Auto-configured)")

    def get_optimal_worker_count(self, reserve_cores=4):
        """根据系统负载动态计算可用核数"""
        try:
            total_cores = len(os.sched_getaffinity(0))
        except AttributeError:
            total_cores = os.cpu_count()
        try:
            load_1min, _, _ = os.getloadavg()
            current_load = int(load_1min)
        except OSError:
            current_load = 0
            
        return max(2, min(total_cores - current_load - reserve_cores, total_cores - 2))

    def load_images(self, things_path, nsd_path):
        global global_things_img, global_nsd_img
        print(f"📥 Loading raw images into RAM (Fast Mode)...")
        
        # 假设 .npy 存储的是 uint8 (0-255)
        global_things_img = np.load(things_path)
        global_nsd_img = np.load(nsd_path)
        
        t_size = global_things_img.nbytes / (1024**3)
        n_size = global_nsd_img.nbytes / (1024**3)
        print(f"   Done. Things: {global_things_img.shape} ({t_size:.2f} GB)")
        print(f"         NSD:    {global_nsd_img.shape} ({n_size:.2f} GB)")
    def run_pipeline(self, feat_things, feat_nsd, weights, top_k=50, top_r=1, batch_size=4096):
        """
        feat_things/nsd: 字典 {'clip': np_array, 'alex2': np_array ...}
        weights: 字典 {'clip': 0.8, 'alex2': 0.2 ...}
        batch_size: 4096 (针对 48GB 显存优化)
        """
        
        num_things = next(iter(feat_things.values())).shape[0]
        # 随便取一个 nsd 特征看总数，用于计算 loop range
        num_nsd = next(iter(feat_nsd.values())).shape[0]
        
        final_results = []
        
        # NSD 分块大小 (20000 * 140k * 4B ≈ 10.4 GB)
        # 加上 Things 的 2GB 和系统开销，控制在 15-20GB 左右，绝对安全
        NSD_CHUNK_SIZE = 20000
        
        print(f"🔥 Starting Retrieval Loop (A-Batch={batch_size}, NSD-Chunk={NSD_CHUNK_SIZE}, Top-K={top_k})...")
        
        # --- Phase 2: Coarse Search (Matrix Multiplication) ---
        for i in tqdm(range(0, num_things, batch_size), desc="Processing Batches"):
            start = i
            end = min(i + batch_size, num_things)
            curr_bs = end - start
            
            # 初始化当前 Batch 的总分矩阵 (Batch_Size, Num_NSD)
            # 注意：这个矩阵只有 4096 x 73000 x 4B ≈ 1.1GB，常驻显存完全没问题
            total_sim = torch.zeros((curr_bs, num_nsd), device=self.device)
            
            # 累加各特征分数
            for key, w in weights.items():
                if w == 0: continue
                
                # 1. 准备 Things Batch (A) - 常驻显存
                # 即使是 Alex2，4096 张也才 2GB，可以接受
                ft_things = torch.from_numpy(feat_things[key][start:end]).float()
                ft_things = ft_things.contiguous().to(self.device)
                ft_things = F.normalize(ft_things, p=2, dim=1)
                
                # 2. 准备 NSD 数据源 (B) - 仍在 CPU 内存中
                nsd_source_cpu = feat_nsd[key]
                
                # 3. 分块计算 NSD 相似度
                sim_chunks = [] # 用于暂存各块的计算结果
                
                for n_start in range(0, num_nsd, NSD_CHUNK_SIZE):
                    n_end = min(n_start + NSD_CHUNK_SIZE, num_nsd)
                    
                    # [搬运] 切片 -> 搬入 GPU (最多占用 10GB)
                    ft_nsd_chunk = torch.from_numpy(nsd_source_cpu[n_start:n_end]).float()
                    ft_nsd_chunk = ft_nsd_chunk.contiguous().to(self.device)
                    ft_nsd_chunk = F.normalize(ft_nsd_chunk, p=2, dim=1)
                    
                    # [计算] (Batch_A, Dim) @ (Dim, Chunk_B) -> (Batch_A, Chunk_B)
                    # 结果矩阵很小，不用担心
                    chunk_sim = torch.matmul(ft_things, ft_nsd_chunk.T)
                    
                    # [暂存]
                    sim_chunks.append(chunk_sim)
                    
                    # [释放] 显式删除引用，确保显存回收给下一块
                    del ft_nsd_chunk
                
                # 4. 拼合 & 加权
                # 把所有小块拼成完整的 (Batch_A, Num_NSD)
                full_sim_matrix = torch.cat(sim_chunks, dim=1)
                total_sim += w * full_sim_matrix
                
                # 释放 Things 特征
                del ft_things
                # 释放中间结果
                del full_sim_matrix, sim_chunks

            # --- 至此，当前 Batch A 的所有特征加权总分已算出 ---
            
            # GPU 上直接取 Top-K
            top_k_scores, top_k_indices = torch.topk(total_sim, k=top_k, dim=1)
            
            # 转 CPU 准备精排
            indices_np = top_k_indices.cpu().numpy()
            scores_np = top_k_scores.cpu().numpy()
            
            # 释放 total_sim，腾出空间给下一个 Batch
            del total_sim
            
            # --- Phase 3: Fine Re-ranking (Multiprocessing CPU) ---
            # 准备任务列表
            tasks = []
            for b in range(curr_bs):
                t_idx = start + b
                # 将该 query 的任务打包
                tasks.append((t_idx, indices_np[b], scores_np[b], top_r))
            
            # 启动进程池
            current_workers = self.get_optimal_worker_count(reserve_cores=4)
            
            with multiprocessing.Pool(processes=current_workers) as pool:
                # 并行计算 SSIM & PixCorr
                batch_fine_results = pool.map(calculate_fine_metrics_worker, tasks)
            
            # 收集结果
            for res in batch_fine_results:
                final_results.extend(res)

        return final_results
import sys
import json
if __name__ == "__main__":
    # 1. 实例化系统
    system = RetrievalSystem()
    # 2. 路径配置 (请修改为你的实际路径)
    base_path = "/home/ysunem/12.21/THINGS-NSD_code/" # 你的数据目录
    things_npy = os.path.join(base_path, "things_img.npy") # 必须是 uint8
    nsd_npy = os.path.join(base_path, "nsd_img.npy")       # 必须是 uint8
    
    # 如果文件不存在，请先确保路径正确
    if os.path.exists(things_npy):
        system.load_images(things_npy, nsd_npy)
    else:
        print("⚠️ Warning: Image files not found. Skipping image loading (Demo mode).")
    if global_things_img.dtype != np.uint8 or global_nsd_img.dtype != np.uint8:
        print("❌ Error: Image .npy files must be of dtype uint8 (0-255).")
        sys.exit(1)


    # 4. 模拟/加载 Embedding 数据
    # 加载刚才生成的特征
    print("📥 Loading Precomputed Embeddings...")

    # 你的 CLIP 特征 (假设你本来就有)
    things_clip = np.load(base_path + 'things_fea.npy').astype(np.float32)
    nsd_clip = np.load(base_path + 'nsd_fea.npy').astype(np.float32)
    
    feat_things_dict = np.load(base_path + 'features/feat_things.npy', allow_pickle=True).item()
    feat_nsd_dict = np.load(base_path + 'features/feat_nsd.npy', allow_pickle=True).item()

    # 合并字典
    feat_things = feat_things_dict
    feat_things['clip'] = things_clip 

    feat_nsd = feat_nsd_dict
    feat_nsd['clip'] = nsd_clip 
    

    # 5. 定义粗筛权重 (Coarse Weights)
    co = np.array([7, 3, 3, 0.1, 3, 0.1]) ## __adjustable__
    co /= np.sum(co)
    weights = {
        'clip': co[0], 
        'alex2': co[1],  
        'alex5': co[2],  
        'incep': co[3],
        'eff': co[4],    
        'swav': co[5]
    }

    keep_index = load_subj_img_index(subj_id1=1) ## $$$
    for k, v in feat_nsd.items():
        feat_nsd[k] = v[keep_index]
    global_nsd_img = global_nsd_img[keep_index] 
    
    gc.collect()
    # 6. 运行 Pipeline
    results = system.run_pipeline(feat_things, feat_nsd, weights, top_k=50, top_r=5, batch_size=4096)


    # --- 配置阈值 (根据你的实际分数分布调整) ---
    # 提示：Final Score 是归一化的，理论最大值是 1.0
    THRESHOLD_1 = None  # 高置信度 (High Confidence)
    THRESHOLD_2 = None  # 中置信度 (Medium Confidence)


    # ================= 6.5. 绘制分数分布直方图 =================
    import matplotlib.pyplot as plt
    
    print("📈 Plotting score distributions...")
    
    # 1. 提取数据 (转为 Numpy 方便切片)
    # results结构: [t_idx, n_idx, final, coarse, ssim, pixcorr]
    data_arr = np.array(results) 
    
    scores_final = data_arr[:, 2]
    scores_coarse = data_arr[:, 3]
    scores_ssim = data_arr[:, 4]
    scores_pixcorr = data_arr[:, 5]
    THRESHOLD_1, THRESHOLD_2 = np.percentile(scores_final, [90, 70]) ## __adjustable__
    
    try :
        scores_struct = (scores_ssim * co1[1] + scores_pixcorr * co1[2]) / (co1[1] + co1[2])
    except:
        scores_struct = (scores_ssim * 0.3 + scores_pixcorr * 0.1) / 0.4
    # 3. 创建画布
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # --- Plot 1: Coarse Score (语义相似度) ---
    axes[0].hist(scores_coarse, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_title('Coarse Score (Deep Features)', fontsize=14)
    axes[0].set_xlabel('Score')
    axes[0].set_ylabel('Count')
    axes[0].grid(axis='y', alpha=0.5)

    # --- Plot 2: Structural Score (SSIM & PixCorr) ---
    axes[1].hist(scores_struct, bins=100, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1].set_title('Structural Score\n(0.3*SSIM + 0.1*Pix) / 0.4', fontsize=14)
    axes[1].set_xlabel('Score')
    axes[1].grid(axis='y', alpha=0.5)

    # --- Plot 3: Final Score (最终融合) ---
    axes[2].hist(scores_final, bins=100, color='salmon', edgecolor='black', alpha=0.7)
    axes[2].set_title('Final Weighted Score', fontsize=14)
    axes[2].set_xlabel('Score')
    axes[2].grid(axis='y', alpha=0.5)

    # (可选) 在 Final Score 图上画出你预想的 Threshold 辅助线
    # 帮你判断 T1 和 T2 切在哪里合适
    axes[2].axvline(THRESHOLD_1, color='red', linestyle='dashed', linewidth=2, label='T1 (High)')
    axes[2].axvline(THRESHOLD_2, color='blue', linestyle='dashed', linewidth=2, label='T2 (Med)')
    axes[2].legend()

    plt.tight_layout()
    
    # 保存图片
    plt.savefig("score_distribution.png", dpi=300)
    print("✅ Histogram saved to score_distribution.png. Check this file to set thresholds!")
    
    # 7. 结果过滤与保存 (JSON 格式)

    print(f"📊 Filtering results with T1={THRESHOLD_1}, T2={THRESHOLD_2}...")

    # 初始化 JSON 结构
    json_output = {
        "metadata": {
            "total_queries": len(results),
            "threshold_high": THRESHOLD_1,
            "threshold_medium": THRESHOLD_2,
            "weights": weights
        },
        "high_confidence_group": [],   # > T1
        "medium_confidence_group": [],  # T2 <= score < T1
        "discarded": []  # score < T2
    }

    # 计数器
    count_high = 0
    count_med = 0
    count_discard = 0

    for row in results:
        # row 格式: [t_idx, n_idx, final, coarse, ssim, pixcorr]
        # 注意：这里必须转为 Python 原生类型 (float/int)，否则 json.dump 会报错
        item = {
            "things_id": int(row[0]),
            "nsd_id": int(row[1]),
            "score_final": round(float(row[2]), 4),
            "score_coarse": round(float(row[3]), 4),
            "score_ssim": round(float(row[4]), 4),
            "score_pixcorr": round(float(row[5]), 4)
        }

        score = item["score_final"]

        if score >= THRESHOLD_1:
            json_output["high_confidence_group"].append(item)
            count_high += 1
        elif score >= THRESHOLD_2:
            json_output["medium_confidence_group"].append(item)
            count_med += 1
        else:
            json_output["discarded"].append(item)
            count_discard += 1

    # 保存文件
    output_json_path = f"retrieval_filtered{get_current_time_info()}.json"
    with open(output_json_path, 'w') as f:
        json.dump(json_output, f, indent=4) 

    print(f"✅ Filtered JSON saved to {output_json_path}")
    print(f"   - High Confidence (> {THRESHOLD_1}): {count_high} pairs")
    print(f"   - Medium Confidence ([{THRESHOLD_2}, {THRESHOLD_1})): {count_med} pairs")
    print(f"   - Discarded (< {THRESHOLD_2}): {count_discard} pairs")

    import pandas as pd
    pd.DataFrame(results).to_csv(f"retrieval_raw_backup{get_current_time_info()}.csv", index=False)