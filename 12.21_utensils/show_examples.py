import os
import json
import numpy as np
import random
from PIL import Image
from tqdm import tqdm

import sys
import os

import datetime
import time

def get_current_time_info():
    now = datetime.datetime.now()
    standard_format = now.strftime("%m-%d_%H-%M-%S")
    return standard_format


# 1. Get the absolute path to the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 2. Add the parent directory to sys.path
sys.path.append(parent_dir)

# 3. Now you can import normally
from utils import plot

# ================= 配置 =================

## __adjustable__
base_path = "/home/ysunem/12.21/THINGS-NSD_code/"  # 你的数据根目录
json_path = "retrieval_rearranged_12-22_20-39__55_50_47.json"             # 你的结果文件
output_root = f"samples_{get_current_time_info()}/"                     # 导出图片的根目录
SAMPLES_PER_GROUP = 40                            # 每组抽多少对

# ================= 工具函数 =================
def save_image(img_array, save_path):
    """把 numpy (H,W,3) uint8 转存为 png"""
    # 确保是 uint8
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    
    img = Image.fromarray(img_array)
    img.save(save_path)

# ================= 主流程 =================
if __name__ == "__main__":
    # 1. 加载图片大矩阵 (这是最耗内存的一步，但为了导出图片必须做)
    print("📥 Loading raw images from .npy (Might take memory)...")
    things_npy = os.path.join(base_path, "things_img.npy")
    nsd_npy = os.path.join(base_path, "nsd_img.npy")
    
    # 使用 mmap_mode='r' 可以省内存！不需要把 300G 全读进 RAM，只读需要的
    # 如果你的 SSD 够快，这会非常快且省内存
    things_img = np.load(things_npy, mmap_mode='r') 
    nsd_img = np.load(nsd_npy, mmap_mode='r')
    
    print(f"   Things shape: {things_img.shape}")
    print(f"   NSD shape:    {nsd_img.shape}")

    # 2. 读取 JSON 结果
    print(f"📂 Loading results from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    groups = {
        "high": data.get("group0", []),
        "medium": data.get("group0", []),
        "low": data.get("group2", []) # 如果之前保存了的话
    }

    # 3. 采样并保存
    for group_name, item_list in groups.items():
        total_items = len(item_list)
        if total_items == 0:
            print(f"⚠️ Group '{group_name}' is empty. Skipping.")
            continue
            
        # 创建子目录
        save_dir = os.path.join(output_root, group_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # 随机采样 (如果不够 40 个就全取)
        sample_count = min(SAMPLES_PER_GROUP, total_items)
        sampled_items = random.sample(item_list, sample_count)
        
        print(f"📸 Dumping {sample_count} pairs for [{group_name}]...")
        
        for i, item in enumerate(tqdm(sampled_items)):
            t_idx = item['things_id']
            n_idx = item['nsd_id']
            score = item['score_final']
            
            # --- 命名规范 ---
            # 格式: pair_{序号}_score_{分数}_{来源}.png
            # 这样在文件夹里按名称排序时，每一对图片会挨在一起
            t_name = f"pair_{i:02d}_score_{score:.4f}_A_Things.png"
            n_name = f"pair_{i:02d}_score_{score:.4f}_B_NSD.png"
            
            # 提取像素 (mmap 模式下，这里才真正发生磁盘 IO)
            img_t = things_img[t_idx]
            img_n = nsd_img[n_idx]
            
            # 保存
            save_image(img_t, os.path.join(save_dir, t_name))
            save_image(img_n, os.path.join(save_dir, n_name))
            
    print(f"✅ All samples dumped to: {os.path.abspath(output_root)}")
    print("   Folder structure:")
    print(f"   ├── {output_root}high/")
    print(f"   ├── {output_root}medium/")
    print(f"   └── {output_root}low/")
    plot(f"{output_root}high/", save_path=output_root)
    plot(f"{output_root}medium/", save_path=output_root)
    plot(f"{output_root}low/", save_path=output_root)
