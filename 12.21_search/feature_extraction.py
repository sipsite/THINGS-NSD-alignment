import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ================= 配置区域 =================
# 指定 48GB 显存的那张卡
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 批处理大小 (48GB 显存可以开到 512 甚至更高，这里用 512 很稳)
BATCH_SIZE = 512
NUM_WORKERS = 16 # DataLoader 的 CPU 进程数

# ================= 模型包装类 =================
class AllInOneExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        print(f"🚀 Loading models to {DEVICE}...")
        
        # 1. AlexNet (用于 Alex2, Alex5)
        # weights='DEFAULT' 等同于 pretrained=True
        self.alexnet = models.alexnet(weights='DEFAULT').features.eval()
        
        # 2. Inception V3 (需要输入 299x299)
        self.inception = models.inception_v3(weights='DEFAULT', transform_input=False).eval()
        # Inception 在 eval 模式下直接输出 logits，我们需要提取特征可能需要 hook，
        # 但通常学术界直接用它的输出层作为 embedding，或者 avgpool 层。
        # 这里为了通用性，我们取最后的输出 (fc 前的一层通常效果最好，但 torchvision 接口默认给 logits)
        # 简单起见，我们取 fc 层的输入。Inception forward 稍微复杂，我们只用其 forward 逻辑
        self.inception.fc = nn.Identity() # 替换掉分类头，直接输出 2048 维特征
        
        # 3. EfficientNet-B1
        self.effnet = models.efficientnet_b1(weights='DEFAULT')
        self.effnet.classifier = nn.Identity() # 替换分类头
        self.effnet.eval()
        
        # 4. SwAV (ResNet50 based)
        # SwAV 是无监督训练的 ResNet50
        print("   Loading SwAV from torch.hub...")
        self.swav = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        self.swav.fc = nn.Identity()
        self.swav.eval()

        # 定义预处理
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 定义 GPU 上的 Resize 操作 (极速)
        self.resize_224 = T.Resize((224, 224), antialias=True)
        self.resize_299 = T.Resize((299, 299), antialias=True) # Inception专用

    def forward(self, x_uint8):
        """
        输入: (B, 3, H, W) uint8 Tensor [0-255]
        输出: 字典 {'alex2': ..., 'incep': ...}
        """
        results = {}
        
        # 1. 归一化: uint8 [0-255] -> float [0-1] -> Normalize
        x = x_uint8.float() / 255.0
        x = self.normalize(x)
        
        # --- 分流 A: 224x224 (AlexNet, Eff, SwAV) ---
        x_224 = self.resize_224(x)
        
        # AlexNet (2 & 5)
        # AlexNet features 结构:
        # [0]Conv1 [1]ReLU [2]Pool [3]Conv2 [4]ReLU (Alex2) ... [10]Conv5 [11]ReLU (Alex5)
        with torch.no_grad():
            feat = self.alexnet[:5](x_224)
            results['alex2'] = feat.flatten(start_dim=1).cpu().numpy()
            
            feat = self.alexnet[:12](x_224)
            results['alex5'] = feat.flatten(start_dim=1).cpu().numpy()
            
            # EfficientNet
            results['eff'] = self.effnet(x_224).cpu().numpy()
            
            # SwAV
            results['swav'] = self.swav(x_224).cpu().numpy()

        # --- 分流 B: 299x299 (Inception) ---
        x_299 = self.resize_299(x)
        with torch.no_grad():
            results['incep'] = self.inception(x_299).cpu().numpy()
            
        return results

# ================= 提取工具函数 =================
def extract_all_features(image_array, model, desc="Extracting"):
    """
    image_array: numpy array (N, H, W, 3) uint8
    """
    # 转换为 TensorDataset，利用 DataLoader 的多进程预取
    # 这里的 Tensor 是 CPU 上的，DataLoader 负责搬运
    # 注意：为了节省显存，我们传入 DataLoader 的是 permute 后的引用
    
    # 这一步很快，因为只是 view 变换
    tensor_data = torch.from_numpy(image_array).permute(0, 3, 1, 2) # (N, 3, H, W)
    dataset = TensorDataset(tensor_data)
    
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True # 加速 CPU -> GPU 传输
    )
    
    # 容器初始化
    features = {
        'alex2': [], 'alex5': [], 'incep': [], 'eff': [], 'swav': []
    }
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            imgs = batch[0].to(DEVICE, non_blocking=True) # (B, 3, H, W)
            
            # 前向传播
            batch_feats = model(imgs)
            
            # 收集结果
            for k, v in batch_feats.items():
                features[k].append(v)
    
    # 合并 List 为 Numpy Array
    print(f"Concatenating features for {desc}...")
    final_dict = {}
    for k, v_list in features.items():
        final_dict[k] = np.concatenate(v_list, axis=0)
        print(f"  -> {k}: shape {final_dict[k].shape}")
        
    return final_dict

# ================= 主流程 =================
if __name__ == "__main__":
    # 1. 路径配置
    base_path = "/home/ysunem/12.21/THINGS&NSD_code_ver2/"
    things_path = os.path.join(base_path, "things_img.npy")
    nsd_path = os.path.join(base_path, "nsd_img.npy")
    
    output_dir = "/home/ysunem/12.21/THINGS&NSD_code_ver2/features/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 初始化模型 (一次性加载到 48G 显存)
    extractor = AllInOneExtractor().to(DEVICE)
    
    # 3. 处理 Things 数据
    if os.path.exists(things_path):
        print(f"📥 Loading Things images from {things_path}...")
        things_img = np.load(things_path) # (16k, H, W, 3) uint8
        
        feats_things = extract_all_features(things_img, extractor, desc="Things")
        
        # 保存
        save_path = os.path.join(output_dir, "feat_things.npy")
        np.save(save_path, feats_things)
        print(f"✅ Saved Things features to {save_path}")
        
        # 释放内存
        del things_img, feats_things
    else:
        print("⚠️ Things images not found.")

    # 4. 处理 NSD 数据
    if os.path.exists(nsd_path):
        print(f"📥 Loading NSD images from {nsd_path}...")
        nsd_img = np.load(nsd_path) # (73k, H, W, 3) uint8
        
        feats_nsd = extract_all_features(nsd_img, extractor, desc="NSD")
        
        # 保存
        save_path = os.path.join(output_dir, "feat_nsd.npy")
        np.save(save_path, feats_nsd)
        print(f"✅ Saved NSD features to {save_path}")
        
        del nsd_img, feats_nsd
    else:
        print("⚠️ NSD images not found.")
        
    print("🎉 All feature extraction completed.")