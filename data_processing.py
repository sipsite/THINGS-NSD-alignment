
## this file : 
## 1. process and save THINGS and NSD images and features as .npy files
## 2. process and save NSD clustering labels (new-ver)

from utils import plot, dict1

import os
import cv2
import random
import matplotlib.pyplot as plt
import datetime
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import math
import h5py

import matplotlib
matplotlib.use('Agg')

def get_current_time_info():
    now = datetime.datetime.now()
    standard_format = now.strftime("%m-%d_%H-%M-%S")
    return standard_format

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")




## --------------------------------------------------------------------------------- ## 
## load NSD labels
## result : dt1 : cluster labels (0~40) for all 73k images

## __adjustable__ : the dir of NSD data
NSD_data_path = ""  

ARR_PATH1 = NSD_data_path + "semantic_cluster_names.npy"
ARR_PATH2 = NSD_data_path + "COCO_73k_semantic_cluster.npy"

def load0():
    names = np.load(ARR_PATH1, allow_pickle=True)

    dt1 = np.load(ARR_PATH2, allow_pickle=True)
    print(f"dt1 :  {dt1.shape}")
    return names, dt1





## --------------------------------------------------------------------------------- ## 
## data processing functions

def equalizeHist111(imgs):
    print("img.shape : ", imgs.shape)
    print("imgs.dtype : ", imgs.dtype)
    if imgs.dtype != np.uint8:
        imgs = (imgs * 255).clip(0, 255).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    result = []
    for i in range(imgs.shape[0]):
        img = imgs[i, :, :, :]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # 对 L 通道做均衡化
        
        # l_channel = clahe.apply(l_channel)
        l_channel = cv2.equalizeHist(l_channel)

        # 合并并转回 BGR
        lab = cv2.merge((l_channel, a, b))
        result.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
    result = np.array(result)
    if result.dtype != np.uint8:
        result = (result * 255).clip(0, 255).astype(np.uint8)
    return result


file_path = NSD_data_path + 'coco_images_224_float16.hdf5'
def get_NSD_images_and_features():
    try:
        with h5py.File(file_path, 'r') as hf:
            with torch.no_grad():
                dataset_name = list(hf.keys())[0]
                images_dataset = hf[dataset_name].astype(np.float32)
                images_dataset = np.transpose(images_dataset, (0, 2, 3, 1))  # NCHW to NHWC
                images_dataset = np.clip(images_dataset, 0.0, 1.0) * 255.0
                images_dataset = images_dataset.astype(np.uint8)
                images_dataset = equalizeHist111(images_dataset)
                print("images_dataset.shape : ", images_dataset.shape)

                # compute features
                fea11 = []
                batch_size = 512
                for i in range(0, len(images_dataset), batch_size):
                    batch_images = images_dataset[i : i + batch_size, :, :, :]
                    inputs = processor(images=batch_images, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(device)
                    with torch.no_grad():
                        batch_fea = model.get_image_features(pixel_values=pixel_values)
                        fea11.append(batch_fea.detach().cpu().numpy())
                fea11 = np.concatenate(fea11, axis=0)
                return images_dataset, fea11
    except Exception as e:
        print(f"读取 HDF5 文件时出错: {e}")


## __adjustable__ : the root dir of THINGS-EEG2 images
root_dir = ""
# root_dir1 = root_dir + "test_images/test_images/"
root_dir2 = root_dir + "training_images/training_images/"
def process_THINGS_EEG_images():
    global root_dir
    found_files = []
    root_dir = os.path.abspath(root_dir2)
    folder_names = os.listdir(root_dir) 
    for folder_name_level_1 in folder_names:
        path_level_1 = os.path.join(root_dir, folder_name_level_1)
        if not os.path.isdir(path_level_1) or folder_name_level_1.startswith('.'):
            continue
        for file_name in os.listdir(path_level_1):
                if file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg'):
                    full_file_path = os.path.join(path_level_1, file_name)
                    img_array_rgb = cv2.cvtColor(cv2.imread(full_file_path), cv2.COLOR_BGR2RGB)
                    new_size = (224, 224)
                    downsampled_img_cv = cv2.resize(img_array_rgb, new_size, interpolation=cv2.INTER_AREA)
                    found_files.append(downsampled_img_cv)
    found_files = np.array(found_files)
    found_files = equalizeHist111(found_files)
    print("found_files.shape : ", found_files.shape)
    return found_files


def get_THINGS_EEG_features(found_files):
    images_dataset = found_files
    fea11 = []

    # compute features
    batch_size = 512
    for i in range(0, len(images_dataset), batch_size):
        batch_images = images_dataset[i : i + batch_size, :, :, :]
        inputs = processor(images=batch_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            batch_fea = model.get_image_features(pixel_values=pixel_values)
            fea11.append(batch_fea.detach().cpu().numpy())
    fea11 = np.concatenate(fea11, axis=0)
    return fea11



## --------------------------------------------------------------------------------- ## 
## main function

## __adjustable__ : the save path (dir)
save_path = "" 

if __name__ == "__main__":
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    names, dt1 = load0()
    dt1 = np.array([dict1[_[9:]] for _ in dt1]) # ps. removing "photo of " 
    np.save(save_path + "nsd_label_" + get_current_time_info() + ".npy", dt1)

    img1 = process_THINGS_EEG_images()
    np.save(save_path + "things_img_" + get_current_time_info() + ".npy", img1)
    fea1 = get_THINGS_EEG_features(img1)
    np.save(save_path + "things_fea_" + get_current_time_info() + ".npy", fea1)
    print("things data saved.")

    img2, fea2 = get_NSD_images_and_features()
    np.save(save_path + "nsd_img_" + get_current_time_info() + ".npy", img2)
    np.save(save_path + "nsd_fea_" + get_current_time_info() + ".npy", fea2)
    print("nsd data saved.")