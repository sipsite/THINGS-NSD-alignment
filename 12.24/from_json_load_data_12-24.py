
import os
import numpy as np
import h5py
import json
import torch
import sys
import datetime
import time
from PIL import Image
import h5py
import about_index

def get_current_time_info(format = 3): # 1:m+d; 2. h+m; 3.m+d+h+m; 4:m+d+h+m+s
    now = datetime.datetime.now()
    if format == 1:
        standard_format = now.strftime("%m-%d")
    elif format == 2:
        standard_format = now.strftime("%H-%M")
    elif format == 3:
        standard_format = now.strftime("%m-%d_%H-%M")
    elif format == 4:
        standard_format = now.strftime("%m-%d_%H-%M-%S")
    return standard_format

nsd_data_path = "nsd-data/"
things_data_path = "things-eeg-data/"

def from_json_load_eeg_data(match_index, things_subj_id=1):
    data_path=things_data_path
    things_subj_sig_path = os.path.join(data_path, "Preprocessed_data_250Hz_whiten/", f"sub-{things_subj_id:02d}", "train.pt")
    things_data_all = torch.load(things_subj_sig_path, map_location='cpu', weights_only=False)["eeg"]
    things_data = []
    all_items = match_index.get(f"data", [])

    things_img_path_base = os.path.join(data_path, "training_images")
    things_img_find = np.load(os.path.join(data_path, "image_metadata.npy"), allow_pickle=True).item()
    things_imgs = []
    for item in all_items:
        things_id = item["things_id"]

        things_data.append(things_data_all[things_id, :, :])
        things_img_path = os.path.join(things_img_path_base, things_img_find['train_img_concepts'][things_id], things_img_find['train_img_files'][things_id])
        target_size = (224, 224)
        img = Image.open(things_img_path).convert("RGB").resize(target_size, resample=Image.LANCZOS)
        things_imgs.append(img)
    things_imgs = np.array(things_imgs)

    return things_data, things_imgs

def from_json_load_nsd_data(match_index, subj_index):
    good_index = about_index.process_nsd_index(subj_index)
    img2fmri_index = about_index.img2fmri_index(subj_index, is_train=True)

    data_path = nsd_data_path
    nsd_subj1_sig_path = os.path.join(data_path, f"betas_all_subj0{subj_index}_fp32_renorm.hdf5")
    nsd_embed_path = os.path.join(f"fMRI_embeddings/subj-0{subj_index}/train/embeddings_subj0{subj_index}_train.npy")
    f11 = h5py.File(nsd_subj1_sig_path, "r")
    nsd_data_all = f11["betas"]
    nsd_imgs = []
    f22 = h5py.File(os.path.join(data_path, "coco_images_224_float16.hdf5"), "r")
    nsd_img_all = np.array(f22['images'])
    nsd_data = []
    nsd_embed_all = np.load(nsd_embed_path, allow_pickle=True)
    nsd_embed = []
    all_items = []
    all_items = match_index.get(f"data", [])
    for item in all_items:
        nsd_id = item["nsd_id"]
        c3 = []
        embed3 = []
        for _ in range(3):  # each nsd_id has 3 repetitions
            c3.append(nsd_data_all[good_index[nsd_id][_], :])
            embed3.append(nsd_embed_all[img2fmri_index[nsd_id][_], :])
        nsd_embed.append(np.array(embed3))
        nsd_data.append(c3)
        nsd_imgs.append(nsd_img_all[nsd_id, :, :, :])
    nsd_imgs = np.array(nsd_imgs)
    f11.close()
    f22.close()
    return nsd_data, nsd_imgs, nsd_embed


## para : 
## nsd_subj (a number), things_subj (a list), threshold, save_path (relative path)
def load(nsd_subj=1, things_subj=np.arange(1,11).tolist(), threshold=0.45):
    save_path = f"match_save_{get_current_time_info(3)}/subj{nsd_subj:02d}/"
    print(f"start running, time : {get_current_time_info(2)}; (approximated time : 26 min / nsd subject)")
    os.makedirs(save_path, exist_ok=True)
    
    json_path = f"json_12-24/subj-0{nsd_subj}/"
    wanted_json_file = f"retrieval_rearranged_subj0{nsd_subj}_t{threshold}_12-24.json" 
    json_path = os.path.join(json_path, wanted_json_file)
    try : 
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File {json_path} not found.")
        exit(1)
    
    
    good_index = about_index.process_nsd_index(nsd_subj)
    

    save_path = os.path.join(save_path, f"result_{get_current_time_info(4)}/")
    os.makedirs(save_path, exist_ok=True)
    
    nsd_sig, nsd_img, nsd_embed = from_json_load_nsd_data(match_index=json_data, subj_index=nsd_subj)
    print("nsd_sig.shape : ", np.array(nsd_sig).shape)
    print("nsd_img.shape : ", np.array(nsd_img).shape)
    print("nsd_embed.shape : ", np.array(nsd_embed).shape)
    file_name_ns = f"nsd_subj0{nsd_subj}_sig.npy"
    file_name_ni = f"nsd_subj0{nsd_subj}_img.npy"
    file_name_ne = f"nsd_subj0{nsd_subj}_embed.npy"
    np.save(os.path.join(save_path, file_name_ns), nsd_sig)
    np.save(os.path.join(save_path, file_name_ni), nsd_img)
    np.save(os.path.join(save_path, file_name_ne), nsd_embed)
    print(f"NSD data saved. Time : {get_current_time_info(2)}")
    for things_subj_id in things_subj:
        things_sig, things_img = from_json_load_eeg_data(match_index=json_data, things_subj_id=things_subj_id)
        if things_subj_id == things_subj[0]:
            print("things_sig.shape : ", np.array(things_sig).shape)
            print("things_img.shape : ", np.array(things_img).shape)
        file_name_es = f"things_subj{things_subj_id:02d}_sig.npy"
        file_name_ei = f"things_subj{things_subj_id:02d}_img.npy"
        np.save(os.path.join(save_path, file_name_es), things_sig)
        np.save(os.path.join(save_path, file_name_ei), things_img)
    print(f"THINGS data saved. Time : {get_current_time_info(2)}")
    print(f"done. Time : {get_current_time_info(2)}")


if __name__ == "__main__":

    ## before running : 
    ## 1. put eeg signal folder, "Preprocessed_data_250Hz_whiten", under "things-eeg-data" folder
    ## 2. put things images folder "training_images", and "image_metadata.npy", under "things-eeg-data" folder
    ## 3. put all nsd data directly under "nsd-data" folder
    ## 4. put fMRI embeddings folder, "fMRI_embeddings", under root directory (i.e. "12.24" folder)
    
    load(nsd_subj=2, threshold=0.45)
    ## ps. threshold can be chosen from : 0.4, 0.42, 0.45, 0.47, 0.5
    