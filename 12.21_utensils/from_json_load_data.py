
import os
import numpy as np
import h5py
import json
import torch
import sys
import datetime
import time

def get_current_time_info():
    now = datetime.datetime.now()
    standard_format = now.strftime("%m-%d_%H-%M-%S")
    return standard_format

save_path = "/home/ysunem/12.21/THINGS-NSD_code/12.21_utensils/sig_match_save/"
good_index = None
nsd_data_path = "/home/ysunem/12.21/nsd-data/"
things_data_path = "/home/ysunem/12.21/things-eeg-data/Preprocessed_data_250Hz_whiten/"

def from_json_load_eeg_data(match_index, wanted_grp = [0, 1, 2], data_path=things_data_path, things_subj_id=1):
    things_subj_sig_path = os.path.join(data_path, f"subj-{things_subj_id:02d}", "train.pt")
    things_data_all = torch.load(things_subj_sig_path, map_location='cpu', weights_only=False)["eeg"]
    things_data = []    
    all_items = []
    for grp_id in wanted_grp:
        all_items.extend(match_index.get(f"group{grp_id}", []))
    for item in all_items:
        things_id = item["things_id"]
        things_data.append(things_data_all[things_id, :, :, :])
    return things_data

def from_json_load_nsd_data(match_index, wanted_grp = [0, 1, 2], data_path=nsd_data_path):
    nsd_subj1_sig_path = os.path.join(data_path, "betas_all_subj01_fp32_renorm.hdf5")
    global good_index
    with h5py.File(nsd_subj1_sig_path, "r") as f:
        nsd_data_all = f["betas"]
        nsd_data = []
        all_items = []
        for grp_id in wanted_grp:
            all_items.extend(match_index.get(f"group{grp_id}", []))
        for item in all_items:
            nsd_id = item["nsd_id"]
            c4 = []
            for _ in range(3):  # each nsd_id has 3 repetitions
                print(good_index[nsd_id][_])
            nsd_data.append(c4)
    return nsd_data

def process_nsd_index():
    nsd_index_path = os.path.join(nsd_data_path, "COCO_73k_subj_indices.hdf5")
    good_index = dict()
    with h5py.File(nsd_index_path, "r") as f:
        subj_id = 1 
        good_index[f"subj{subj_id:02d}"] = {}
        subj_data = f[f"subj{subj_id:02d}"][:]
        for img_idx, nsd_idx in enumerate(subj_data):
            good_index[f"subj{subj_id:02d}"].setdefault(nsd_idx, []).append(img_idx)
    save_file_name = os.path.join(save_path, f"nsd_good_index_{get_current_time_info()}.pt")
    torch.save(good_index, save_file_name)
    print(f"Saved dictionary to {save_file_name}")
    return good_index["subj01"]


if __name__ == "__main__":
    os.makedirs(save_path, exist_ok=True)
    json_path = "/home/ysunem/12.21/THINGS-NSD_code/12.21_utensils/retri_ver3.json"
    try : 
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File {json_path} not found.")
        exit(1)

    ## good_index for nsd subj01
    ## set it to 0 after the first time
    first_time = 0
    if first_time:
        good_index = process_nsd_index()
    else : 
        good_index_path = os.path.join(save_path, f"nsd_good_index_12-22_16-11-40.pt")
        good_index = torch.load(good_index_path, map_location='cpu', weights_only=False)["subj01"]
    
    save_path = os.path.join(save_path, f"{get_current_time_info()}/")
    os.makedirs(save_path, exist_ok=True)
    
    wanted_grp = [0, 1, 2]
    nsd_data = from_json_load_nsd_data(match_index=json_data, wanted_grp=wanted_grp)
    print("nsd_data.shape : ", np.array(nsd_data).shape)
    file_name1 = f"nsd_subj01_data.pt"
    torch.save(nsd_data, os.path.join(save_path, file_name1))
    for things_subj_id in range(1, 11):
        things_data = from_json_load_eeg_data(match_index=json_data, wanted_grp=wanted_grp, things_subj_id=things_subj_id)
        if things_subj_id == 1:
            print("things_data.shape : ", things_data.shape)
        file_name2 = f"things_subj{things_subj_id:02d}_data.pt"
        torch.save(things_data, os.path.join(save_path, file_name2))