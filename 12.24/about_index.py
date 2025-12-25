
## some fact about index : 
## nsd sig : 30K
## nsd img : 73K
## nsd embed : 27K (training) or 3K (test)

from turtle import back
import h5py
import numpy as np
import random
import os

nsd_path = "nsd-data/"

# get index of training images (i.e. images that are shown to subject uniquely) 
# subj_id1=-1 -> get index of test images (i.e. images shown to all subjects)
def load_subj_img_index(subj_id1):
    subj_id_path = nsd_path + "COCO_73k_subj_indices.hdf5"
    
    with h5py.File(subj_id_path, "r") as f:
        subj_1_data = f[f"subj0{subj_id1}"][:]
        subj_id2 = 2 if subj_id1 == 1 else 1
        subj_2_data = f[f"subj0{subj_id2}"][:]
        seen_img1 = set()
        seen_img2 = set()
        for _ in subj_1_data:
            seen_img1.add(_)
        for _ in subj_2_data:
            seen_img2.add(_)
        if subj_id1 == -1:
            seen_img1 = seen_img1 & seen_img2
        else:
            seen_img1 = seen_img1 - seen_img2
        return sorted(list(seen_img1))

## good_index (a dict) : img index -> a list of 3 sig index (total size=10K)
## "COCO_73k_subj_indices.hdf5" : sig index -> img index (total size=30K)
def process_nsd_index(subj_id):
    nsd_index_path = os.path.join(nsd_path, "COCO_73k_subj_indices.hdf5")
    good_index = dict()
    with h5py.File(nsd_index_path, "r") as f:
        good_index[f"subj{subj_id:02d}"] = {}
        subj_data = f[f"subj{subj_id:02d}"][:]
        for img_idx, nsd_idx in enumerate(subj_data):
            good_index[f"subj{subj_id:02d}"].setdefault(nsd_idx, []).append(img_idx)
    good_index = good_index[f"subj{subj_id:02d}"]
    return good_index

## img2fmri_index (a dict) : img index -> fMRI index 
## ps. fMRI_embeddings' original index : the same order as sig index; but due to selection of training/testing set, index jumps
def img2fmri_index(nsd_subj, is_train):
    good_index = process_nsd_index(nsd_subj)
    sig_list = []
    if is_train:
        img_index = load_subj_img_index(nsd_subj)
    else:
        img_index = load_subj_img_index(-1)
    for img_idx in img_index:
        sig_list.extend(good_index[img_idx])
    sig_list = sorted(sig_list) ## fMRI_embeddings index -> sig index
    sig2fmri_index = {sig_idx: fmri_idx for fmri_idx, sig_idx in enumerate(sig_list)}
    img2fmri_index = {}
    for img_idx in img_index:
        img2fmri_index[img_idx] = [sig2fmri_index[sig_idx] for sig_idx in good_index[img_idx]]
    return img2fmri_index


if __name__ == "__main__":
    pass