

import h5py
import numpy as np
import random


nsd_path = "/home/ysunem/12.21/nsd-data/"

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
        seen_img1 = seen_img1 - seen_img2
        return list(seen_img1)





        