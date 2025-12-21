
## this file : show examples, do PCA and plot, do THINGS-to-NSD retrieval




import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cate_pair", type = int, default = None)






import h5py
import os
import matplotlib.pyplot as plt
from PIL import Image
import math
import random

import datetime
import time
import cv2

from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')


import pandas as pd
import numpy as np

import scipy.io

def get_current_time_info():
    now = datetime.datetime.now()
    standard_format = now.strftime("%m-%d_%H-%M-%S")
    return standard_format


color_array = np.array([
    '#DC143C', '#B8860B', '#FFA500', '#DAA520', '#CD5C5C', 
    '#B22222', '#C0C000', '#D8BFD8', '#BA55D3', 

    '#008B8B', '#00CED1', '#4682B4', '#1E90FF', '#00BFFF', 
    '#6495ED', '#48D1CC', '#5F9EA0', '#40E0D0',

    '#3CB371', '#008000', '#2E8B57', '#6B8E23', '#556B2F', 
    '#9ACD32', '#ADFF2F', '#7CFC00', '#8FBC8F',
    '#000000'
], dtype='<U7')


from utils import plot
from utils import dict1, dict0

dt1 = np.load("nsd_label.npy")  


## --------------------------------------------------------------------------------- ##

## load THINGS-EEG2 cluster labels 

load_path2 = "things_label.npy" # cluster labels for THINGS-EEG2 images
cluster_labels = np.load(load_path2) + 1
print("cluster_labels.shape : ", cluster_labels.shape)
num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0) 


## --------------------------------------------------------------------------------- ##
## (preparation)

## __adjustable__ : the save path (dir)
save_path_0 = f"pca_and_retri_{get_current_time_info()}/"



from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel
import torch

## load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# no need to modify these 
show_examples = None 
current_cate = None
report_file = save_path_0 + f"report_{get_current_time_info()}.txt"
num_of_examples = None

# write [message] to report_file
def log_message(message):
    with open(report_file, 'a', encoding='utf-8') as f:
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
        f.write(f"{timestamp} {message}\n")




## __adjustable__
## 1 means run the function; 0 means not run
show_examples = 1
do_plot = 1
do_e2n_retrieval = 1
## number of examples to show for each category
num_of_examples = 10 

## this the number of samples to use. (can use a list of different scales)
scale = [1200] 

## which category pairs to process (according to dict0)
only_wanted_cate = np.arange(24).tolist()


random.seed(1234 + datetime.datetime.now().minute * 97 + datetime.datetime.now().second)
args = parser.parse_args()
if args.cate_pair is not None:
    only_wanted_cate.append(args.cate_pair)
if not os.path.exists(save_path_0):
    os.makedirs(save_path_0)


## --------------------------------------------------------------------------------- ##
## ------------------------- the main function starts here ------------------------- ##

## load data
## results : 
##          nsd_img0, nsd_fea0 : all NSD images and features
##          things_img0, things_eeg_fea0 : all THINGS-EEG2 images and features

## __adjustable__ : path to data files
nsd_img0 = np.load("nsd_img.npy") 
nsd_fea0 = np.load("nsd_fea.npy")
print("got NSD data.")

things_img0 = np.load("things_img.npy")
things_eeg_fea0 = np.load("things_fea.npy")
print("got THINGS data.")

## do sth for each category pair and each scale
for i in range(len(scale)):
    for _j in only_wanted_cate:
        current_cate = _j
        log_message(f"Processing scale: {scale[i]}, category pair: {_j}")
        print("===================================")
        print("Processing scale: ", scale[i])
        print("Processing category pair: ", _j)

        # select data according to category, and randomly sample. (for NSD)
        nsd_keep_indices = np.array([(_ in dict0[_j][1]) for _ in dt1])
        if np.sum(nsd_keep_indices) > scale[i]:
            nsd_keep_indices2 = np.where(nsd_keep_indices)[0]
            selected_indices = np.random.choice(nsd_keep_indices2, size=scale[i], replace=False)
            nsd_keep_indices = np.zeros(nsd_keep_indices.shape[0], dtype=bool)
            nsd_keep_indices[selected_indices] = 1
        nsd_img = nsd_img0[nsd_keep_indices]
        nsd_fea = nsd_fea0[nsd_keep_indices]

        # select data according to category, and randomly sample. (for THINGS)
        wanted_indices1 = np.array([(_ in dict0[_j][0]) for _ in cluster_labels])
        if np.sum(wanted_indices1) > scale[i]:
            wanted_indices2 = np.where(wanted_indices1)[0]
            selected_indices = np.random.choice(wanted_indices2, size=scale[i], replace=False)
            wanted_indices1 = np.zeros(wanted_indices1.shape[0], dtype=bool)
            wanted_indices1[selected_indices] = 1
        things_img = things_img0[wanted_indices1, :, :, :]
        things_eeg_fea = things_eeg_fea0[wanted_indices1, :]

        nsd_fea = np.array(nsd_fea)
        things_eeg_fea = np.array(things_eeg_fea)

        # part 1 : show examples
        if show_examples == 1:
            folder_path = save_path_0 + f"show_examples/cate{current_cate}_THINGS_EEG/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            for _ in range(num_of_examples):
                ii = random.randint(0, things_img.shape[0] - 1)
                plt.imshow(things_img[ii, :, :, :])
                plt.axis('off')
                plt.savefig(folder_path + "THINGS_EEG_example_" + get_current_time_info() + "_" + str(ii) + ".png")
                plt.close()

            folder_path = save_path_0 + f"show_examples/cate{current_cate}_NSD/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            for _ in range(num_of_examples):
                ii = random.randint(0, nsd_img.shape[0] - 1)
                plt.imshow(nsd_img[ii, :, :, :])
                plt.axis('off')
                plt.savefig(folder_path + "NSD_example_" + get_current_time_info() + "_" + str(ii) + ".png")
                plt.close()

        ## part 2 : PCA and plot
        if do_plot == 1:
            pca = PCA(n_components=2)
            transformed_data = pca.fit_transform(np.concatenate([nsd_fea, things_eeg_fea], axis = 0))
            transformed_data1 = transformed_data[: nsd_fea.shape[0], :]
            transformed_data2 = transformed_data[nsd_fea.shape[0]: , :]
            scale_i = str(scale[i])
            try:
                folder_name11 = save_path_0 + "N-vs-E/"
                if not os.path.exists(folder_name11):
                    os.makedirs(folder_name11)
                plt.figure(figsize=(10, 8))
                plt.scatter(transformed_data1[:, 0], transformed_data1[:, 1], c="#1A53A1", alpha=0.6)
                plt.scatter(transformed_data2[:, 0], transformed_data2[:, 1], c="#B22222", alpha=0.6)
                plt.tight_layout()
                plt.grid(True)
                plt.savefig(folder_name11 + "N-vs-E_" + get_current_time_info() + "_n" + scale_i + "_cate"+ str(_j) + ".png", dpi = 1600)
                plt.close()
                print("Figure saved.")
            except Exception as e:
                print(f"Error saving figure: {e}")

        ## part 3 : E2N retrieval
        ## __adjustable__ : the cosine similarity threshold
        threshold = 0.87
        if do_e2n_retrieval == 1:
            print("start doing E2N retrieval")
            try: 
                t = get_current_time_info()
                folder_name = save_path_0 + f"E2N_retrieval/cate{_j}_{t}/"
                os.makedirs(save_path_0 + "E2N_retrieval/", exist_ok=True)
                os.makedirs(folder_name, exist_ok=True)
                pr11 = 1
                log_message(f"E2N_retrieval for scale {scale[i]}, category {_j} : {(len(things_img)) // pr11} attempts")
                cos_sim_hist = np.zeros(101)
                cca = 100
                ccb = 0
                print("things_img.shape : ", things_img.shape)
                for index in range(len(things_img)):
                    _ = index
                    img0 = things_img[index, :, :, :]
                    N = things_eeg_fea[index, :]
                    lN = np.sum(N ** 2) ** 0.5
                    lE = np.sum(nsd_fea ** 2, axis = 1) ** 0.5
                    cos_sim_score0 = np.sum(N[None, :] * nsd_fea / lN / lE[:, None], axis = 1)
                    rt_index = np.argmax(cos_sim_score0, axis = 0)
                    cos_sim_score_val = np.max(cos_sim_score0, axis = 0)
                    cos_sim_hist[max(min(int(cos_sim_score_val * cca + ccb), 100), 0)] += 1
                    if cos_sim_score_val < threshold:
                        continue
                    log_message(f"cos_sim score for example {_}: {cos_sim_score_val}") ## only log those above threshold
                    cos_sim_score_val = float(f"{cos_sim_score_val:.4f}")
                    img1 = nsd_img[rt_index, :, :, :]
                    plt.imshow(img0)
                    plt.axis('off')
                    plt.savefig(folder_name + f"E2N_{t}_n{scale_i}_{_}_E.png")
                    plt.close()
                    plt.imshow(img1)
                    plt.axis('off')
                    plt.savefig(folder_name + f"E2N_{t}_n{scale_i}_{_}_N.png")
                    plt.close()

                print("cos_sim_hist: ...")
                ## plot a histogram of cosine similarity
                if np.sum(cos_sim_hist) > 0:
                    plt.figure(figsize=(8, 4))
                    bins = np.arange(len(cos_sim_hist))
                    plt.bar(bins, cos_sim_hist)
                    tick_indices = bins[::5]
                    plt.xticks(tick_indices, [f"{((b - ccb) / cca):.2f}" for b in tick_indices], rotation=45)
                    plt.xlabel("cos_sim bins")
                    plt.ylabel("count")
                    plt.tight_layout()
                    plt.savefig(save_path_0 + f"cos_sim_hist_{t}.png")
                    plt.close()
                print("plotting...")
                plot(folder_name, save_path_0)
                print("E2N retrieval done.")
            except Exception as e:
                print(f"Error E2N: {e}")




