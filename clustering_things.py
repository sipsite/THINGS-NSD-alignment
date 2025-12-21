
## this file : clustering THINGS-EEG2


from utils import plot

import os
import matplotlib.pyplot as plt
import datetime
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import math

def get_current_time_info():
    now = datetime.datetime.now()
    standard_format = now.strftime("%m-%d_%H-%M-%S")
    return standard_format



## __adjustable__ : the save path (dir)
save_path = "THINGS_cluster_" + get_current_time_info() + "/"
if not os.path.exists(save_path):
    os.makedirs(save_path)


## load THINGS images and features
## __adjustable__
from data_processing import save_path as data_save_path
img_path = data_save_path + "xxx.npy"
img = np.load(img_path)
load_path = data_save_path + "xxx.npy"
data = np.load(load_path)
print("data.shape : ", data.shape)



## --------------------------------------------------------------------------------- ##
## __adjustable__ : choose clustering method
## 1 : Faiss Kmeans ; 2 : PCA + KMeans ; 3 : UMAP + HDBSCAN. 
## 3 is recommended.
method_type = 3  

print("start clustering ...")
# type 1 : Faiss
n_centroids = 45  
n_iter = 20       
if method_type == 1 :
    use_gpu = True
    kmeans = faiss.Kmeans(data.shape[1], n_centroids, niter=n_iter, verbose=True, gpu=use_gpu)
    kmeans.train(data)

    D, I = kmeans.index.search(data, 1) 
    print(I[:5]) 
    np.save(save_path + get_current_time_info() + "_labels.npy", I)


# type 2 : PCA + KMeans
num_examples = 20
if method_type == 2 :
    ## either do clustering or load clustering results
    if False:
        # Step 1: PCA 
        pca = PCA(n_components=50)
        data_reduced = pca.fit_transform(data)

        # Step 2: K-Means clustering
        kmeans = KMeans(n_clusters=n_centroids, random_state=42)
        labels = kmeans.fit_predict(data_reduced)
        count_labels = np.sum(labels[:, None] == np.arange(n_centroids)[None, :], axis = 0)
        print(count_labels)
        np.save(save_path + get_current_time_info() + "_labels.npy", labels)
    elif True:
        load_path2 = "xxx.npy"
        labels = np.load(load_path2)
        print("labels.shape : ", labels.shape)

    ## show examples from each cluster
    if True:
        for _ in range(n_centroids):
            print("Cluster ", _, ": ")
            indices = np.where(labels == _)[0]
            folderr_name = f"clusters_{_ + 1}__{get_current_time_info()}/"
            folder_path = save_path + folderr_name
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            if len(indices) > num_examples:
                indices = np.random.choice(indices, num_examples, replace=False)
            for i, idx in enumerate(indices):
                plt.imshow(img[idx, :, :, :])
                plt.axis('off')
                plt.savefig(folder_path + f"cluster{_ + 1}_" + "_" + str(i + 1) + ".png")
                plt.close()
            plot(folder_path)



## type 3 : UMAP + HDBSCAN
import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
if method_type == 3 : 

    ## either do clustering or load clustering results
    if False:
        data_scaled = StandardScaler().fit_transform(data)
        print("步骤 1: UMAP 降维中 (可能需要几十秒)...")
        reducer = umap.UMAP(
            n_neighbors=30,      
            n_components=2,      
            min_dist=0.0,        
            metric='euclidean',  
            random_state=42
        )
        embedding = reducer.fit_transform(data_scaled)


        print("步骤 2: HDBSCAN 聚类中...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=100,
            min_samples=10,
            gen_min_span_tree=True 
        )
        cluster_labels = clusterer.fit_predict(embedding)

        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        num_noise = list(cluster_labels).count(-1)

        print(f"\n--- 聚类结果 ---")
        print(f"发现簇的数量: {num_clusters}")
        print(f"被标记为噪声的点 (Label -1): {num_noise} (占比 {num_noise/len(data):.2%})")

        print("正在绘图...")
        plt.figure(figsize=(10, 8))
        clustered = (cluster_labels >= 0)
        plt.scatter(embedding[~clustered, 0], embedding[~clustered, 1], c=(0.5, 0.5, 0.5), s=1, alpha=0.3, label='Noise')
        plt.scatter(embedding[clustered, 0], embedding[clustered, 1], c=cluster_labels[clustered], cmap='Spectral', s=1, alpha=0.5)
        plt.title(f'UMAP + HDBSCAN Clustering (Found {num_clusters} clusters)')
        plt.legend(markerscale=5)
        plt.savefig(f"visualize_clusters" + get_current_time_info() +".png")
        plt.close()

        count_labels = np.sum(cluster_labels[:, None] == np.arange(num_clusters)[None, :], axis = 0)
        print(count_labels)
        np.save(save_path + get_current_time_info() + "_labels_type3.npy", cluster_labels)
    else:
        load_path2 = save_path + "xxx.npy"
        cluster_labels = np.load(load_path2)
        print("cluster_labels.shape : ", cluster_labels.shape)
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)


    ## show examples from each cluster
    if True:
        for _ in range(-1, num_clusters):
            print("Cluster ", _, ": ")
            indices = np.where(cluster_labels == _)[0]
            folderr_name = f"clusters_{_ + 1}_type3_{get_current_time_info()}/"
            folder_path = save_path + folderr_name
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            if len(indices) > num_examples:
                indices = np.random.choice(indices, num_examples, replace=False)
            for i, idx in enumerate(indices):
                plt.imshow(img[idx, :, :, :])
                plt.axis('off')
                plt.savefig(folder_path + f"cluster{_ + 1}_" + "_" + str(i + 1) + ".png")
                plt.close()
            plot(folder_path)
