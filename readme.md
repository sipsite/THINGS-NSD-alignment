
only everything under folder 12.24 is needed.

## before running : 
1. put eeg signal folder, "Preprocessed_data_250Hz_whiten", under "things-eeg-data" folder
2. put things images folder "training_images" and "image_metadata.npy" under "things-eeg-data" folder
3. put all nsd data directly under "nsd-data" folder
4. put fMRI embeddings folder, "fMRI_embeddings", under root directory(i.e. 12.24)

plz ignore the below.
----

A simple guide (updated on 12.22): 

1. run `data_processing.py` : get CLIP embedding
2. run `12.21_utensils/feature_extraction.py` : get other embeddings
3. run `12.21_utensils/similarity_functions.py` : do searching and get top5 results (20~40 min approx)
4. run `12.21_utensils/rearange_json.py` : filter json file based on selected threshold
5. run `12.21_utensils/show_examples.py` : show examples of each group
6. run `12.21_utensils/from_json_load_data.py` : load fMRI and EEG signals based on json file

-----
readme (old version)

# THINGS-NSD Images Alignment

this project does the following : 
- get CLIP embedding features for THINGS and NSD images
- do clustering for THINGS
- select image samples using manual category alignment and random sampling
- do PCA and plot the result
- do NSD-to-THINGS retrieval based on cosine-simliarity.

below are steps to run the code.

### 1 getting dataset

NSD dataset : 
- use the same data as [MindEye2](https://github.com/MedARC-AI/MindEyeV2?tab=readme-ov-file)
- In the github page, follow `The below code will download ...` to download images and clustering labels

THINGS dataset : 
- use THINGS-EEG2
- download images at its website [OSF](https://osf.io/y63gw/overview)

### 2 using the code
You can run the codes in the following order :
1. run `data_processing.py` to process the whole THINGS and NSD dataset, saving images and (CLIP embedding) features as .npy
2. run `clustering_things.py` to cluster THINGS and save labels as .npy
3. run `pca_and_retri.py` to show examples, do PCA and plot, and do THINGS-to-NSD retrieval

Remark : 
- in the codes, comments `__adjustable__` denote adjustable parameters or path placeholders (for quick searching)

### 3 dataset info

NSD : 
- 73,000 images
- 41 clustering labels (corresponding to "new-ver" in this project)

THINGS-EEG2 : 
- 16,540 (training) images
- 48 clustering labels and 1 "unclassified" label (corresponding to "new-ver" in this project)
- 27 category labels in THINGS's original categorization (remark: nearly half of the images are unlabeled) (corresponding to "old-ver" in this project)
