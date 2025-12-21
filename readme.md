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
