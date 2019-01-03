# Human-Protein-Atlas-Image-Classification
Human Protein Atlas Image Classification - SENet architecture

Competition URL: https://www.kaggle.com/c/human-protein-atlas-image-classification#description

## Description
In this competition, Kagglers will develop models capable of classifying mixed patterns of proteins in microscope images. The Human Protein Atlas will use these models to build a tool integrated with their smart-microscopy system to identify a protein's location(s) from a high-throughput image.

Proteins are “the doers” in the human cell, executing many functions that together enable life. Historically, classification of proteins has been limited to single patterns in one or a few cell types, but in order to fully understand the complexity of the human cell, models must classify mixed patterns across a range of different human cells.

Images visualizing proteins in cells are commonly used for biomedical research, and these cells could hold the key for the next breakthrough in medicine. However, thanks to advances in high-throughput microscopy, these images are generated at a far greater pace than what can be manually evaluated. Therefore, the need is greater than ever for automating biomedical image analysis to accelerate the understanding of human cells and disease.

## Evaluation
Submissions will be evaluated based on their macro F1 score.

Submission File
For each Id in the test set, you must predict a class for the Target variable as described in the data page. Note that multiple labels can be predicted for each sample.

The file should contain a header and have the following format:

Id,Predicted  
00008af0-bad0-11e8-b2b8-ac1f6b6435d0,0 1  
0000a892-bacf-11e8-b2b8-ac1f6b6435d0,2 3
0006faa6-bac7-11e8-b2b7-ac1f6b6435d0,0  
0008baca-bad7-11e8-b2b9-ac1f6b6435d0,0  
000cce7e-bad4-11e8-b2b8-ac1f6b6435d0,0  
00109f6a-bac8-11e8-b2b7-ac1f6b6435d0,1 28  
...

## Dataset 
What files do I need?
You will need to download a copy of the images. Due to size, we have provided two versions of the same images. On the data page below, you will find a scaled set of 512x512 PNG files in train.zip and test.zip. Alternatively, if you wish to work with full size original images (a mix of 2048x2048 and 3072x3072 TIFF files) you may download train_full_size.7z and test_full_size.7z from here (warning: these are ~250 GB total).

You will also need the training labels from train.csv and the filenames for the test set from sample_submission.csv.

What should I expect the data format to be?
The data format is two-fold - first, the labels are provided for each sample in train.csv.

The bulk of the data is in the images - train.zip and test.zip. Within each of these is a folder containing four files per sample. Each file represents a different filter on the subcellular protein patterns represented by the sample. The format should be [filename]_[filter color].png for the PNG files, and [filename]_[filter color].tif for the TIFF files.

What am I predicting?
You are predicting protein organelle localization labels for each sample. There are in total 28 different labels present in the dataset. The dataset is acquired in a highly standardized way using one imaging modality (confocal microscopy). However, the dataset comprises 27 different cell types of highly different morphology, which affect the protein patterns of the different organelles. All image samples are represented by four filters (stored as individual files), the protein of interest (green) plus three cellular landmarks: nucleus (blue), microtubules (red), endoplasmic reticulum (yellow). The green filter should hence be used to predict the label, and the other filters are used as references.

The labels are represented as integers that map to the following:

0.  Nucleoplasm  
1.  Nuclear membrane   
2.  Nucleoli   
3.  Nucleoli fibrillar center   
4.  Nuclear speckles   
5.  Nuclear bodies   
6.  Endoplasmic reticulum   
7.  Golgi apparatus   
8.  Peroxisomes   
9.  Endosomes   
10.  Lysosomes   
11.  Intermediate filaments   
12.  Actin filaments   
13.  Focal adhesion sites   
14.  Microtubules   
15.  Microtubule ends   
16.  Cytokinetic bridge   
17.  Mitotic spindle   
18.  Microtubule organizing center   
19.  Centrosome   
20.  Lipid droplets   
21.  Plasma membrane   
22.  Cell junctions   
23.  Mitochondria   
24.  Aggresome   
25.  Cytosol   
26.  Cytoplasmic bodies   
27.  Rods & rings  

## File descriptions
1. train.csv - filenames and labels for the training set.
2. sample_submission.csv - filenames for the test set, and a guide to constructing a working submission.
3. train.zip - All images for the training set.
4. test.zip - All images for the test set.

## Data fields
1. Id - the base filename of the sample. As noted above all samples consist of four files - blue, green, red, and yellow.
2. Target - in the training data, this represents the labels assigned to each sample.
