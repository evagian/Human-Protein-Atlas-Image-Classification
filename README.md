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

