import os
from matplotlib import pyplot as plt
import numpy as np
import glob
import pandas as pd
import random
import cv2
from sklearn.utils import shuffle

#create custom color maps
cdict1 = {'red':   ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'green': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

plt.register_cmap(name='greens', data=cdict1)

# number of images in batch
batch_size = 32
# models are saved here
ckpt_dir = './checkpoint'
# samples are saved here
sample_dir = './sample'


train_set = 'train'

#import training data
train_labels = pd.read_csv("./dataset/kaggle_protein_classes_augmented_one_hot.csv")
train_labels = train_labels.sort_values(by=['Id']).reset_index()
print(train_labels.head())
print(len(train_labels))

seed = 100
train_labels = shuffle(train_labels, random_state=seed).reset_index(drop=True)
print(train_labels.head())
print(len(train_labels))

datay = train_labels.loc[1001:10000]

datay = datay.iloc[:, 4:]
# get image id
data_im_id = train_labels.loc[1001:10000, "Id"]


# read train data files
data_files = []
for im_id in data_im_id:
    data_files.append(glob.glob('./dataset/train/{}_green.png'.format(im_id)))
#data_files.sort()
filepaths = [''.join(x) for x in data_files]

print("number of training data x", len(filepaths))

# data augmentation options
def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 3:
        # flip left and right
        return np.flipud(image)

# data augmentation xDATA_AUG_TIMES times
DATA_AUG_TIMES = 1
count = 0

# calculate the number of patches
step = 0
pat_size = 300
stride = 300

count = len(filepaths)

origin_patch_num = count * DATA_AUG_TIMES
print('origin_patch_num', origin_patch_num)
numClasses = 28
bat_size = 32
# use power of 2 bat_size ex 128

# calculate number of batches and patches
# must be whole number
if origin_patch_num % bat_size != 0:
    numPatches = (divmod(origin_patch_num, bat_size)[0]+1) * bat_size

else:
    numPatches = origin_patch_num
print("total patches =", numPatches, ", batch size =", bat_size,
      ", total batches =", numPatches / bat_size)

# data matrix 4-D
numPatches = int(numPatches)
inputs = np.zeros((numPatches, pat_size, pat_size, 1), dtype="uint8")
inputsy = np.zeros((numPatches, numClasses), dtype="uint8")

count = 0

# generate patches
for i in range(len(filepaths)):
    # get image id
    im_id = data_im_id.iloc[i]

    # open x
    img_s = cv2.imread('./dataset/train/{}_green.png'.format(im_id), 0)


    # open y
    imgy =  datay.iloc[i]


    img_sy = np.array(imgy, dtype="uint8")

    # data augmentation
    for j in range(DATA_AUG_TIMES):
        im_h, im_w= img_s.shape



        z = random.randint(0, 212)
        inputs[count, :, :, 0] = data_augmentation(
            img_s[z:z + pat_size, z:z + pat_size], random.randint(0, 3))

        inputsy[count, :] = img_sy

        count += 1

#imgplot = plt.imshow(inputs[count-9, :, :,0] )
#plt.show()

# pad training examples into the empty patches of the last batch
if count < numPatches:
    to_pad = numPatches - count
    inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]
    inputsy[-to_pad:, :] = inputsy[:to_pad, :]



#imgplot = plt.imshow(inputs[count-9, :, :,0] )
#plt.show()


# directory of patches
save_dir='./dataset/transformed_data/'


# save x
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
np.save(os.path.join(save_dir,
                     "protein_image_patches_pat_size_300_bat_size_32_1001_10000"),
        inputs)
print("size of x inputs tensor = ", str(inputs.shape))

# save y

np.save(os.path.join(save_dir,
                     "protein_image_classes_pat_size_300_bat_size_32_1001_10000"),
        inputsy)
print("size of y inputs tensor = ", str(inputsy.shape))

### up to here it generates patches
