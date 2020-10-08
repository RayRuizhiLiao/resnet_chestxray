import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import csv
import os
import numpy as np
from math import floor, ceil
import scipy.ndimage as ndimage
from skimage import io

from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F


# Convert edema severity to ordinal encoding
def convert_to_ordinal(severity):
    if severity == 0:
        return [0,0,0]
    elif severity == 1:
        return [1,0,0]
    elif severity == 2:
        return [1,1,0]
    elif severity == 3:
        return [1,1,1]
    else:
        raise Exception("No other possibilities of ordinal labels are possible")

# Convert edema severity to one-hot encoding
def convert_to_onehot(severity):
    if severity == 0:
        return [1,0,0,0]
    elif severity == 1:
        return [0,1,0,0]
    elif severity == 2:
        return [0,0,1,0]
    elif severity == 3:
        return [0,0,0,1]
    else:
        raise Exception("No other possibilities of ordinal labels are possible")

# Load an .npy or .png image 
def load_image(img_path):
    if img_path[-3:] == 'npy':
        image = np.load(img_path)
    if img_path[-3:] == 'png':
        image = io.imread(img_path)
        image = image.astype(np.float32)
        image = image/np.max(image)
    return image

# Read the images and store them in the memory
def read_images(img_ids, root_dir):
    images = {}
    for img_id in list(img_ids.keys()):
        img_path = os.path.join(root_dir, img_id+'.npy')
        image = np.load(img_path)
        images[img_id] = image
    return images


#Customizing dataset class for chest xray images
class CXRImageDataset(Dataset):
    
    def __init__(self, img_ids, labels, root_dir, 
    			 transform=None, image_format='png',
    			 encoding='ordinal'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.img_ids = img_ids
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
        self.image_format = image_format
        self.encoding = encoding

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = list(self.img_ids.keys())[idx]
        img_path = os.path.join(self.root_dir,
                                img_id+'.'+self.image_format)
        image = load_image(img_path)
        if self.transform:
            image = self.transform(image)
        image = image.reshape(1, image.shape[0], image.shape[1])
        
        label = self.labels[img_id]
        if self.encoding == 'ordinal':
        	label = convert_to_ordinal(label[0])
        if self.encoding == 'onehot':
        	label = convert_to_onehot(label[0])
        label = torch.tensor(label, dtype=torch.float32)

        sample = [image, label]

        return sample


class CenterCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. 
        If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):

        image = self.__pad_2Dimage(image)
        h, w = image.shape[0:2]
        new_h, new_w = self.output_size

        if new_h>h or new_w>w:
            raise ValueError('This image needs to be padded!')

        top = floor((h - new_h) / 2)
        down = top + new_h
        left = floor((w - new_w) / 2)
        right = left + new_w
        
        return image[top:down, left:right]
    
    def __pad_2Dimage(self, image):
        'Pad 2D images to match output_size'
        h, w = image.shape[0:2]
        h_output, w_output = self.output_size[0:2]

        pad_h_length = max(0, float(h_output - h))
        pad_h_length_1 = floor(pad_h_length / 2) + 4  # 4 is extra padding
        pad_h_length_2 = floor(pad_h_length / 2) + 4  # 4 is extra padding

        pad_w_length = max(0, float(w_output - w))
        pad_w_length_1 = floor(pad_w_length / 2) + 4  # 4 is extra padding
        pad_w_length_2 = floor(pad_w_length / 2) + 4  # 4 is extra padding

        image = np.pad(image, ((pad_h_length_1, pad_h_length_2), (pad_w_length_1, pad_w_length_2)),
                       'constant', constant_values=((0, 0), (0, 0)))

        return image


class RandomTranslateCrop(object):
    """Translate, rotate and crop the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. 
        If int, square crop is made.
    """

    def __init__(self, output_size, shift_mean=0,
                 shift_std=200, rotation_mean=0, rotation_std=20):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.shift_mean = shift_mean
        self.shift_std = shift_std
        self.rotation_mean = rotation_mean
        self.rotation_std = rotation_std

    def __call__(self, image):

        image = self.__translate_2Dimage(image)
        #image = self.__rotate_2Dimage(image)
        h, w = image.shape[0:2]
        new_h, new_w = self.output_size

        if new_h>h or new_w>w:
            raise ValueError('This image needs to be padded!')

        top = floor((h - new_h) / 2)
        down = top + new_h
        left = floor((w - new_w) / 2)
        right = left + new_w
        
        return image[top:down, left:right]

    def __translate_2Dimage(self, image):
        'Translate 2D images as data augmentation'
        h, w = image.shape[0:2]
        h_output, w_output = self.output_size[0:2]

        # Generate random Gaussian numbers for image shift as data augmentation
        shift_h = int(np.random.normal(self.shift_mean, self.shift_std))
        shift_w = int(np.random.normal(self.shift_mean, self.shift_std))
        if abs(shift_h) > 2 * self.shift_std:
            shift_h = 0
        if abs(shift_w) > 2 * self.shift_std:
            shift_w = 0

        # Pad the 2D image
        pad_h_length = max(0, float(h_output - h))
        pad_h_length_1 = floor(pad_h_length / 2) + 4  # 4 is extra padding
        pad_h_length_2 = floor(pad_h_length / 2) + 4  # 4 is extra padding
        pad_h_length_1 = pad_h_length_1 + max(shift_h , 0)
        pad_h_length_2 = pad_h_length_2 + max(-shift_h , 0)

        pad_w_length = max(0, float(w_output - w))
        pad_w_length_1 = floor(pad_w_length / 2) + 4  # 4 is extra padding
        pad_w_length_2 = floor(pad_w_length / 2) + 4  # 4 is extra padding
        pad_w_length_1 = pad_w_length_1 + max(shift_w , 0)
        pad_w_length_2 = pad_w_length_2 + max(-shift_w , 0)

        image = np.pad(image, ((pad_h_length_1, pad_h_length_2), (pad_w_length_1, pad_w_length_2)),
                       'constant', constant_values=((0, 0), (0, 0)))

        return image


# def _split_tr_val(split_list_path, training_folds, validation_folds):
#     """Extracting finding labels
#     """
#     print('split_list_path: ', split_list_path)

#     train_labels = {}
#     train_ids = {}
#     val_labels = {}
#     val_ids = {}

#     with open(split_list_path, 'r') as train_label_file:
#         train_label_file_reader = csv.reader(train_label_file)
#         row = next(train_label_file_reader)
#         for row in train_label_file_reader:
#             if row[-1] != 'TEST':
#                 if int(row[-1]) in training_folds:
#                     train_labels[row[2]] = [float(row[3])]
#                     train_ids[row[2]] = row[1]
#                 if int(row[-1]) in validation_folds:
#                     val_labels[row[2]] = [float(row[3])]
#                     val_ids[row[2]] = row[1]

#     return train_labels, train_ids, val_labels, val_ids


# Given a data split list (.csv), training folds and validation folds,
# return DICOM IDs and the associated labels for training and validation
def _split_tr_val(split_list_path, training_folds, validation_folds, use_test_data=False):
    """Extracting finding labels
    """

    print('Data split list being used: ', split_list_path)

    train_labels = {}
    train_ids = {}
    val_labels = {}
    val_ids = {}
    test_labels = {}
    test_ids = {}


    with open(split_list_path, 'r') as train_label_file:
        train_label_file_reader = csv.reader(train_label_file)
        row = next(train_label_file_reader)
        for row in train_label_file_reader:
            if row[-1] != 'TEST':
                if int(row[-1]) in training_folds:
                    train_labels[row[2]] = [float(row[3])]
                    train_ids[row[2]] = row[1]
                if int(row[-1]) in validation_folds and not use_test_data:
                    val_labels[row[2]] = [float(row[3])]
                    val_ids[row[2]] = row[1]
            if row[-1] == 'TEST' and use_test_data:
                    test_labels[row[2]] = [float(row[3])]
                    test_ids[row[2]] = row[1]               

    print("Training and validation folds: ", training_folds, validation_folds)
    print("Total number of training labels: ", len(train_labels))
    print("Total number of training DICOM IDs: ", len(train_ids))
    print("Total number of validation labels: ", len(val_labels))
    print("Total number of validation DICOM IDs: ", len(val_ids))
    print("Total number of test labels: ", len(test_labels))
    print("Total number of test DICOM IDs: ", len(test_ids))

    if use_test_data:
        return train_labels, train_ids, test_labels, test_ids
    else:
        return train_labels, train_ids, val_labels, val_ids