'''
Author: Ruizhi Liao

Model_utils script to support
residual network model instantiation
'''

import csv
import os
import numpy as np
from math import floor, ceil
import scipy.ndimage as ndimage
from skimage import io
import pandas as pd
import gin

from utils import MimicID

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
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

class CXRImageDataset(torchvision.datasets.VisionDataset):
    """A CXR iamge dataset class that loads png images 
    given a metadata file and return images and labels 

    Args:
        data_dir (string): Root directory for the CXR images.
        dataset_metadata (string): File path of the metadata 
            that will be used to contstruct this dataset. 
            This metadata file should contain data IDs that are used to
            load images and labels associated with data IDs.
        data_key (string): The name of the column that has image IDs.
        label_key (string): The name of the column that has labels.
        transform (callable, optional): A function/tranform that takes in an image 
            and returns a transfprmed version.
    """
    
    def __init__(self, data_dir, dataset_metadata, 
                 data_key='mimic_id', label_key='label',
    			 transform=None):
        self.data_dir = data_dir
        self.dataset_metadata = pd.read_csv(dataset_metadata)
        self.data_key = data_key
        self.label_key = label_key
        self.transform = transform
        self.image_ids = self.dataset_metadata[data_key]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        images_ids, labels = self.dataset_metadata.loc[idx, [self.data_key, self.label_key]]
        dcm_path = os.path.join(
            self.mimiccxr_dir, f'p{subject_id}', f's{study_id}', f'{dicom_id}.dcm')

        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # img_id = list(self.img_ids.keys())[idx]
        # img_path = os.path.join(self.root_dir,
        #                         img_id+'.'+self.image_format)
        # image = load_image(img_path)
        # if self.transform:
        #     image = self.transform(image)
        # image = image.reshape(1, image.shape[0], image.shape[1])
        
        # label = self.labels[img_id]
        # label_raw = label[0]
        # label = convert_to_onehot(label[0])
        # label = torch.tensor(label, dtype=torch.float32)
        # label_raw = torch.tensor(label_raw, dtype=torch.long)

        sample = [images_ids, labels]

        return sample

    @staticmethod
    @gin.configurable
    def create_dataset_metadata(mimiccxr_metadata, label_metadata, save_path,
                                data_key='study_id', label_key='edema_severity',
                                mimiccxr_selection={'view': ['frontal']},
                                holdout_metadata=None, holdout_key='subject_id'):
        """Create a dataset metadata file for CXRImageDataset 
        given a MIMIC-CXR metadata file and a label metadata file.
        """

        mimiccxr_metadata = pd.read_csv(mimiccxr_metadata)
        label_metadata = pd.read_csv(label_metadata)

        dataset_metadata = mimiccxr_metadata[mimiccxr_metadata[data_key].isin(label_metadata[data_key])]

        if mimiccxr_selection != None:
            for key in mimiccxr_selection:
                dataset_metadata = dataset_metadata[dataset_metadata[key].isin(mimiccxr_selection[key])]

        if holdout_metadata != None:
            holdout_metadata = pd.read_csv(holdout_metadata)
            dataset_metadata = dataset_metadata[~dataset_metadata[holdout_key].isin(holdout_metadata[holdout_key])]

        label_metadata = label_metadata[[data_key, label_key]]
        dataset_metadata = dataset_metadata.merge(label_metadata, left_on=data_key, right_on=data_key)

        dataset_metadata['mimic_id'] = dataset_metadata.apply(lambda row: \
            MimicID(row['subject_id'], row['study_id'], row['dicom_id']).__str__(), axis=1)
        dataset_metadata = dataset_metadata[['mimic_id', label_key]]

        dataset_metadata.to_csv(save_path, index=False)


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
