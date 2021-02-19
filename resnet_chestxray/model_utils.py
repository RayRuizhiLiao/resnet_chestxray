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
import cv2

from .utils import MimicID

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
                 data_key='mimic_id', label_key='edema_severity',
    			 transform=None, cache=False):
        super(CXRImageDataset, self).__init__(root=None, transform=transform)
        self.data_dir = data_dir
        self.dataset_metadata = pd.read_csv(dataset_metadata)
        self.data_key = data_key
        self.label_key = label_key
        self.transform = transform
        self.image_ids = self.dataset_metadata[data_key]
        self.select_valid_labels()
        self.cache = cache
        if self.cache:
            self.cache_dataset() 
        else:
            self.images = None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id, label = self.dataset_metadata.loc[idx, [self.data_key, self.label_key]]

        if self.cache:
            img = self.images[str(idx)]
        else:
            png_path = os.path.join(self.data_dir, f'{img_id}.png')
            img = cv2.imread(png_path, cv2.IMREAD_ANYDEPTH)

        if self.transform is not None:
            img = self.transform(img)

        img = np.expand_dims(img, axis=0)

        return img, label, img_id

    def select_valid_labels(self):
        self.dataset_metadata = self.dataset_metadata[self.dataset_metadata[self.label_key]>=0]
        self.dataset_metadata = self.dataset_metadata.reset_index(drop=True)
        self.image_ids = self.dataset_metadata[self.data_key]

    def cache_dataset(self):
        for idx in range(self.__len__()):
            img_id, label = self.dataset_metadata.loc[idx, [self.data_key, self.label_key]]
            png_path = os.path.join(self.data_dir, f'{img_id}.png')
            img = cv2.imread(png_path, cv2.IMREAD_ANYDEPTH)
            if idx == 0:
                self.images = {}
            self.images[str(idx)] = img

    @staticmethod
    def create_dataset_metadata(mimiccxr_metadata, label_metadata, save_path,
                                data_key='study_id', label_key=['edema_severity'],
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

        label_metadata = label_metadata[[data_key]+label_key]
        dataset_metadata = dataset_metadata.merge(label_metadata, left_on=data_key, right_on=data_key)

        dataset_metadata['mimic_id'] = dataset_metadata.apply(lambda row: \
            MimicID(row['subject_id'], row['study_id'], row['dicom_id']).__str__(), axis=1)
        dataset_metadata = dataset_metadata[['mimic_id']+label_key]

        dataset_metadata.to_csv(save_path, index=False)
