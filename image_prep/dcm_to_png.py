import os
import numpy as np
import pandas as pd
import pydicom
import cv2

import torch
import torchvision
from torch.utils.data import DataLoader

from utils import MimicID


current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)


class MimicCxrMetadata:

    def __init__(self, mimiccxr_metadata):
        self.mimiccxr_metadata = pd.read_csv(mimiccxr_metadata)

    def get_sub_columns(self, columns: list):
        return self.mimiccxr_metadata[columns]

    def get_sub_rows(self, column: str, values: list):
        return self.mimiccxr_metadata[self.mimiccxr_metadata[column].isin(values)]

    @staticmethod
    def overlap_by_column(metadata1, metadata2, column: str):
        return metadata1[metadata1[column].isin(metadata2[column])]     


class MimicCxrDataset(torchvision.datasets.VisionDataset):
    """A MIMIC-CXR dataset class that loads dicom images from MIMIC-CXR 
    given a metadata file and return images in npy

    Args: 
        mimiccxr_dir (string): Root directory for the MIMIC-CXR dataset.
        mimiccxr_metadata (string): File path of the entire MIMIC-CXR metadata.
        dataset_metadata (string): File path of the metadata 
            that will be used to contstruct this dataset.
        overlap_key (string): The name of the column that will be used to find overlap 
            between mimiccxr_metadata and dataset_metadata. 
        transform (callable, optional): A function/tranform that takes in an image 
            and returns a transfprmed version.
    """

    mimiccxr_dir = '/data/vision/polina/projects/chestxray/data_v2/dicom_reports/'
    mimiccxr_metadata_path = os.path.join(parent_dir, 
                                          'mimic_cxr_edema/auxiliary_metadata/'\
                                          'mimic_cxr_metadata_available_CHF_view.csv')

    def __init__(self, dataset_metadata, overlap_key='dicom_id', transform=None):
        super(MimicCxrDataset, self).__init__(root=None, transform=transform)

        self.mimiccxr_metadata = MimicCxrMetadata(self.mimiccxr_metadata_path).\
            get_sub_columns(['subject_id', 'study_id', 'dicom_id'])
       
        dataset_ids = MimicCxrMetadata(dataset_metadata).get_sub_columns(
            [overlap_key])
        self.dataset_metadata = MimicCxrMetadata.overlap_by_column(
            self.mimiccxr_metadata, dataset_ids, overlap_key).reset_index(drop=True)
        
        self.transform = transform

    def __len__(self):
        return len(self.dataset_metadata)

    def select_by_column(self, metadata: str, column: str, values: list):
        metadata_selected = MimicCxrMetadata(metadata).get_sub_rows(column = column, values=values)
        self.dataset_metadata = MimicCxrMetadata.overlap_by_column(
            self.dataset_metadata, metadata_selected, 'dicom_id').reset_index(drop=True)

    def __getitem__(self, idx):
        subject_id, study_id, dicom_id = \
            self.dataset_metadata.loc[idx, ['subject_id', 'study_id', 'dicom_id']]
        dcm_path = os.path.join(
            self.mimiccxr_dir, f'p{subject_id}', f's{study_id}', f'{dicom_id}.dcm')
        
        if os.path.isfile(dcm_path):
            dcm = pydicom.dcmread(dcm_path)
            img = dcm.pixel_array
            dcm_exists = True
        else:
            img = -1
            dcm_exists = False
        
        if self.transform is not None:
            img = self.transform(img)

        return img, dcm_exists, str(subject_id), str(study_id), str(dicom_id)


def frontal_dcm_to_png(img_size, save_folder, dataset_metadata, 
                       overlap_key='dicom_id', view_metadata=None):
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda img: img.astype(np.int32)),
        # PIL accepts in32, not uint16
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.Lambda(
            lambda img: np.array(img).astype(np.int32))
    ])
    mimiccxr_dataset = MimicCxrDataset(dataset_metadata=dataset_metadata,
                                       overlap_key=overlap_key,
                                       transform=transform)
    print(f'Total number of images: {mimiccxr_dataset.__len__()}')
    if view_metadata != None:
        # Select frontal view images
        mimiccxr_dataset.select_by_column(view_metadata, 'view', ['frontal'])
    print(f'Total number of frontal view images: {mimiccxr_dataset.__len__()}')
    mimiccxr_loader = DataLoader(mimiccxr_dataset, batch_size=1, shuffle=False,
                                 num_workers=1, pin_memory=True)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    num_notexist = 0
    num_exist = 0
    for i, (img, dcm_exists, subject_id, study_id, dicom_id) in enumerate(mimiccxr_loader):
        if dcm_exists:
            img = img.cpu().numpy().astype(np.float)
            mimic_id = MimicID(subject_id[0], study_id[0], dicom_id[0])
            png_path = os.path.join(save_folder, f"{mimic_id.__str__()}.png")
            image = 65535*img[0]/np.amax(img[0])
            cv2.imwrite(png_path, image.astype(np.uint16))
            num_exist+=1
            if num_exist%1000==0:
                print(f'{num_exist} images saved!')
        else:
            num_notexist+=1
    print(f'{num_exist} frontal view images saved!')
    print(f'{num_notexist} frontal view images do not exist!')


metadata = os.path.join(parent_dir,
                        'mimic_cxr_edema/regex_report_edema_severity.csv')
#metadata = os.path.join(parent_dir,
#                        'mimic_cxr_edema/auxiliary_metadata/mimic_cxr_metadata_available_CHF_view.csv')
view_metadata = os.path.join(parent_dir,
                             'mimic_cxr_edema/auxiliary_metadata/mimic_cxr_metadata_available_CHF_view.csv')
save_dir = '/data/vision/polina/scratch/ruizhi/chestxray/data/png_16bit_512/'
frontal_dcm_to_png(img_size=512, save_folder=save_dir, overlap_key='study_id',
                   dataset_metadata=metadata, view_metadata=view_metadata)