import os
import time
import itertools
import gin

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
import pydicom
import cv2

from data_utils import MimicID, MimicCxrMetadata


@gin.configurable
class MimicCxrDataset(torchvision.datasets.VisionDataset):
    """A MIMIC-CXR dataset class that loads dicom images from MIMIC-CXR 
    given a metadata file and return images in npy

    Args: 
        mimiccxr_dir (string): Root directory for the MIMIC-CXR dataset.
        mimiccxr_metadata (string): File path of the entire MIMIC-CXR metadata.
        dataset_metadata (string): File path of the metadata 
            that will be used to contstruct this dataset.
        transform (callable, optional): A function/tranform that takes in an image 
            and returns a transfprmed version.
    """

    def __init__(self, mimiccxr_dir, mimiccxr_metadata,
                 dataset_metadata, overlap_key='dicom_id', transform=None):
        super(MimicCxrDataset, self).__init__(root=None, transform=transform)

        self.mimiccxr_dir = mimiccxr_dir

        self.mimiccxr_metadata = MimicCxrMetadata(mimiccxr_metadata).get_sub_columns(
            ['subject_id', 'study_id', 'dicom_id'])
       
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

    def __getitem__(self, i):
        subject_id, study_id, dicom_id = \
            self.dataset_metadata.loc[i, ['subject_id', 'study_id', 'dicom_id']]
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


def save_png_images(img_size, save_folder, dataset_metadata, overlap_key='dicom_id', view_metadata=None):
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda img: img.astype(np.int32)),
        # PIL accepts in32, not uint16
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.Lambda(
            lambda img: np.array(img).astype(np.int32))
    ])
    mimiccxr_dataset = MimicCxrDataset(dataset_metadata=metadata,
                                       overlap_key=overlap_key,
                                       transform=transform)
    print(mimiccxr_dataset.__len__())
    if view_metadata != None:
        mimiccxr_dataset.select_by_column(view_metadata, 'view', ['frontal'])
    print(mimiccxr_dataset.__len__())
    mimiccxr_loader = DataLoader(mimiccxr_dataset, batch_size=1, shuffle=False,
                                 num_workers=1, pin_memory=True)

    for i, (img, dcm_exists, subject_id, study_id, dicom_id) in enumerate(mimiccxr_loader):
        if dcm_exists:
            img = img.cpu().numpy().astype(np.float)
            mimic_id = MimicID(subject_id[0], study_id[0], dicom_id[0])
            png_path = os.path.join(save_folder, f"{mimic_id.__str__()}.png")
            image = 65535*img[0]/np.amax(img[0])
            cv2.imwrite(png_path, image.astype(np.uint16))
            if i%1000==0:
                print(i)


def create_cached_images(img_size, save_folder):
    dataset = MimicCxrDataset(
        transform=torchvision.transforms.Compose([
            # unnormalized uint16 -> int32 (PIL accepts in32, not uint16)
            torchvision.transforms.Lambda(lambda img: img.astype(np.int32)),
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.CenterCrop(img_size),
            torchvision.transforms.Lambda(
                lambda img: np.array(img).astype(np.int32))
        ]))
    loader = DataLoader(dataset, batch_size=75, shuffle=False,
                        num_workers=15, pin_memory=False)

    imgs, dicom_ids_on_disk = [], []
    for it, (img, dicom_id, dcm_exists) in enumerate(loader):
        exists_id = torch.nonzero(dcm_exists)
        dicom_ids_on_disk.append([dicom_id[i] for i in exists_id])
        imgs.append(img[exists_id].squeeze().numpy().astype(np.uint16))

    dicom_ids_on_disk = list(itertools.chain.from_iterable(dicom_ids_on_disk))
    imgs = np.concatenate(imgs, axis=0)
    print(f'Cached a total of {imgs.shape[0]} Images')

    '''
    Save downsampled images 
    '''

    os.makedirs(save_folder, exist_ok=True)

    df = pd.DataFrame({'dicom_id': dicom_ids_on_disk,
                       'index_to_cached_array': list(range(len(dicom_ids_on_disk)))})
    df_path = os.path.join(save_folder, 
                          'dicom_id_to_cached_array_index_mapping.csv')
    df.to_csv(df_path, index=False)

    cache_name = os.path.join(save_folder, f"mimiccxr-{img_size}")
    np.save(f'{cache_name}.npy', imgs)

    return imgs, df


metadata = '/data/vision/polina/projects/chestxray/'\
           'work_space_v2/report_processing/edema_labels-12-03-2019/'\
           'mimic-cxr-sub-img-edema-split-manualtest.csv'

# for im_size in [256]:
#     start = time.time()
#     create_cached_images(
#         im_size,
#         '/data/vision/polina/scratch/wpq/github/interpretability/notebooks/data/MimicCxrChexpertDataset')
#     end = time.time()
#     print(f'im_size: {im_size} done!')