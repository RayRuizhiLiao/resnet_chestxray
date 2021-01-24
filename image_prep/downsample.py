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

from data_utils import MimicCxrMetadata


@gin.configurable
class MimicCxrDataset(torchvision.datasets.VisionDataset):

    def __init__(self, mimiccxr_dir, mimiccxr_metadata,
                 dataset_metadata, transform=None):
        super(MimicCxrDataset, self).__init__(root=None, transform=transform)

        self.mimiccxr_dir = mimiccxr_dir

        mimiccxr_metadata_df = MimicCxrMetadata(mimiccxr_metadata).get_sub_columns(
            ['subject_id', 'study_id', 'dicom_id'])
        dataset_metadata_df = MimicCxrMetadata(dataset_metadata).get_sub_columns(
            ['dicom_id'])
        metadata_df = MimicCxrMetadata.overlap_by_column(mimiccxr_metadata_df,
                                                           dataset_metadata_df,
                                                           'dicom_id')
        self.metadata_df = metadata_df.reset_index(drop=True)
        
        self.transform = transform

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, i):
        subject_id, study_id, dicom_id = \
            self.metadata_df.loc[i, ['subject_id', 'study_id', 'dicom_id']]
        dcm_path = os.path.join(
            self.mimiccxr_dir, f'p{subject_id}', f's{study_id}', f'{dicom_id}.dcm')
        if os.path.isfile(dcm_path):
            dcm = pydicom.dcmread(dcm_path)
            img = dcm.pixel_array
            dcm_exists = True
        else:
            img = np.zeros((256, 256), dtype=np.uint16)
            dcm_exists = False
        if self.transform is not None:
            img = self.transform(img)
        return img, dicom_id, dcm_exists


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


# for im_size in [256]:
#     start = time.time()
#     create_cached_images(
#         im_size,
#         '/data/vision/polina/scratch/wpq/github/interpretability/notebooks/data/MimicCxrChexpertDataset')
#     end = time.time()
#     print(f'im_size: {im_size} done!')