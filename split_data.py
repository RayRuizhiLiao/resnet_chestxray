import os
import pandas as pd
import numpy as np

from resnet_chestxray import model_utils


current_dir = os.path.dirname(__file__)

mimiccxr_metadata = os.path.join(current_dir,
								 'mimic_cxr_edema/auxiliary_metadata/mimic_cxr_metadata_available_CHF_view.csv')

regex_labels = os.path.join(current_dir, 
							'mimic_cxr_edema/regex_report_edema_severity.csv')
consensus_labels = os.path.join(current_dir,
								'mimic_cxr_edema/consensus_image_edema_severity.csv')


save_path = os.path.join(current_dir, 'data/training_edema.csv')
model_utils.CXRImageDataset.create_dataset_metadata(mimiccxr_metadata=mimiccxr_metadata,
													label_metadata=regex_labels,
													data_key='study_id',
													save_path=save_path,
													holdout_metadata=consensus_labels, 
													holdout_key='subject_id')

save_path = os.path.join(current_dir, 'data/test_edema.csv')
model_utils.CXRImageDataset.create_dataset_metadata(mimiccxr_metadata=mimiccxr_metadata,
													label_metadata=consensus_labels,
													data_key='dicom_id',
													save_path=save_path)

'''
Split data using both chexpert labels and edema severity labels
'''

chexpert_labels = os.path.join(current_dir, 'mimic_cxr_labels/mimic-cxr-2.0.0-chexpert.csv')
test_edema = os.path.join(current_dir, 'data/test_edema.csv')
train_edema = os.path.join(current_dir, 'data/training_edema.csv')
chexpert_split =  os.path.join(current_dir, 'mimic_cxr_labels/mimic-cxr-2.0.0-split.csv')

test_edema = pd.read_csv(test_edema)
train_edema = pd.read_csv(train_edema)
test_edema_subjectid = [id[1:9] for id in test_edema.mimic_id.tolist()]
train_edema_subjectid = [id[1:9] for id in train_edema.mimic_id.tolist()]

chexpert_split = pd.read_csv(chexpert_split)
test_chexpert_split = chexpert_split[chexpert_split['split'].isin(['test'])].reset_index(drop=True)
train_chexpert_split = chexpert_split[chexpert_split['split'].isin(['train'])].reset_index(drop=True)

test_chexpert_split_edema = test_chexpert_split[~test_chexpert_split['subject_id'].apply(str).isin(train_edema_subjectid)].reset_index(drop=True)
test_chexpert_split_edema_subjectid = test_chexpert_split_edema['subject_id'].apply(str).tolist()

test_final_subjectid = test_chexpert_split_edema_subjectid+test_edema_subjectid

chexpert_labels = pd.read_csv(chexpert_labels)
chexpert_labels = chexpert_labels.replace(np.nan, -2)
test_chexpert = chexpert_labels[chexpert_labels['subject_id'].apply(str).isin(test_final_subjectid)].reset_index(drop=True)
test_chexpert.to_csv(os.path.join(current_dir, 'data/test_chexpert_tmp.csv'), index=False)
train_chexpert = chexpert_labels[~chexpert_labels['subject_id'].apply(str).isin(test_final_subjectid)].reset_index(drop=True)
train_chexpert.to_csv(os.path.join(current_dir, 'data/training_chexpert_tmp.csv'), index=False)

chexpert_keys = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',\
				 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',\
				 'Pneumothorax', 'Support Devices']
save_path = os.path.join(current_dir, 'data/training_chexpert.csv')
model_utils.CXRImageDataset.create_dataset_metadata(mimiccxr_metadata=mimiccxr_metadata,
													label_metadata=os.path.join(current_dir, 'data/training_chexpert_tmp.csv'),
													data_key='study_id',
													label_key=chexpert_keys,
													save_path=save_path)
save_path = os.path.join(current_dir, 'data/test_chexpert.csv')
model_utils.CXRImageDataset.create_dataset_metadata(mimiccxr_metadata=mimiccxr_metadata,
													label_metadata=os.path.join(current_dir, 'data/test_chexpert_tmp.csv'),
													data_key='study_id',
													label_key=chexpert_keys,
													save_path=save_path)