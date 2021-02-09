import os

from resnet_chestxray import model_utils


current_dir = os.path.dirname(__file__)

mimiccxr_metadata = os.path.join(current_dir,
								 'mimic_cxr_edema/auxiliary_metadata/mimic_cxr_metadata_available_CHF_view.csv')

regex_labels = os.path.join(current_dir, 
							'mimic_cxr_edema/regex_report_edema_severity.csv')
consensus_labels = os.path.join(current_dir,
								'mimic_cxr_edema/consensus_image_edema_severity.csv')


save_path = os.path.join(current_dir, 'data/training.csv')
model_utils.CXRImageDataset.create_dataset_metadata(mimiccxr_metadata=mimiccxr_metadata,
													label_metadata=regex_labels,
													data_key='study_id',
													save_path=save_path,
													holdout_metadata=consensus_labels, 
													holdout_key='subject_id')

save_path = os.path.join(current_dir, 'data/test.csv')
model_utils.CXRImageDataset.create_dataset_metadata(mimiccxr_metadata=mimiccxr_metadata,
													label_metadata=consensus_labels,
													data_key='dicom_id',
													save_path=save_path)