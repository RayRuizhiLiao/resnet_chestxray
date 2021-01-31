import os

import pandas as pd


class MimicID:
	subject_id = ''
	study_id = ''
	dicom_id = ''

	def __init__(self, subject_id, study_id, dicom_id):
		self.subject_id = str(subject_id)
		self.study_id = str(study_id)
		self.dicom_id = str(dicom_id)

	def __str__(self):
		return f"p{self.subject_id}_s{self.study_id}_{self.dicom_id}"


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
