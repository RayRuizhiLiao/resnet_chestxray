import os

import pandas as pd


class MimicCxrMetadata():

	def __init__(self, mimiccxr_metadata):
		self.metadata_path = mimiccxr_metadata
		self.metadata_df = pd.read_csv(self.metadata_path)

	def get_sub_columns(self, keys: list):
		return self.metadata_df[keys]

	def get_sub_rows(self, key: str, value: list):
		return self.metadata_df[self.metadata_df[key].isin(values)]

	@staticmethod
	def overlap_by_column(df1, df2, key: str):
		return df1[df1[key].isin(df2[key])]		
