'''
Author: Ruizhi Liao
Main script to run inference
'''

import os
import argparse
import logging
import json
import pandas as pd
import json

import torch

from resnet_chestxray.main_utils import ModelManager, build_model
from resnet_chestxray import utils

current_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--label_key', default='edema_severity', type=str,
					help='The label key/classification task')

parser.add_argument('--img_size', default=256, type=int,
                    help='The size of the input image')
parser.add_argument('--output_channels', default=4, type=int,
                    help='The number of ouput channels')
parser.add_argument('--model_architecture', default='resnet256_6_2_1', type=str,
                    help='Neural network architecture to be used')

parser.add_argument('--save_dir', type=str,
					default='/data/vision/polina/scratch/ruizhi/chestxray/experiments/supervised_image/'\
					'tmp_postmiccai_v2/')
parser.add_argument('--checkpoint_name', type=str,
					default='pytorch_model_epoch300.bin')


def infer(img_path):
	args = parser.parse_args()

	'''
	Check cuda
	'''
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	assert torch.cuda.is_available(), "No GPU/CUDA is detected!"

	'''
	Create a sub-directory under save_dir 
	based on the label key
	'''
	args.save_dir = os.path.join(args.save_dir, args.model_architecture+'_'+args.label_key)

	checkpoint_path = os.path.join(args.save_dir, args.checkpoint_name)

	model_manager = ModelManager(model_name=args.model_architecture, 
								 img_size=args.img_size,
								 output_channels=args.output_channels)
	inference_results = model_manager.infer(device=device,
											args=args,
											checkpoint_path=checkpoint_path,
											img_path=img_path)

	return inference_results

def get_all_mimiccxr():
	all_metadata = os.path.join(current_dir, 'mimic_cxr_edema', 
								'auxiliary_metadata', 'mimic_cxr_metadata_available_CHF_view.csv')
	all_metadata = pd.read_csv(all_metadata)

	all_metadata = all_metadata[all_metadata['dicom_available']==True].reset_index(drop=True)
	all_metadata = all_metadata[all_metadata['CHF']==True].reset_index(drop=True)
	all_metadata = all_metadata[all_metadata['view']=='frontal'].reset_index(drop=True)

	all_mimicids = []
	for i in range(len(all_metadata)):
		subject_id = all_metadata['subject_id'][i]
		study_id = all_metadata['study_id'][i]
		dicom_id = all_metadata['dicom_id'][i]
		mimicid = utils.MimicID(subject_id, study_id, dicom_id).__str__()
		all_mimicids.append(mimicid)

	return all_mimicids

def infer_all_imgs():
	all_mimicids = get_all_mimiccxr()
	img_dir = '/data/vision/polina/scratch/ruizhi/chestxray/data/png_16bit_256/'
	mimiccxr_edema_prob = {}
	i=0
	for mimicid in all_mimicids:
		img_path = os.path.join(img_dir, mimicid+'.png')
		inference_results = infer(img_path)
		edema_prob = inference_results['pred_prob'][0]
		mimiccxr_edema_prob[mimicid] = edema_prob.tolist()
		i+=1
		if i%1000==0:
			print(f"{i} out of {len(all_mimicids)} done!")

	with open("mimiccxr_edema_prob.json", "w") as outfile:
		json.dump(mimiccxr_edema_prob, outfile)

infer_all_imgs() 