'''
Author: Ruizhi Liao

Main script to run evaluation
'''

import os
import argparse
import logging
import json

import torch

from resnet_chestxray.main_utils import ModelManager, build_model

current_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--label_key', default='Edema', type=str,
					help='The label key/classification task')

parser.add_argument('--img_size', default=256, type=int,
                    help='The size of the input image')
parser.add_argument('--output_channels', default=1, type=int,
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

	print(args)

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

	print(inference_results)

img_dir = '/data/vision/polina/scratch/ruizhi/chestxray/data/png_16bit_256/'
img_path = os.path.join(img_dir, 
						'p10062617_s55170181_5b8f4e5f-074a3958-ca8e7fc2-100ffa07-6f553e72.png')
infer(img_path)