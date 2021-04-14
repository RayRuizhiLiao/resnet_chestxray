'''
Author: Ruizhi Liao

Main script to run training
'''

import os
import argparse
import logging

import torch

from resnet_chestxray.main_utils import ModelManager

current_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=64, type=int,
					help='Mini-batch size')
parser.add_argument('--num_train_epochs', default=300, type=int,
                    help='Number of training epochs')
parser.add_argument('--loss_method', type=str,
                    default='BCEWithLogitsLoss',
                    help='Loss function for model training')
parser.add_argument('--init_lr', default=5e-4, type=float, 
                    help='Intial learning rate')

parser.add_argument('--img_size', default=256, type=int,
                    help='The size of the input image')
parser.add_argument('--output_channels', default=1, type=int,
                    help='The number of ouput channels')
parser.add_argument('--model_architecture', default='resnet256_6_2_1', type=str,
                    help='Neural network architecture to be used')

parser.add_argument('--data_dir', type=str,
					default='/data/vision/polina/scratch/ruizhi/chestxray/data/png_16bit_256/',
					help='The image data directory')
parser.add_argument('--dataset_metadata', type=str,
					default=os.path.join(current_dir, 'data/training_chexpert.csv'),
					help='The metadata for the model training ')
parser.add_argument('--save_dir', type=str,
					default='/data/vision/polina/scratch/ruizhi/chestxray/experiments/'\
					'supervised_image/tmp_postmiccai_v2/')
parser.add_argument('--label_key', type=str,
                    default='Edema',
                    help='The supervised task (the key of the corresponding label column)')


def train():
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
	args.save_dir = os.path.join(args.save_dir, 
								 args.model_architecture+'_'+args.label_key)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	# Configure the log file
	log_path = os.path.join(args.save_dir, 'training.log')
	logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w', 
						format='%(asctime)s - %(name)s %(message)s', 
						datefmt='%m-%d %H:%M')

	model_manager = ModelManager(model_name=args.model_architecture, 
								 img_size=args.img_size,
								 output_channels=args.output_channels)

	model_manager.train(device=device,
						args=args)

train()