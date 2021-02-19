'''
Author: Ruizhi Liao

Main script to run training
'''

import os
import argparse
import logging

from resnet_chestxray.main_utils import ModelManager, build_model

current_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=64, type=int,
					help='Mini-batch size')

parser.add_argument('--label_key', default='Edema', type=str,
					help='The label key/classification task')

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
					'supervised_image/tmp_test_chexpert_edema/')


def train():
	args = parser.parse_args()

	args.save_dir = os.path.join(args.save_dir, args.model_architecture+'_'+args.label_key)

	print(args)

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
	model_manager.train(data_dir=args.data_dir, 
						dataset_metadata=args.dataset_metadata,
						batch_size=args.batch_size, save_dir=args.save_dir,
						label_key=args.label_key, loss_method='BCEWithLogitsLoss')

train()