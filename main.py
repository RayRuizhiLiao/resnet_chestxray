'''
Author: Ruizhi Liao

Main script to run training and evaluation of a residual network model
on chest x-ray images
'''

import os
import numpy as np
from math import floor, ceil
import scipy.ndimage as ndimage
import logging
import json

import sys
from pathlib import Path
current_path = os.path.dirname(os.path.abspath(__file__))
current_path = Path(current_path)
# Should not use sys.path.append here
sys.path.insert(0, str(current_path)) 
print("sys.path: ", sys.path) 

import git # This is used to track commit sha
repo = git.Repo(path=current_path)
sha = repo.head.object.hexsha
print("Current git commit sha: ", sha)

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import resnet7_2_1
import parser
import main_utils


def main():
	args = parser.get_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	assert torch.cuda.is_available(), "No GPU/CUDA is detected!"

	assert args.do_train or args.do_eval, \
		"Either do_train or do_eval has to be True!"
	assert not(args.do_train and args.do_eval), \
		"do_train and do_eval cannot be both True!"

	# To track results from different commits (temporary)
	if args.commit_sha == None:
		args.run_id = args.run_id+'_'+str(sha)
	else:
		args.run_id = args.run_id+'_'+args.commit_sha

	if not args.run_id == None:
		args.output_dir = os.path.join(args.output_dir, args.run_id)
	if not(os.path.exists(args.output_dir)) and args.do_train:
		os.makedirs(args.output_dir)
	if args.do_eval:
		# output_dir has to exist if doing evaluation
		assert os.path.exists(args.output_dir), \
			"Output directory {} doesn't exist!".format(args.output_dir)
		# if args.data_split_mode=='testing': 
		# 	# Checkpoint has to exist if doing evaluation with testing split
		# 	assert os.path.exists(args.checkpoint_path), \
		# 		"Checkpoint doesn't exist!"

	'''
	Configure a log file
	'''
	if args.do_train:
		log_path = os.path.join(args.output_dir, 'training.log')
	if args.do_eval:
		log_path = os.path.join(args.output_dir, 'evaluation.log')
	logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w', 
						format='%(asctime)s - %(name)s %(message)s', 
						datefmt='%m-%d %H:%M')

	'''
	Log important info
	'''
	logger = logging.getLogger(__name__)
	logger.info("***** Code info *****")
	logger.info("  Git commit sha: %s", sha)

	'''
	Print important info
	'''
	print('Model architecture:', args.model_architecture)
	print('Training folds:', args.training_folds)
	print('Evaluation folds:', args.evaluation_folds)
	print('Device being used:', device)
	print('Output directory:', args.output_dir)
	print('Logging in:\t {}'.format(log_path))
	print('Input image formet:', args.image_format)

	if args.do_train:

		'''
		Create tensorboard and checkpoint directories if they don't exist
		'''
		args.tsbd_dir = os.path.join(args.output_dir, 'tsbd')
		args.checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
		directories = [args.tsbd_dir, args.checkpoints_dir]
		for directory in directories:
			if not(os.path.exists(directory)):
				os.makedirs(directory)
		# Avoid overwriting previous tensorboard and checkpoint data
		args.tsbd_dir = os.path.join(args.tsbd_dir, 
									 'tsbd_{}'.format(len(os.listdir(args.tsbd_dir))))
		if not os.path.exists(args.tsbd_dir):
			os.makedirs(args.tsbd_dir)
		args.checkpoints_dir = os.path.join(args.checkpoints_dir, 
											'checkpoints_{}'.format(len(os.listdir(args.checkpoints_dir))))
		if not os.path.exists(args.checkpoints_dir):
			os.makedirs(args.checkpoints_dir)

		'''
		Create an instance of a resnet model
		'''
		output_channels = 4
		if args.model_architecture == 'resnet7_2_1':
			resnet_model = resnet7_2_1(output_channels=output_channels)
		resnet_model = resnet_model.to(device)

		'''
		Train the model
		'''
		print("***** Training the model *****")
		main_utils.train(args, device, resnet_model)
		print("***** Finished training *****")

	if args.do_eval:

		def run_eval_on_checkpoint(checkpoint_path):
			'''
			Create an instance of a resnet model and load a checkpoint
			'''
			output_channels = 4
			if args.model_architecture == 'resnet7_2_1':
				resnet_model = resnet7_2_1(pretrained=True, 
										   pretrained_model_path=checkpoint_path,
										   output_channels=output_channels)
			resnet_model = resnet_model.to(device)

			'''
			Evaluate the model
			'''
			print("***** Evaluating the model *****")
			eval_results, embeddings, labels_raw = main_utils.evaluate(args, 
																	   device, 
																	   resnet_model)
			print("***** Finished evaluation *****")

			return eval_results, embeddings, labels_raw

		eval_results, _, _ = run_eval_on_checkpoint(checkpoint_path=args.checkpoint_path)

		results_path = os.path.join(args.output_dir, 'eval_results.json')
		with open(results_path, 'w') as fp:
			json.dump(eval_results, fp)


if __name__ == '__main__':
    main()