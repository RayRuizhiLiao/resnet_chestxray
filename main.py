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

from model import resnet14_1, resnet14_16, resnet14_4, resnet10_16
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

	if not(os.path.exists(args.output_dir)) and args.do_train:
		os.makedirs(args.output_dir)
	if args.do_eval:
		# output_dir has to exist if doing evaluation
		assert os.path.exists(args.output_dir), \
			"Output directory doesn't exist!"
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
	print('Validation folds:', args.validation_folds)
	print('Device being used:', device)
	print('Output directory:', args.output_dir)
	print('Logging in:\t {}'.format(log_path))
	print('Input image formet:', args.image_format)
	print('Label encoding:', args.label_encoding)

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
		Create instance of a resnet model
		'''
		# TODO: remove args.label_encoding
		args.label_encoding == 'onehot':
		add_softmax = True
		output_channels = 4
		if args.model_architecture == 'resnet14_1':
			resnet_model = resnet14_1(add_softmax=add_softmax, 
									  output_channels=output_channels)
		if args.model_architecture == 'resnet14_16':
			resnet_model = resnet14_16(add_softmax=add_softmax, 
									   output_channels=output_channels)
		if args.model_architecture == 'resnet14_4':
			resnet_model = resnet14_4(add_softmax=add_softmax, 
									  output_channels=output_channels)
		if args.model_architecture == 'resnet10_16':
			resnet_model = resnet10_16(add_softmax=add_softmax, 
									   output_channels=output_channels)
		resnet_model = resnet_model.to(device)

		'''
		Train the model
		'''
		print("***** Training the model *****")
		main_utils.train(args, device, resnet_model)
		print("***** Finished training *****")

	if args.do_eval:

		def run_eval_on_checkpoint():
			# Create instance of a resnet model and load a checkpoint
			if args.label_encoding == 'onehot':
				add_softmax = True
				output_channels = 4
			elif args.label_encoding == 'ordinal':
				add_softmax = False
				output_channels = 3
			if args.model_architecture == 'resnet14_1':
				resnet_model = resnet14_1(pretrained=True, 
										  checkpoint=args.checkpoint_path,
										  add_softmax=add_softmax,
										  output_channels=output_channels)
			if args.model_architecture == 'resnet14_16':
				resnet_model = resnet14_16(pretrained=True, 
										   checkpoint=args.checkpoint_path,
										   add_softmax=add_softmax,
										   output_channels=output_channels)
			if args.model_architecture == 'resnet14_4':
				resnet_model = resnet14_4(pretrained=True, 
										  checkpoint=args.checkpoint_path,
										  add_softmax=add_softmax,
										  output_channels=output_channels)
			if args.model_architecture == 'resnet10_16':
				resnet_model = resnet10_16(pretrained=True, 
										   checkpoint=args.checkpoint_path,
										   add_softmax=add_softmax,
										   output_channels=output_channels)
			resnet_model = resnet_model.to(device)

			# Evaluate the model
			print("--Evaluating the model--")
			eval_results, embeddings, labels_raw = main_utils.evaluate(args, device, resnet_model, args.checkpoint_path, eval_ids, eval_labels)
			print('--Finished Evaluation--')

			return eval_results, embeddings, labels_raw

		if args.data_split_mode == 'cross_val':
			epoch = args.eval_epoch
			cross_val_log_path = os.path.join(raw_output_dir, 'cross_val_evaluation_'+str(epoch)+'.txt')
			with open(cross_val_log_path, 'w') as cross_val_log:
				# train_labels, train_ids, eval_labels, eval_ids = _split_tr_val(args.data_split_path, [2,3,4,5], [1])
				# args.checkpoint_path = os.path.join(raw_output_dir, 
				# 									'train2345_val1/checkpoints/checkpoints0/pytorch_model_epoch'+str(epoch)+'.bin')
				train_labels, train_ids, eval_labels, eval_ids = _split_tr_val(args.data_split_path, [1,2,3,4], [0])
				args.checkpoint_path = os.path.join(raw_output_dir, 
													'train1234_val0/checkpoints/checkpoints0/pytorch_model_epoch'+str(epoch)+'.bin')				
				eval_results_1 = run_eval_on_checkpoint()
				# train_labels, train_ids, eval_labels, eval_ids = _split_tr_val(args.data_split_path, [1,3,4,5], [2])
				# args.checkpoint_path = os.path.join(raw_output_dir, 
				# 									'train1345_val2/checkpoints/checkpoints0/pytorch_model_epoch'+str(epoch)+'.bin')
				train_labels, train_ids, eval_labels, eval_ids = _split_tr_val(args.data_split_path, [0,2,3,4], [1])
				args.checkpoint_path = os.path.join(raw_output_dir, 
													'train0234_val1/checkpoints/checkpoints0/pytorch_model_epoch'+str(epoch)+'.bin')
				eval_results_2 = run_eval_on_checkpoint()
				# train_labels, train_ids, eval_labels, eval_ids = _split_tr_val(args.data_split_path, [1,2,4,5], [3])
				# args.checkpoint_path = os.path.join(raw_output_dir, 
				# 									'train1245_val3/checkpoints/checkpoints0/pytorch_model_epoch'+str(epoch)+'.bin')
				train_labels, train_ids, eval_labels, eval_ids = _split_tr_val(args.data_split_path, [0,1,3,4], [2])
				args.checkpoint_path = os.path.join(raw_output_dir, 
													'train0134_val2/checkpoints/checkpoints0/pytorch_model_epoch'+str(epoch)+'.bin')
				eval_results_3 = run_eval_on_checkpoint()
				# train_labels, train_ids, eval_labels, eval_ids = _split_tr_val(args.data_split_path, [1,2,3,5], [4])
				# args.checkpoint_path = os.path.join(raw_output_dir, 
				# 									'train1235_val4/checkpoints/checkpoints0/pytorch_model_epoch'+str(epoch)+'.bin')
				train_labels, train_ids, eval_labels, eval_ids = _split_tr_val(args.data_split_path, [0,1,2,4], [3])
				args.checkpoint_path = os.path.join(raw_output_dir, 
													'train0124_val3/checkpoints/checkpoints0/pytorch_model_epoch'+str(epoch)+'.bin')
				eval_results_4 = run_eval_on_checkpoint()
				# train_labels, train_ids, eval_labels, eval_ids = _split_tr_val(args.data_split_path, [1,2,3,4], [5])
				# args.checkpoint_path = os.path.join(raw_output_dir, 
				# 									'train1234_val5/checkpoints/checkpoints0/pytorch_model_epoch'+str(epoch)+'.bin')
				train_labels, train_ids, eval_labels, eval_ids = _split_tr_val(args.data_split_path, [0,1,2,3], [4])
				args.checkpoint_path = os.path.join(raw_output_dir, 
													'train0123_val4/checkpoints/checkpoints0/pytorch_model_epoch'+str(epoch)+'.bin')
				eval_results_5 = run_eval_on_checkpoint()

				cross_val_log.write(str(round(eval_results_1['auc'][0],2))+',')
				cross_val_log.write(str(round(eval_results_2['auc'][0],2))+',')
				cross_val_log.write(str(round(eval_results_3['auc'][0],2))+',')
				cross_val_log.write(str(round(eval_results_4['auc'][0],2))+',')
				cross_val_log.write(str(round(eval_results_5['auc'][0],2))+'\n')

				cross_val_log.write(str(round(eval_results_1['auc'][1],2))+',')
				cross_val_log.write(str(round(eval_results_2['auc'][1],2))+',')
				cross_val_log.write(str(round(eval_results_3['auc'][1],2))+',')
				cross_val_log.write(str(round(eval_results_4['auc'][1],2))+',')
				cross_val_log.write(str(round(eval_results_5['auc'][1],2))+'\n')

				cross_val_log.write(str(round(eval_results_1['auc'][2],2))+',')
				cross_val_log.write(str(round(eval_results_2['auc'][2],2))+',')
				cross_val_log.write(str(round(eval_results_3['auc'][2],2))+',')
				cross_val_log.write(str(round(eval_results_4['auc'][2],2))+',')
				cross_val_log.write(str(round(eval_results_5['auc'][2],2))+'\n')

				if args.label_encoding == 'onehot':
					cross_val_log.write(str(round(eval_results_1['auc'][3],2))+',')
					cross_val_log.write(str(round(eval_results_2['auc'][3],2))+',')
					cross_val_log.write(str(round(eval_results_3['auc'][3],2))+',')
					cross_val_log.write(str(round(eval_results_4['auc'][3],2))+',')
					cross_val_log.write(str(round(eval_results_5['auc'][3],2))+'\n')

				cross_val_log.write(str(round(eval_results_1['pairwise_auc']['0v1'],2))+',')
				cross_val_log.write(str(round(eval_results_2['pairwise_auc']['0v1'],2))+',')
				cross_val_log.write(str(round(eval_results_3['pairwise_auc']['0v1'],2))+',')
				cross_val_log.write(str(round(eval_results_4['pairwise_auc']['0v1'],2))+',')
				cross_val_log.write(str(round(eval_results_5['pairwise_auc']['0v1'],2))+'\n')

				cross_val_log.write(str(round(eval_results_1['pairwise_auc']['0v2'],2))+',')
				cross_val_log.write(str(round(eval_results_2['pairwise_auc']['0v2'],2))+',')
				cross_val_log.write(str(round(eval_results_3['pairwise_auc']['0v2'],2))+',')
				cross_val_log.write(str(round(eval_results_4['pairwise_auc']['0v2'],2))+',')
				cross_val_log.write(str(round(eval_results_5['pairwise_auc']['0v2'],2))+'\n')

				cross_val_log.write(str(round(eval_results_1['pairwise_auc']['0v3'],2))+',')
				cross_val_log.write(str(round(eval_results_2['pairwise_auc']['0v3'],2))+',')
				cross_val_log.write(str(round(eval_results_3['pairwise_auc']['0v3'],2))+',')
				cross_val_log.write(str(round(eval_results_4['pairwise_auc']['0v3'],2))+',')
				cross_val_log.write(str(round(eval_results_5['pairwise_auc']['0v3'],2))+'\n')

				cross_val_log.write(str(round(eval_results_1['pairwise_auc']['1v2'],2))+',')
				cross_val_log.write(str(round(eval_results_2['pairwise_auc']['1v2'],2))+',')
				cross_val_log.write(str(round(eval_results_3['pairwise_auc']['1v2'],2))+',')
				cross_val_log.write(str(round(eval_results_4['pairwise_auc']['1v2'],2))+',')
				cross_val_log.write(str(round(eval_results_5['pairwise_auc']['1v2'],2))+'\n')

				cross_val_log.write(str(round(eval_results_1['pairwise_auc']['1v3'],2))+',')
				cross_val_log.write(str(round(eval_results_2['pairwise_auc']['1v3'],2))+',')
				cross_val_log.write(str(round(eval_results_3['pairwise_auc']['1v3'],2))+',')
				cross_val_log.write(str(round(eval_results_4['pairwise_auc']['1v3'],2))+',')
				cross_val_log.write(str(round(eval_results_5['pairwise_auc']['1v3'],2))+'\n')

				cross_val_log.write(str(round(eval_results_1['pairwise_auc']['2v3'],2))+',')
				cross_val_log.write(str(round(eval_results_2['pairwise_auc']['2v3'],2))+',')
				cross_val_log.write(str(round(eval_results_3['pairwise_auc']['2v3'],2))+',')
				cross_val_log.write(str(round(eval_results_4['pairwise_auc']['2v3'],2))+',')
				cross_val_log.write(str(round(eval_results_5['pairwise_auc']['2v3'],2))+'\n')

				cross_val_log.write(str(round(eval_results_1['mse'],2))+',')
				cross_val_log.write(str(round(eval_results_2['mse'],2))+',')
				cross_val_log.write(str(round(eval_results_3['mse'],2))+',')
				cross_val_log.write(str(round(eval_results_4['mse'],2))+',')
				cross_val_log.write(str(round(eval_results_5['mse'],2))+'\n')

				cross_val_log.write(str(round(eval_results_1['macro_f1'],2))+',')
				cross_val_log.write(str(round(eval_results_2['macro_f1'],2))+',')
				cross_val_log.write(str(round(eval_results_3['macro_f1'],2))+',')
				cross_val_log.write(str(round(eval_results_4['macro_f1'],2))+',')
				cross_val_log.write(str(round(eval_results_5['macro_f1'],2))+'\n')

				cross_val_log.write(str(round(eval_results_1['accuracy'],2))+',')
				cross_val_log.write(str(round(eval_results_2['accuracy'],2))+',')
				cross_val_log.write(str(round(eval_results_3['accuracy'],2))+',')
				cross_val_log.write(str(round(eval_results_4['accuracy'],2))+',')
				cross_val_log.write(str(round(eval_results_5['accuracy'],2))+'\n')
		else:
			epoch = args.eval_epoch
			eval_log_path = os.path.join(raw_output_dir, 'evaluation_'+str(epoch)+'.txt')
			with open(eval_log_path, 'w') as eval_log:
				if args.data_split_mode == 'testing':
					use_test_data = True
				train_labels, train_ids, eval_labels, eval_ids = _split_tr_val(args.data_split_path, [1,2,3,4,5], [],
																			   use_test_data=use_test_data)				
				eval_results, embeddings, labels_raw = run_eval_on_checkpoint()

				eval_log.write(str(round(eval_results['auc'][0],2))+'\n')
				eval_log.write(str(round(eval_results['auc'][1],2))+'\n')
				eval_log.write(str(round(eval_results['auc'][2],2))+'\n')
				if args.label_encoding == 'onehot':
					eval_log.write(str(round(eval_results['auc'][3],2))+'\n')
				eval_log.write(str(round(eval_results['pairwise_auc']['0v1'],2))+'\n')
				eval_log.write(str(round(eval_results['pairwise_auc']['0v2'],2))+'\n')
				eval_log.write(str(round(eval_results['pairwise_auc']['0v3'],2))+'\n')
				eval_log.write(str(round(eval_results['pairwise_auc']['1v2'],2))+'\n')
				eval_log.write(str(round(eval_results['pairwise_auc']['1v3'],2))+'\n')
				eval_log.write(str(round(eval_results['pairwise_auc']['2v3'],2))+'\n')
				eval_log.write(str(round(eval_results['mse'],2))+'\n')
				eval_log.write(str(round(eval_results['macro_f1'],2))+'\n')
				eval_log.write(str(round(eval_results['accuracy'],2))+'\n')
			
				out_labels_raw_path = os.path.join(raw_output_dir, "eval_results_labels")
				np.save(out_labels_raw_path, labels_raw)
				img_embeddings_path = os.path.join(raw_output_dir, "eval_results_image_embeddings")
				np.save(img_embeddings_path, embeddings)
# txt_embeddings_path = os.path.join(eval_output_dir, "eval_results_text_embeddings")
# np.save(txt_embeddings_path, txt_embeddings)



if __name__ == '__main__':
    main()