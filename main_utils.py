'''
Author: Ruizhi Liao

Main_utils script to run training and evaluation 
of a residual network model on chest x-ray images
'''

import os
from tqdm import tqdm, trange
import logging
from scipy.stats import logistic
import numpy as np
import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse
import csv
from scipy.special import softmax

import torch
import torchvision
from pytorch_transformers.optimization import WarmupLinearSchedule
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import BCEWithLogitsLoss, BCELoss, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from model_utils import CXRImageDataset
from model_utils import CenterCrop, RandomTranslateCrop
import eval_metrics


# TODO: optimize this method and maybe the csv format
def split_tr_eval(split_list_path, training_folds, evaluation_folds):
	"""
	Given a data split list (.csv), training folds and evaluation folds,
	return DICOM IDs and the associated labels for training and evaluation
	"""

	print('Data split list being used: ', split_list_path)

	train_labels = {}
	train_ids = {}
	eval_labels = {}
	eval_ids = {}

	with open(split_list_path, 'r') as train_label_file:
		train_label_file_reader = csv.reader(train_label_file)
		row = next(train_label_file_reader)
		for row in train_label_file_reader:
			if row[-1] != 'TEST':
				if int(row[-1]) in training_folds:
					train_labels[row[2]] = [float(row[3])]
					train_ids[row[2]] = row[1]
				if int(row[-1]) in evaluation_folds:
					eval_labels[row[2]] = [float(row[3])]
					eval_ids[row[2]] = row[1]
			if row[-1] == 'TEST' and -1 in evaluation_folds:
					eval_labels[row[2]] = [float(row[3])]
					eval_ids[row[2]] = row[1]              

	print("Training and evaluation folds: ", training_folds, evaluation_folds)
	print("Total number of training labels: ", len(train_labels))
	print("Total number of training DICOM IDs: ", len(train_ids))
	print("Total number of evaluation labels: ", len(eval_labels))
	print("Total number of evaluation DICOM IDs: ", len(eval_ids))

	return train_labels, train_ids, eval_labels, eval_ids

# Model training function
def train(args, device, model):

	'''
	Create a logger for logging model training
	'''
	logger = logging.getLogger(__name__)

	''' 
	Create an instance of loss
	'''
	# BCE_loss_criterion = BCELoss().to(device)
	CE_loss_criterion = CrossEntropyLoss().to(device)

	'''
	Create an instance of traning data loader
	'''
	xray_transform = RandomTranslateCrop(2048)
	train_labels, train_dicom_ids, _, _ = split_tr_eval(args.data_split_path,
													    args.training_folds,
													    args.evaluation_folds)
	cxr_dataset = CXRImageDataset(train_dicom_ids, train_labels, args.image_dir,
	                              transform=xray_transform, image_format=args.image_format)
	data_loader = DataLoader(cxr_dataset, batch_size=args.batch_size,
	                         shuffle=True, num_workers=8,
	                         pin_memory=True)
	print('Total number of training images: ', len(cxr_dataset))

	'''
	Create an instance of optimizer and learning rate scheduler
	'''
	optimizer = optim.Adam(model.parameters(), 
							lr=args.init_lr)
	if args.scheduler == 'WarmupLinearSchedule': # TODO: remove WarmupLinearSchedule
		num_train_optimization_steps = len(data_loader) * args.num_train_epochs
		args.warmup_steps = args.warmup_proportion * num_train_optimization_steps
		scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps,
										 t_total=num_train_optimization_steps)
	if args.scheduler == 'ReduceLROnPlateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-6)

	'''
	Log training info
	'''
	logger.info("***** Training info *****")
	logger.info("  Model architecture: %s", args.model_architecture)
	logger.info("  Data split file: %s", args.data_split_path)
	logger.info("  Training folds: %s\t Evaluation folds: %s"%(args.training_folds, args.evaluation_folds))
	logger.info("  Number of training examples: %d", len(cxr_dataset))
	logger.info("  Number of epochs: %d", args.num_train_epochs)
	logger.info("  Batch size: %d", args.batch_size)
	logger.info("  Initial learning rate: %f", args.init_lr)
	logger.info("  Learning rate scheduler: %s", args.scheduler)
	if args.scheduler == 'WarmupLinearSchedule':
		logger.info("  Total number of training steps: %d", num_train_optimization_steps)
		logger.info("  Number of steps for warming up learning rate: %d", args.warmup_steps)

	'''
	Create an instance of a tensorboard writer
	'''
	tsbd_writer = SummaryWriter(log_dir=args.tsbd_dir)
	# images, labels = next(iter(data_loader))
	# images = images.to(device)
	# labels = labels.to(device)
	# tsbd_writer.add_graph(model, images)

	'''
	Train the model
	'''
	model.train()
	train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
	global_step = 0
	running_loss = 0
	logger.info("***** Training the model *****")
	for epoch in train_iterator:
	    logger.info("  Starting a new epoch: %d", epoch + 1)
	    epoch_iterator = tqdm(data_loader, desc="Iteration")
	    tr_loss = 0
	    for i, batch in enumerate(epoch_iterator, 0):
	        # Get the batch 
	        batch = tuple(t.to(device, non_blocking=True) for t in batch)
	        inputs, labels, labels_raw = batch

	        # Zero the parameter gradients
	        optimizer.zero_grad()

	        # Forward + backward + optimize
	        outputs = model(inputs)
	        # loss = BCE_loss_criterion(outputs[0], labels) #TODO: check this outputs[0]
	        loss = CE_loss_criterion(outputs[-1], labels_raw)
	        loss.backward()
	        optimizer.step()

			# Update learning rate scheduler
	        if args.scheduler == 'WarmupLinearSchedule':
	        	scheduler.step()

	        # Print and record statistics
	        running_loss += loss.item()
	        tr_loss += loss.item()
	        global_step += 1
	        if global_step % args.logging_steps == 0:
	            #grid = torchvision.utils.make_grid(inputs)
	            #tsbd_writer.add_image('images', grid, global_step)
	            tsbd_writer.add_scalar('loss/train', 
	                                   running_loss / (args.logging_steps*args.batch_size), 
	                                   global_step)
	            tsbd_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
	            logger.info("  [%d, %5d, %5d] learning rate = %f"%\
	            	(epoch + 1, i + 1, global_step, optimizer.param_groups[0]['lr']))
	            logger.info("  [%d, %5d, %5d] loss = %.5f"%\
	            	(epoch + 1, i + 1, global_step, running_loss / (args.logging_steps*args.batch_size)))        
	            running_loss = 0
	    logger.info("  Finished an epoch: %d", epoch + 1)
	    logger.info("  Training loss of epoch %d = %.5f"%\
	    	(epoch+1, tr_loss / (len(cxr_dataset)*args.batch_size)))
	    model.save_pretrained(args.checkpoints_dir, epoch=epoch + 1)
	    if args.scheduler == 'ReduceLROnPlateau':
	    	scheduler.step(tr_loss)

	tsbd_writer.close()

# Model evaluation function
def evaluate(args, device, model):

	'''
	Create a logger for logging model evaluation results
	'''
	logger = logging.getLogger(__name__)

	# TODO: remove output_channel_encoding
	output_channel_encoding = 'multiclass'

	'''
	Create an instance of evaluation data loader
	'''
	xray_transform = CenterCrop(2048)
	_, _, eval_labels, eval_dicom_ids = split_tr_eval(args.data_split_path,
													  args.training_folds,
													  args.evaluation_folds)
	cxr_dataset = CXRImageDataset(eval_dicom_ids, eval_labels, args.image_dir,
								  transform=xray_transform, 
								  image_format=args.image_format)
	data_loader = DataLoader(cxr_dataset, batch_size=args.batch_size,
	                         num_workers=8, pin_memory=True)
	print('Total number of evaluation images: ', len(cxr_dataset))

	'''
	Log evaluation info
	'''
	logger.info("***** Evaluation info *****")
	logger.info("  Model architecture: %s", args.model_architecture)
	logger.info("  Data split file: %s", args.data_split_path)
	logger.info("  Training folds: %s\t Evaluation folds: %s"%(args.training_folds, args.evaluation_folds))
	logger.info("  Number of evaluation examples: %d", len(cxr_dataset))
	logger.info("  Number of epochs: %d", args.num_train_epochs)
	logger.info("  Batch size: %d", args.batch_size)
	logger.info("  Model checkpoint {}:".format(args.checkpoint_path))

	'''
	Evaluate the model
	'''

	logger.info("***** Evaluating the model *****")

	# For storing labels and model predictions
	preds = []
	labels = []
	embeddings = []

	model.eval()
	epoch_iterator = tqdm(data_loader, desc="Iteration")
	for i, batch in enumerate(epoch_iterator, 0):
		# Get the batch; each batch is a list of [image, label]
		batch = tuple(t.to(device, non_blocking=True) for t in batch)
		image, label, _ = batch
		with torch.no_grad():
			output, embedding, _ = model(image)
			pred = output.detach().cpu().numpy()
			embedding = embedding.detach().cpu().numpy()
			label = label.detach().cpu().numpy()
			# TODO: remove this piece
			# print(output)
			# pred = logistic.cdf(pred)
			# print(pred)
			# preds_logits=logistic.cdf(pred)
			# print(preds_logits)
			# print(softmax(preds_logits, axis=1))
			# print(pred)
			for j in range(len(pred)):
				preds.append(pred[j])
				labels.append(label[j])
				embeddings.append(embedding[j])

	labels_raw = np.argmax(labels, axis=1)
	eval_results = {}

	ordinal_aucs = eval_metrics.compute_ordinal_auc(labels, preds)
	eval_results['ordinal_aucs'] = ordinal_aucs

	pairwise_aucs = eval_metrics.compute_pairwise_auc(labels, preds)
	eval_results['pairwise_auc'] = pairwise_aucs

	multiclass_aucs = eval_metrics.compute_multiclass_auc(labels, preds)
	eval_results['multiclass_aucs'] = multiclass_aucs

	eval_results['mse'] = eval_metrics.compute_mse(labels_raw, preds)

	results_acc_f1, _, _ = eval_metrics.compute_acc_f1_metrics(labels_raw, preds)
	eval_results.update(results_acc_f1)

	logger.info("  AUC(0v123) = %4f", eval_results['ordinal_aucs'][0])
	logger.info("  AUC(01v23) = %4f", eval_results['ordinal_aucs'][1])
	logger.info("  AUC(012v3) = %4f", eval_results['ordinal_aucs'][2])

	logger.info("  AUC(0v1) = %4f", eval_results['pairwise_auc']['0v1'])
	logger.info("  AUC(0v2) = %4f", eval_results['pairwise_auc']['0v2'])
	logger.info("  AUC(0v3) = %4f", eval_results['pairwise_auc']['0v3'])
	logger.info("  AUC(1v2) = %4f", eval_results['pairwise_auc']['1v2'])
	logger.info("  AUC(1v3) = %4f", eval_results['pairwise_auc']['1v3'])
	logger.info("  AUC(2v3) = %4f", eval_results['pairwise_auc']['2v3'])

	logger.info("  AUC(0v123) = %4f", eval_results['multiclass_aucs'][0])
	logger.info("  AUC(1v023) = %4f", eval_results['multiclass_aucs'][1])
	logger.info("  AUC(2v013) = %4f", eval_results['multiclass_aucs'][2])
	logger.info("  AUC(3v012) = %4f", eval_results['multiclass_aucs'][3])

	logger.info("  MSE = %4f", eval_results['mse'])

	logger.info("  Macro_F1 = %4f", eval_results['macro_f1'])
	logger.info("  Accuracy = %4f", eval_results['accuracy'])

	return eval_results, embeddings, labels_raw
