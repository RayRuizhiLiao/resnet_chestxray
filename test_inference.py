'''
Author: Ruizhi Liao

Test script to run inference of a residual network model
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet_chestxray.model import resnet7_2_1
from resnet_chestxray.model_utils import load_image
import parser
import resnet_chestxray.main_utils as main_utils
from gradcam.grad_cam import GradCAM, save_gradcam_overlay


def run_inference(model_architecture='resnet7_2_1', 
				  checkpoint_path='./data/pytorch_model_epoch300.bin',
				  image_path='./data/52658141.png'):
	device = 'cpu'

	'''
	Create an instance of a resnet model and load a checkpoint
	'''
	output_channels = 4
	if model_architecture == 'resnet7_2_1':
		resnet_model = resnet7_2_1(pretrained=True, 
								   pretrained_model_path=checkpoint_path,
								   output_channels=output_channels)
	resnet_model = resnet_model.to(device)

	'''
	Load the input image
	'''
	image = load_image(image_path)

	'''
	Run model inference on the image
	'''
	pred = main_utils.inference(resnet_model, image)
	pred = pred[0]
	severity = sum([i*pred[i] for i in range(len(pred))])

	print(f"{image_path} has severity of {severity}")

def run_inference_gradcam(model_architecture='resnet7_2_1', 
						  checkpoint_path='./data/pytorch_model_epoch300.bin',
						  image_path='./data/52658141.png',
						  gcam_path='/mnt/images/52658141_gcam.png'):
	device = 'cpu'

	'''
	Create an instance of a resnet model and load a checkpoint
	'''
	output_channels = 4
	if model_architecture == 'resnet7_2_1':
		resnet_model = resnet7_2_1(pretrained=True, 
								   pretrained_model_path=checkpoint_path,
								   output_channels=output_channels)
	resnet_model = resnet_model.to(device)

	'''
	Create an instance of model with Grad-CAM 
	'''
	model_gcam = GradCAM(model=resnet_model)

	'''
	Load the input image
	'''
	image = load_image(image_path)

	'''
	Run model inference on the image with Grad-CAM
	'''
	pred, gcam_img, input_img = main_utils.inference_gradcam(model_gcam, image,
															 'layer7.1.conv2')

	pred = pred[0]
	severity = sum([i*pred[i] for i in range(len(pred))])

	print(f"{image_path} has severity of {severity}")

	save_gradcam_overlay(gcam_path, gcam_img[0], input_img[0])

	print(f"Grad-CAM overlay saved at {gcam_path}")

	return

run_inference()