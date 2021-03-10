import os
import logging

from resnet_chestxray.main_utils import ModelManager, build_model

current_dir = os.path.dirname(__file__)

model_name = 'resnet512_6_2_1'
img_size = 512
model_manager = ModelManager(model_name=model_name, img_size=img_size)

data_dir = f'/data/vision/polina/scratch/ruizhi/chestxray/data/png_16bit_{img_size}/'
dataset_metadata = os.path.join(current_dir, 'data/training.csv')
batch_size = 256
save_dir = f'/data/vision/polina/scratch/ruizhi/chestxray/experiments/'\
		   f'supervised_image/tmp_test_resolution/{model_name}'

if not os.path.exists(save_dir):
	os.makedirs(save_dir)
log_path = os.path.join(save_dir, 'training.log')
logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w', 
					format='%(asctime)s - %(name)s %(message)s', 
					datefmt='%m-%d %H:%M')

model_manager.train(data_dir=data_dir, 
                    dataset_metadata=dataset_metadata,
                    batch_size=batch_size, save_dir=save_dir)