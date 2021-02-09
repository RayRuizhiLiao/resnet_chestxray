# resnet_chestxray

Residual neural networks for pulmonary edema assessment in chest radiographs 

## Generate PNG images from MIMIC-CXR DICOM data
Run `python image_prep/dcm_to_png.py` to generate PNG images from MIMIC-CXR DICOM data, given specified metadata information. This script also resizes the images (the width to the disired size, and the length accordingly without changing the length:width ratio).  

## Docker image

The Docker image of this repo is stored at: https://hub.docker.com/repository/docker/rayruizhiliao/mlmodel_cxr_edema.

To build the Docker image, run 
```
sudo docker build -t mlmodel_cxr_edema .
```

To run the Docker image, run
```
sudo docker run -it rayruizhiliao/mlmodel_cxr_edema:latest
```
