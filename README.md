# resnet_chestxray

Residual neural networks for pulmonary edema assessment in chest radiographs 

## Generate PNG images from MIMIC-CXR DICOM data
Run `python image_prep/dcm_to_png.py` to generate PNG images from MIMIC-CXR DICOM data, given specified metadata information. This script also resizes the images (the width to the desired size, and the length accordingly without changing the length:width ratio of the original DICOM image).  

## Split data
We use the regex labels for the image model training while holding out images/labels that come from the subjects in the test dataset. We use the consensus image labels as our test set. All the metadata information are under `mimic_cxr_edema/`.

Run `python split_data.py` to generate training and test metadata files.

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
