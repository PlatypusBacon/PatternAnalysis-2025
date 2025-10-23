# Alzheimer's Disease Classification using ConvNeXt

## Project Description
This project focuses on classifying Alzheimer's disease (AD) as well as normal cognition brain scans (NC) using MRI data from the alzheimer's disease neuroimaging initiative (ADNI) dataset. This task was attempted using a ConvNeXt convolutional neural network (CNN), a modern architecture able to achieve excellent performance on image recognition tasks while keeping efficiency.

## Model Architecture

ConvNeXt is structured combining aspects of convolutional neural networks (CNNs), as well as successful elements from Vision Transformers (ViTs). This is done through several innovations, namely the patchify stem which uses a 4x4 (stride 4) convolution rather than a 7x7 as the first stage, resembling the patches of ViTs.
The network also uses GELU activations instead of ReLU, and Layer in in place of Batch Normalisation
(Liu et al., 2022)

The full ConvNeXt architecture can be seen below:

![ConvNeXt architecture](ConvNeXt-structure-1.webp)

This shows several other key innovations in ConvNeXt, namely the four stage structure of ConvNeXt blocks.

### ConvNeXt Block

The ConvNeXt block is defined as seen below:

![ConvNeXt block](ConvNeXt-structure-2.webp)


## Dataset Overview
The image dataset is split as seen in the below file tree

```text
AD_NC  
├── test  
│   ├── AD  
│   └── NC  
└── train  
    ├── AD  
    └── NC  
```

the AD folders contain images (jpeg) of MRI scans classified with alzheimers disease, while NC contains normal cognition
These are present in both test and train folders, and represent the 2 classes for the data. 

There are a total of 21520 training images, and 9000 testing images.

the sizes of the images are as follows:
(256, 240)    30520
Name: count, dtype: int64

This shows uniform sizes across all images, making dataset preprocessing much easier and removes the need for cropping, squishing, or padding of images.

Split of classes:
[10400 11120] - Train
[4460 4540] - Test
This is a very well balanced dataset, and class weights can probably be ignored for the loss function. 
Considering this classification problem, cross entropy loss will be used as this loss function.

### Data augmentation
To improve generalisation and reduce overfitting from the model, data augmentation techniques were applied to the training set before training.
These were performed as follows:

* Horizontal flip
* Rotation
* Colour 
* Affine
* Gaussian Blur

## Results

## References
Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T. and Xie, S. (2022). A ConvNet for the 2020s. [online] Available at: https://arxiv.org/pdf/2201.03545.