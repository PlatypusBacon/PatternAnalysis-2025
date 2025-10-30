# Alzheimer's Disease Classification using ConvNeXt

## Project Description
This project focuses on classifying Alzheimer's disease (AD) as well as normal cognition brain scans (NC) using MRI data from the alzheimer's disease neuroimaging initiative (ADNI) dataset. This task was attempted using a ConvNeXt convolutional neural network (CNN), a modern architecture able to achieve excellent performance on image recognition tasks while keeping efficiency.

## Model Architecture

ConvNeXt is structured combining aspects of convolutional neural networks (CNNs), as well as successful elements from Vision Transformers (ViTs). This is done through several innovations, namely the patchify stem which uses a 4x4 (stride 4) convolution rather than a 7x7 as the first stage, resembling the patches of ViTs.
The network also uses GELU activations instead of ReLU, and Layer in in place of Batch Normalisation [1]

The full ConvNeXt architecture can be seen below:

![ConvNeXt architecture](ConvNeXt-structure-1.webp)

This shows several other key innovations in ConvNeXt, namely the four stage structure of ConvNeXt blocks.

### ConvNeXt Block

The ConvNeXt block is defined as seen below:

![ConvNeXt block](ConvNeXt-structure-2.webp)

### ConvNeXt class

In `modules.py`, the class is defined with a stem, stages, normalisation and head. This performs operations in the following stages:

1. The stem performs Conv2d and LayerNorm2d
2. Stages perform downsampling and then apply a ConvNeXt block (no downsample on first layer)
    1. In the ConvNeXt block, first a depthwise conv is performed (Conv2d(groups=dim))
    2. LayerNorm2d: Applies layer normalisation over mini-batch
    3. Conv2d: Applies 2d convolution over input signal
    4. GELU: Gaussian Error Linear units function
    5. Conv2d
    6. Layer Scale: learnable scaling parameter
    7. Drop path: stochastic depth, drops entire residual connections
3. LayerNorm2d
4. Linear: final classification layer

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
For a ConvNeXt architecture, this is a relatively small dataset, meaning model size may have to be reduced.

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
These were performed as follows. These represent noise that may be present in actual MRI images for training as explained below:

* Horizontal flip - used as images may be flipped in test set
* Rotation - used as images may be ooriented differently
* Colour Jitter - used to brighten sections of MRI for noise
* Affine - used as scale may change across images
* Gaussian Blur - used as resolution may change or sections may be blurred

These augmentations change the images loaded and allow for a much more regularised training, due to the smaller dataset used.
## Training
To train the ConvNeXt model, the following command is used:

`python train.py [--checkpoint checkpoint_path]`

This trains the model on the ADNI dataset, saving the model as checkpoints to `/checkpoints/{model name}`. If the checkpoint argument is included, this resumes training from the specified checkpoint.
The training script additionally saves a confusion matrix and training results images to the save directory

## Predicting
To predict results with the ConvNeXt model, this can be done using the following command.

`python predict.py [--model model_name] [--output output_path] [--image image_path] [--directory directory_path]`

where image_path or diretory_path are optional such that one must be included. This either predicts all images in a directory or only the one image passed in. The output_path specifies where prediction results will be saved to.
## Results
### Early Results
Initial Training of the model resulted in accuracies of >80%, peaking at around 75%. This was due to overfitting of the training set, which can be seen looking at the epochs over training.

![alt text](checkpoints/convnext_small_20251020_095048/training_history.png)

This shows training accuracy continuing to rise while validation accuracy fluctuates. To fix this, several attempts were made. Firstly the model size was reduced as this is a possible contributing factor. However, this did not show any significant improvements. The drop path was then greatly increased, where the model was found to never train successfully, i.e. beat random chance or to overfit again. This was also extended with a dropout, mixup and gaussian noise additions with many epochs to attampt to train, and the model still peaked at ~75% accuracy.

Next a different approach was taken, where a much larger model was implemented, due to the large training dataset. This was done with $\frac{5}{3}$ times dimensions size, 

### Final Results

### Plots

## References
[1] Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T. and Xie, S. (2022). A ConvNet for the 2020s. [online] Available at: https://arxiv.org/pdf/2201.03545.