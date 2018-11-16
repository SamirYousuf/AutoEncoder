# LT2316 H18 Assignment 2

Git project for implementing assignment 2 in [Asad Sayeed's](https://asayeed.github.io) machine learning class in the University of Gothenburg's Masters
of Language Technology programme.

## Your notes

Place instructions for running your project here, as per the assignment
description in GUL.

## Autoencoder architecture 


###### Input Layer
    - Input
###### Encoder Layer
    - Convolution Layer
    - MaxPooling
    - Dropout
    - Convolution Layer
    - MaxPooing
    - Convolution Layer
###### Decoder Layer
    - Convolution layer
    - UpSampling
    - Convolution layer
    - UpSampling
    - Convolution layer
    - UpSampling
    - Convolution Layer
    - Flatten
    - Dense layer
    - Output Layer (Dense layer)
    
## Description of architecture and parameters

In this autoencoder model I used multiple convolution layer for both encoding and decoding to see the results with deep network. Parameters are

    - *Batch size of 50 and 100
    - Epoch = 30
    - Steps per epoch = 10
    - Dropout of 0.1 and 0.5
    - Input layer with shape (200, 200, 3)*

## Architecture and different parameter results

There is an pdf file in this repo with plots of the loss results with different parameters and also the prediction results

## Model files

There are 4 model files also included in the repo that can be used for test

    - *autoencoderB50D01.h5
    - autoencoderB100D01.h5
    - autoencoderB50D05.h5
    - autoencoderB100D05.h5*
    
## Results and disadvantages of using training data for test

The results from the different model shows that the batch size and dropout are important parameters to understand the model and how it can be improved. To understand how well the model is performing we need to use test data that is not in the training data
