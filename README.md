# Malaria Cell Image Classification Project

## Overview

This project aims to classify cell images as either parasitized or uninfected using a Convolutional Neural Network (CNN). The dataset consists of cell images with labels indicating whether the cells are parasitized by the malaria parasite or not. The goal is to build and train a CNN model that can accurately classify the cell images.

![image](https://github.com/Veto2922/Cells-image-classification/assets/114834171/6b9850bf-6287-45a1-9103-cc8cc51e9ffb)


Data source : https://drive.google.com/file/d/1N1gcN8_5dZVlIejoC00QZLSZFhGoSoQb/view

## Table of Contents

- [Installation](#installation)
- [Data Description](#data-description)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/malaria-cell-classification.git
    ```
2. Navigate to the project directory:
    ```bash
    cd malaria-cell-classification
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Data Description

The dataset is organized into the following directories:

- `cell_images/train/parasitized`: Contains images of parasitized cells.
- `cell_images/train/uninfected`: Contains images of uninfected cells.
- `cell_images/test/parasitized`: Contains images of parasitized cells for testing.
- `cell_images/test/uninfected`: Contains images of uninfected cells for testing.

Each image is a PNG file with dimensions of 130x130 pixels.

## Data Preprocessing

### Importing Libraries
We import necessary libraries such as pandas, matplotlib, numpy, seaborn, and tensorflow.keras.

### Visualizing the Data
We load and visualize sample images from both parasitized and uninfected categories.

### Image Dimensions
We calculate and visualize the average dimensions of the images in the dataset.

### Image Data Generator
We set up an ImageDataGenerator for data augmentation, which includes operations such as rotation, width and height shifts, shear, zoom, and horizontal flip. The images are also rescaled.

## Model Architecture

The model is built using Keras and TensorFlow. It consists of multiple convolutional layers followed by max-pooling layers, a flattening layer, dense layers, and a dropout layer to prevent overfitting. The final layer uses a sigmoid activation function for binary classification.

## Training the Model

### Early Stopping
We use early stopping to monitor the validation loss and stop training if the model does not improve for a certain number of epochs.

### Model Training
The model is trained on the training dataset and validated on the test dataset. Training is performed for a specified number of epochs or until early stopping criteria are met.

## Evaluation

### Model Loading and Evaluation
The trained model is loaded and evaluated on the test dataset to measure its performance.

### Classification Report and Confusion Matrix
We generate a classification report and confusion matrix to assess the model's performance in terms of precision, recall, F1-score, and accuracy.

## Usage

### Predicting on a New Image
We demonstrate how to preprocess a new image and use the trained model to predict whether the cell is parasitized or uninfected.

## Results

The model achieves a certain level of accuracy on the test dataset. Detailed classification metrics, including precision, recall, and F1-score, are provided in the evaluation section.


