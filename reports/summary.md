# Brain Tumor Detection Project

## Overview

This project uses a Convolutional Neural Network (CNN) model to classify brain MRI images into categories indicating the presence of a brain tumor.

## Project Structure

- `data/`: Directory for training and testing data.
- `model/`: Folder containing the saved model.
- `src/`: Contains the data preprocessing, model training, utility, and Flask app code.
- `reports/`: Directory for reports, including images and summary.

## Model

The model is a CNN with multiple convolutional and max-pooling layers, followed by dense layers for classification.

## Deployment

The model is deployed using Flask to provide a RESTful API for making predictions.

## Results

The model achieved high accuracy on the validation set. See the `images/training_history.png` for training and validation performance.