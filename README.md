# Garbage Image Classification Project

This project is a deep learning-based image classification system designed to classify images of garbage into six categories: cardboard, glass, metal, paper, plastic, and trash.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Contributing](#contributing)

## Overview

In this project, I have developed a convolutional neural network (CNN) using TensorFlow and Keras to classify garbage images. The CNN model is trained on a dataset containing labeled images of different types of garbage.

## Dataset

The training dataset consists of images of garbage items categorized into six classes: cardboard, glass, metal, paper, plastic, and trash. The dataset is divided into a training set and a validation set. 

## Getting Started

To get started with this project, follow the instructions below.

### Prerequisites

Before you begin, you need to have the following software and libraries installed:

- Python 3.x
- TensorFlow
- Keras
- Matplotlib
- pandas

### Installation

1. Clone this repository to your local machine:

```
git clone https://github.com/your-username/garbage-classification.git
```

2. Install the required Python libraries:

```
pip install tensorflow keras matplotlib pandas
```

3. Download the training dataset and place it in the appropriate directory:

```
- /content/drive/MyDrive/*FOLERNAME*/TRAIN
    - cardboard/
    - glass/
    - metal/
    - paper/
    - plastic/
    - trash/
```

## Training the Model

To train the model, run the Jupyter Notebook file:

```
train_garbage_classifier.ipynb
```

This notebook contains the code for training the garbage classification model using the provided dataset.

## Evaluation

The model's performance can be evaluated by running the evaluation code in the notebook. The accuracy and loss of the model on the validation set are displayed.

## Inference

To make predictions on new garbage images, you can use the trained model. You need to provide a folder containing the test images and use the inference code provided in the notebook to classify the images.

## Contributing

If you want to contribute to this project, you can fork the repository, make changes, and submit a pull request. Your contributions are welcome!
