## This file contains the code to predict the output of the image using the pretrained models
#importing the required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

def predict_fcn_resnet101(image_path):

    """
    Function to predict the output of the image using the FCN Resnet101 model.
    """

    # loading the model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet101', pretrained=True)

    # setting the model to evaluation mode
    model.eval()

    # loading the input image
    input_image = Image.open(image_path)

    # converting the image to RGB
    input_image = input_image.convert("RGB")

    # set preprocessing for image
    preprocess = transforms.Compose([
        # converting the image to tensor
        transforms.ToTensor(),
        # normalizing the image
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # preprocessing the input image
    input_tensor = preprocess(input_image)
    # adding the batch dimension
    input_batch = input_tensor.unsqueeze(0)

    # checking if the GPU is available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # predicting the output
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    
    # getting the predictions
    output_predictions = output.argmax(0)

    # returning the input image and the output predictions
    return input_image , output_predictions

def predict_fcn_resnet50(image_path):

    """
    Function to predict the output of the image using the FCN Resnet50 model.
    """

    # loading the model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)

    # setting the model to evaluation mode
    model.eval()

    # loading the input image
    input_image = Image.open(image_path)

    # converting the image to RGB
    input_image = input_image.convert("RGB")

    # set preprocessing for image
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # preprocessing the input image
    input_tensor = preprocess(input_image)
    # adding the batch dimension
    input_batch = input_tensor.unsqueeze(0)

    # checking if the GPU is available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    
    # getting the predictions
    output_predictions = output.argmax(0)

    # returning the input image and the output predictions
    return input_image , output_predictions

def predict_deeplabv3_resnet50(image_path):

    """
    Function to predict the output of the image using the DeepLabV3 Resnet50 model.
    """

    # loading the model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)

    # setting the model to evaluation mode
    model.eval()

    # loading the input image
    input_image = Image.open(image_path)

    # converting the image to RGB
    input_image = input_image.convert("RGB")

    # set preprocessing for image
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # preprocessing the input image
    input_tensor = preprocess(input_image)
    # adding the batch dimension
    input_batch = input_tensor.unsqueeze(0)

    # checking if the GPU is available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # predicting the output
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # returning the input image and the output predictions
    return input_image , output_predictions

def predict_deeplabv3_resnet101(image_path):
    """
    Function to predict the output of the image using the DeepLabV3 Resnet101 model.
    """

    # loading the model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)

    # setting the model to evaluation mode
    model.eval()

    # loading the input image
    input_image = Image.open(image_path)

    # converting the image to RGB
    input_image = input_image.convert("RGB")

    # set preprocessing for image
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # preprocessing the input image
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # checking if the GPU is available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # predicting the output
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    
    # getting the predictions
    output_predictions = output.argmax(0)

    # returning the input image and the output predictions
    return input_image , output_predictions

# Function to predict the output of the image using the DeepLabV3 MobilenetV3 Small model.
def predict_deeplabv3_mobilenetv3_large(image_path):
    
    """
    Function to predict the output of the image using the DeepLabV3 MobilenetV3 Large model.
    """
    
    # loading the model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)

    # setting the model to evaluation mode
    model.eval()

    # loading the input image
    input_image = Image.open(image_path)

    # converting the image to RGB
    input_image = input_image.convert("RGB")

    # set preprocessing for image
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # preprocessing the input image
    input_tensor = preprocess(input_image)
    # adding the batch dimension
    input_batch = input_tensor.unsqueeze(0)

    # checking if the GPU is available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # predicting the output
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    
    # getting the predictions
    output_predictions = output.argmax(0)

    # returning the input image and the output predictions
    return input_image , output_predictions
