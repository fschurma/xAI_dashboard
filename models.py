##Description: This script contains functions to load the pre-trained models from the PyTorch model zoo.
# import necessary libraries
import torch

# check if GPU is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"



def fcn_resnet50():
    '''
    Function to load the FCN ResNet50 model from the PyTorch model zoo
    '''
    
    fcn_resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True).to(device).eval()
    
    return fcn_resnet50

def fcn_resnet101():
    '''
    Function to load the FCN ResNet101 model from the PyTorch model zoo
    '''
    
    fcn_resnet101 = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet101', pretrained=True).to(device).eval()
    
    return fcn_resnet101

def deeplabv3_resnet50():
    '''
    Function to load the DeepLabV3 ResNet50 model from the PyTorch model zoo
    '''
    
    deeplabv3_resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).to(device).eval()
    
    return deeplabv3_resnet50

def deeplabv3_resnet101():
    '''
    Function to load the DeepLabV3 ResNet101 model from the PyTorch model zoo
    '''
    
    deeplabv3_resnet101 = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True).to(device).eval()
   
    return deeplabv3_resnet101

def deeplabv3_mobilenetv3_large():
    '''
    Function to load the DeepLabV3 MobileNetV3 Large model from the PyTorch model zoo
    '''
    
    deeplabv3_mobilenetv3_large = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True).to(device).eval()

    return deeplabv3_mobilenetv3_large
