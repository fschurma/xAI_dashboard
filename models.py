import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def fcn_resnet101():
    
    fcn_resnet101 = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet101', pretrained=True).to(device).eval()
    
    return fcn_resnet101

def fcn_resnet50():
    
    fcn_resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True).to(device).eval()
    
    return fcn_resnet50

def deeplabv3_resnet50():
    
    deeplabv3_resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).to(device).eval()
    
    return deeplabv3_resnet50

def deeplabv3_resnet101():
    
    deeplabv3_resnet101 = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True).to(device).eval()
   
    return deeplabv3_resnet101

def deeplabv3_mobilenetv3_large():
    
    deeplabv3_mobilenetv3_large = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True).to(device).eval()

    return deeplabv3_mobilenetv3_large
