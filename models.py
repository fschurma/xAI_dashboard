import torch
import torchvision.models as models

def fcn_resnet101():
    return models.segmentation.fcn_resnet101(weights=False, num_classes=20, progress=True)

def fcn_resnet50():
    return models.segmentation.fcn_resnet50(weights=False, num_classes=20, progress=True)

def deeplabv3_resnet101():
    return models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=20, progress=True)

def deeplabv3_resnet50():
    return models.segmentation.deeplabv3_resnet50(pretained=False, num_classes=20, progress=True)

def deeplabv3_mobilenet_v3_large():
    return models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=False, num_classes=20, progress=True)

def lraspp_mobilenet_v3_large():
    return models.segmentation.lraspp_mobilenet_v3_large(pretrained=False, num_classes=20, progress=True)

