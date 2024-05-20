## Description: This file contains the methods for the different interpretability methods.
# Import necessary libraries
from torchvision import transforms
from PIL import Image
from captum.attr import LayerGradCam, FeatureAblation, Saliency, Lime
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np
import torch
from models import fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenetv3_large
import platform
        
# Check if GPU is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Check the operating system
if platform.system() == 'Darwin':
    image_path = f'assets/images/demo_picture.png'
elif platform.system() == 'Windows':
    image_path = f'assets\images\demo_picture.png'
elif platform.system() == 'Linux':
    image_path = f'assets/images/demo_picture.png'


# Open the image and resize it
input_image = Image.open(image_path)
input_image = input_image.resize((int(input_image.width/2), int(input_image.height/2)), resample=Image.LANCZOS)

# Define the preprocessing of the image
preprocessing = transforms.Compose([transforms.ToTensor()])

# set normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Preprocess the image
input_tensor = preprocessing(input_image)

# Normalize the image
normalized_inp = normalize(input_tensor).unsqueeze(0).to(device)
normalized_inp.requires_grad = True


def grad_cam(model, label, input_tensor=input_tensor, normalized_inp=normalized_inp):
    '''
    Function to generate Grad-CAM visualizations
    '''
    
    def outputs(normalized_inp, model):
        '''
        Function to get the output of the model.
        '''                    
        out = model(normalized_inp)['out']
        out_max = torch.argmax(out, dim=1, keepdim=True)

        return  out_max

    # Get the output maximum
    out_max = outputs(normalized_inp, model)


    def agg_segmentation_wrapper_grad(inp):
        '''
        Function to aggregate the segmentation.
        '''
        model_out = model(inp)['out']
        # Creates binary matrix with 1 for original argmax class for each pixel
        # and 0 otherwise. Note that this may change when the input is ablated
        # so we use the original argmax predicted above, out_max.
        selected_inds = torch.zeros_like(model_out[0:1]).scatter_(1, out_max, 1)
        return (model_out * selected_inds).sum(dim=(2,3))



    def grad_on_image():
        '''
        Function to show grad cam on the image, depending on the label.
        '''
        # Define the targets, depending on the number they got from PyTorch pretrained models.
        targets = [2,6,7,14,15,19]

        # Define the LayerGradCam object with the aggregation function.
        layer_gc = LayerGradCam(agg_segmentation_wrapper_grad, model.classifier)

        for target in targets:
            if target in [label]:

                gc_attr = layer_gc.attribute(normalized_inp, target=target)
                attr_norm = (gc_attr - gc_attr.min()) / (gc_attr.max() - gc_attr.min())
                attr_norm_float = attr_norm.detach().numpy().astype(np.float32)
                heatmap = attr_norm_float[0,0]
                heatmap_fin = cv2.resize(heatmap, (input_tensor.shape[2], input_tensor.shape[1]))

                # input_tensor is the original image --> bring it to the right format
                gradcam_img = show_cam_on_image(np.transpose(input_tensor.detach().cpu().numpy(), (1,2,0)), heatmap_fin, use_rgb=True)

                gradcam_img = Image.fromarray(gradcam_img)

                return gradcam_img

    # Call the function                               
    gradcam_on_image = grad_on_image()

    return gradcam_on_image


def feature_ablation(model, label, input_tensor=input_tensor, normalized_inp=normalized_inp):
    '''
    Function to generate Feature Ablation visualizations
    '''
    
    def outputs(normalized_inp, model):
        '''
        Function to get the output of the model.
        '''
                        
        out = model(normalized_inp)['out']
        out_max = torch.argmax(out, dim=1, keepdim=True)

        return  out_max

    # Get the output maximum
    out_max = outputs(normalized_inp, model)


    def agg_segmentation_wrapper_abl(inp):
        '''
        Function to aggregate the segmentation.
        '''

        model_out = model(inp)['out']
        # Creates binary matrix with 1 for original argmax class for each pixel
        # and 0 otherwise. Note that this may change when the input is ablated
        # so we use the original argmax predicted above, out_max.
        selected_inds = torch.zeros_like(model_out[0:1]).scatter_(1, out_max, 1)
        return (model_out * selected_inds).sum(dim=(2,3))


    def feature_ablation_on_image():
        '''
        Function to show feature ablation on the image, depending on the label.
        '''
        # Define the targets, depending on the number they got from PyTorch pretrained models.
        targets = [2,6,7,14,15,19]

        # Define the FeatureAblation object with the aggregation function.
        fa = FeatureAblation(agg_segmentation_wrapper_abl)

        for target in targets:
            if target in [label]:

                fa_attr = fa.attribute(normalized_inp, feature_mask=out_max, perturbations_per_eval=4, target=target)
                fa_attr_norm = (fa_attr - fa_attr.min()) / (fa_attr.max() - fa_attr.min())
                fa_attr_float = fa_attr_norm.detach().numpy().astype(np.float32)
                fa_heatmap = fa_attr_float[0, 0] 
                fa_heatmap_fin = cv2.resize(fa_heatmap, (input_tensor.shape[2], input_tensor.shape[1]))

                # input_tensor is the original image --> bring it to the right format
                featureablation_img = show_cam_on_image(np.transpose(input_tensor.detach().cpu().numpy(), (1, 2, 0)), fa_heatmap_fin, use_rgb=True)

                ablation_img = Image.fromarray(featureablation_img)

                return ablation_img
                                    
    # Call the function
    ablation_on_image = feature_ablation_on_image()

    return ablation_on_image

def saliency_maps(model, label, input_tensor=input_tensor, normalized_inp=normalized_inp):
    '''
    Function to generate Saliency Maps visualizations
    '''
    
    def outputs(normalized_inp, model):
        '''
        Function to get the output of the model.
        '''
                        
        out = model(normalized_inp)['out']
        out_max = torch.argmax(out, dim=1, keepdim=True)

        return  out_max

    # Get the output maximum
    out_max = outputs(normalized_inp, model)


    def agg_segmentation_wrapper_sy(inp):
        '''
        Function to aggregate the segmentation.
        '''
        model_out = model(inp)['out']
        # Creates binary matrix with 1 for original argmax class for each pixel
        # and 0 otherwise. Note that this may change when the input is ablated
        # so we use the original argmax predicted above, out_max.
        selected_inds = torch.zeros_like(model_out[0:1]).scatter_(1, out_max, 1)
        return (model_out * selected_inds).sum(dim=(2,3))
    


    def saliency_on_image ():
        '''
        Function to show saliency maps on the image, depending on the label.
        '''
        # Define the targets, depending on the number they got from PyTorch pretrained models.
        targets = [2,6,7,14,15,19]

        # Define the Saliency object with the aggregation function.
        sy = Saliency(agg_segmentation_wrapper_sy)

        for target in targets:
            if target in [label]:

                sy_attr = sy.attribute(normalized_inp, target=target)
                sy_attr_norm = (sy_attr - sy_attr.min()) / (sy_attr.max() - sy_attr.min())
                sy_attr_float = sy_attr_norm.detach().numpy().astype(np.float32)
                sy_heatmap = sy_attr_float[0, 0] 
                sy_heatmap_fin = cv2.resize(sy_heatmap, (input_tensor.shape[2], input_tensor.shape[1]))

                # input_tensor is the original image --> bring it to the right format
                saliency_img = show_cam_on_image(np.transpose(input_tensor.detach().cpu().numpy(), (1, 2, 0)), sy_heatmap_fin, use_rgb=True, image_weight=0.4)

                saliency_img = Image.fromarray(saliency_img)

                
                return saliency_img
    
    # Call the function
    saliency_on_image = saliency_on_image()

    return saliency_on_image


def lime (model, label, input_tensor=input_tensor, normalized_inp=normalized_inp):
    '''
    Function to generate Lime visualizations
    '''
    
    def outputs(normalized_inp, model):
        '''
        Function to get the output of the model.
        '''
                        
        out = model(normalized_inp)['out']
        out_max = torch.argmax(out, dim=1, keepdim=True)

        return  out_max

    # Get the output maximum
    out_max = outputs(normalized_inp, model)


    def agg_segmentation_wrapper_lime(inp):
        '''
        Function to aggregate the segmentation.
        '''
        model_out = model(inp)['out']
        # Creates binary matrix with 1 for original argmax class for each pixel
        # and 0 otherwise. Note that this may change when the input is ablated
        # so we use the original argmax predicted above, out_max.
        selected_inds = torch.zeros_like(model_out[0:1]).scatter_(1, out_max, 1)
        return (model_out * selected_inds).sum(dim=(2,3))
    

    
    def lime_on_image():
        '''
        Function to show lime on the image, depending on the label.
        '''
        # Define the targets, depending on the number they got from PyTorch pretrained models
        targets = [2,6,7,14,15,19]

        # Define the Lime object with the aggregation function
        lime = Lime(agg_segmentation_wrapper_lime)

        # Define the baselines
        baselines = torch.zeros_like(normalized_inp)

        for target in targets:
            if target in [label]:

                lime_attr = lime.attribute(normalized_inp, target=target, feature_mask=out_max, baselines=baselines, n_samples=20)
                lime_attr_norm = (lime_attr - lime_attr.min()) / (lime_attr.max() - lime_attr.min())
                lime_attr_float = lime_attr_norm.detach().numpy().astype(np.float32)
                lime_heatmap = lime_attr_float[0, 0]
                lime_heatmap_fin = cv2.resize(lime_heatmap, (input_tensor.shape[2], input_tensor.shape[1]))

                lime_img = show_cam_on_image(np.transpose(input_tensor.detach().cpu().numpy(), (1, 2, 0)), lime_heatmap_fin, use_rgb=True, image_weight=0.4)

                lime_img = Image.fromarray(lime_img)

                return lime_img
                                    
    # Call the function
    lime_on_image = lime_on_image()

    return lime_on_image
    

                
                



