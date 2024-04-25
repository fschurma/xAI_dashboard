from torchvision import transforms
from PIL import Image
from captum.attr import LayerGradCam, FeatureAblation, Saliency, Lime
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np
import torch
from models import fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenetv3_large
import platform
        
        
device = "cuda:0" if torch.cuda.is_available() else "cpu"

if platform.system() == 'Darwin':
    image_path = f'assets/images/demo_picture.png'
elif platform.system() == 'Windows':
    image_path = f'assets\images\demo_picture.png'



input_image = Image.open(image_path)
input_image = input_image.resize((int(input_image.width/2), int(input_image.height/2)), resample=Image.LANCZOS)
preprocessing = transforms.Compose([transforms.ToTensor()])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

input_tensor = preprocessing(input_image)

normalized_inp = normalize(input_tensor).unsqueeze(0).to(device)
normalized_inp.requires_grad = True


def grad_cam(model, label, input_tensor=input_tensor, normalized_inp=normalized_inp):
            
    
    def outputs(normalized_inp, model):
                        
        out = model(normalized_inp)['out']
        out_max = torch.argmax(out, dim=1, keepdim=True)

        return  out_max


    out_max = outputs(normalized_inp, model)


    def agg_segmentation_wrapper_grad(inp):
        model_out = model(inp)['out']
        # Creates binary matrix with 1 for original argmax class for each pixel
        # and 0 otherwise. Note that this may change when the input is ablated
        # so we use the original argmax predicted above, out_max.
        selected_inds = torch.zeros_like(model_out[0:1]).scatter_(1, out_max, 1)
        return (model_out * selected_inds).sum(dim=(2,3))



    def grad_on_image():
        targets = [2,6,7,14,15,19]

        layer_gc = LayerGradCam(agg_segmentation_wrapper_grad, model.classifier)

        for target in targets:
            if target in [label]:

                gc_attr = layer_gc.attribute(normalized_inp, target=target)
                attr_norm = (gc_attr - gc_attr.min()) / (gc_attr.max() - gc_attr.min())
                attr_norm_float = attr_norm.detach().numpy().astype(np.float32)
                heatmap = attr_norm_float[0,0]
                heatmap_fin = cv2.resize(heatmap, (input_tensor.shape[2], input_tensor.shape[1]))

                gradcam_img = show_cam_on_image(np.transpose(input_tensor.detach().cpu().numpy(), (1,2,0)), heatmap_fin, use_rgb=True)

                gradcam_img = Image.fromarray(gradcam_img)

                return gradcam_img
                                    
    gradcam_on_image = grad_on_image()

    return gradcam_on_image


def feature_ablation(model, label, input_tensor=input_tensor, normalized_inp=normalized_inp):
            
    
    def outputs(normalized_inp, model):
                        
        out = model(normalized_inp)['out']
        out_max = torch.argmax(out, dim=1, keepdim=True)

        return  out_max


    out_max = outputs(normalized_inp, model)


    def agg_segmentation_wrapper_abl(inp):
        model_out = model(inp)['out']
        # Creates binary matrix with 1 for original argmax class for each pixel
        # and 0 otherwise. Note that this may change when the input is ablated
        # so we use the original argmax predicted above, out_max.
        selected_inds = torch.zeros_like(model_out[0:1]).scatter_(1, out_max, 1)
        return (model_out * selected_inds).sum(dim=(2,3))


    def feature_ablation_on_image():
        targets = [2,6,7,14,15,19]

        fa = FeatureAblation(agg_segmentation_wrapper_abl)

        for target in targets:
            if target in [label]:

                fa_attr = fa.attribute(normalized_inp, feature_mask=out_max, perturbations_per_eval=4, target=target)
                fa_attr_norm = (fa_attr - fa_attr.min()) / (fa_attr.max() - fa_attr.min())
                fa_attr_float = fa_attr_norm.detach().numpy().astype(np.float32)
                fa_heatmap = fa_attr_float[0, 0] 
                fa_heatmap_fin = cv2.resize(fa_heatmap, (input_tensor.shape[2], input_tensor.shape[1]))

                # preprox_img is the original image --> bring it to the right format
                # image 2
                featureablation_img = show_cam_on_image(np.transpose(input_tensor.detach().cpu().numpy(), (1, 2, 0)), fa_heatmap_fin, use_rgb=True)

                ablation_img = Image.fromarray(featureablation_img)

                return ablation_img
                                    
    ablation_on_image = feature_ablation_on_image()

    return ablation_on_image

def saliency_maps(model, label, input_tensor=input_tensor, normalized_inp=normalized_inp):
    
    def outputs(normalized_inp, model):
                        
        out = model(normalized_inp)['out']
        out_max = torch.argmax(out, dim=1, keepdim=True)

        return  out_max


    out_max = outputs(normalized_inp, model)


    def agg_segmentation_wrapper_sy(inp):
        model_out = model(inp)['out']
        # Creates binary matrix with 1 for original argmax class for each pixel
        # and 0 otherwise. Note that this may change when the input is ablated
        # so we use the original argmax predicted above, out_max.
        selected_inds = torch.zeros_like(model_out[0:1]).scatter_(1, out_max, 1)
        return (model_out * selected_inds).sum(dim=(2,3))
    
    
    # Durchlauf für jedes Ziel


    def saliency_on_image ():
        targets = [2,6,7,14,15,19]

        sy = Saliency(agg_segmentation_wrapper_sy)

        for target in targets:
            if target in [label]:

                sy_attr = sy.attribute(normalized_inp, target=target)
                sy_attr_norm = (sy_attr - sy_attr.min()) / (sy_attr.max() - sy_attr.min())
                sy_attr_float = sy_attr_norm.detach().numpy().astype(np.float32)
                sy_heatmap = sy_attr_float[0, 0] 
                sy_heatmap_fin = cv2.resize(sy_heatmap, (input_tensor.shape[2], input_tensor.shape[1]))

                saliency_img = show_cam_on_image(np.transpose(input_tensor.detach().cpu().numpy(), (1, 2, 0)), sy_heatmap_fin, use_rgb=True, image_weight=0.4)

                saliency_img = Image.fromarray(saliency_img)

                return saliency_img

    saliency_on_image = saliency_on_image()

    return saliency_on_image


def lime (model, label, input_tensor=input_tensor, normalized_inp=normalized_inp):
    
    def outputs(normalized_inp, model):
                        
        out = model(normalized_inp)['out']
        out_max = torch.argmax(out, dim=1, keepdim=True)

        return  out_max


    out_max = outputs(normalized_inp, model)


    def agg_segmentation_wrapper_lime(inp):
        model_out = model(inp)['out']
        # Creates binary matrix with 1 for original argmax class for each pixel
        # and 0 otherwise. Note that this may change when the input is ablated
        # so we use the original argmax predicted above, out_max.
        selected_inds = torch.zeros_like(model_out[0:1]).scatter_(1, out_max, 1)
        return (model_out * selected_inds).sum(dim=(2,3))
    

    
    def lime_on_image():
        targets = [2,6,7,14,15,19]

        lime = Lime(agg_segmentation_wrapper_lime)

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
                                    
    lime_on_image = lime_on_image()

    return lime_on_image
    

                
                



