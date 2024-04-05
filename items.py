import dash_bootstrap_components as dbc

def items_models_top_bar():
    return [
        dbc.DropdownMenuItem('FCN ResNet50', id='fcn-resnet50_t'),
        dbc.DropdownMenuItem('SegNet', id='SegNet_t'),
        dbc.DropdownMenuItem('DeepLabV3 ResNet50', id='deeplabv3-resnet50_t'),
        dbc.DropdownMenuItem('DeepLabV3 ResNet101', id='deeplabv3-resnet101_t'),
        dbc.DropdownMenuItem('DeepLabV3 MobileNetV3-Large', id='deeplabv3-mobilenetv3-large_t'),
        dbc.DropdownMenuItem('LR-ASPP MobileNetV3-Large', id='lr-aspp-mobilenetv3-large_t')
        ]


def items_models_card (): 
    return [
        dbc.DropdownMenuItem('FCN ResNet50', id='fcn-resnet50_c'),
        dbc.DropdownMenuItem('SegNet', id='SegNet_c'),
        dbc.DropdownMenuItem('DeepLabV3 ResNet50', id='deeplabv3-resnet50_c'),
        dbc.DropdownMenuItem('DeepLabV3 ResNet101', id='deeplabv3-resnet101_c'),
        dbc.DropdownMenuItem('DeepLabV3 MobileNetV3-Large', id='deeplabv3-mobilenetv3-large_c'),
        dbc.DropdownMenuItem('LR-ASPP MobileNetV3-Large', id='lr-aspp-mobilenetv3-large_c')
        ]


def items_models_filter_section1():
    return [
        dbc.DropdownMenuItem('FCN ResNet50', id='fcn-resnet50_f1'),
        dbc.DropdownMenuItem('FCN ResNet101', id='fcn-resnet101_f1'),
        dbc.DropdownMenuItem('DeepLabV3 ResNet50', id='deeplabv3-resnet50_f1'),
        dbc.DropdownMenuItem('DeepLabV3 ResNet101', id='deeplabv3-resnet101_f1'),
        dbc.DropdownMenuItem('DeepLabV3 MobileNetV3-Large', id='deeplabv3-mobilenetv3-large_f1')
        ]

def items_models_filter_section2():
    return [
        dbc.DropdownMenuItem('FCN ResNet50', id='fcn-resnet50_f2'),
        dbc.DropdownMenuItem('FCN ResNet101', id='fcn-resnet101_f2'),
        dbc.DropdownMenuItem('DeepLabV3 ResNet50', id='deeplabv3-resnet50_f2'),
        dbc.DropdownMenuItem('DeepLabV3 ResNet101', id='deeplabv3-resnet101_f2'),
        dbc.DropdownMenuItem('DeepLabV3 MobileNetV3-Large', id='deeplabv3-mobilenetv3-large_f2')
        ]

def items_windows():
    return [
        dbc.DropdownMenuItem('Result', id='result'),
        dbc.DropdownMenuItem('Difference', id='difference_top_bar')
        ]

def items_method_f1():
    return [
        dbc.DropdownMenuItem('LayerGradCam', id='layer_grad_cam_f1'),
        dbc.DropdownMenuItem('FeatureAblation', id='fa_f1'),
        dbc.DropdownMenuItem('Saliency Maps', id='saliency_f1'),
        dbc.DropdownMenuItem('LIME', id='lime_f1')
        ]

def items_method_f2():
    return [
        dbc.DropdownMenuItem('LayerGradCam', id='layer_grad_cam_f2'),
        dbc.DropdownMenuItem('FeatureAblation', id='fa_f2'),
        dbc.DropdownMenuItem('Saliency Maps', id='saliency_f2'),
        dbc.DropdownMenuItem('LIME', id='lime_f2')
        ]

def items_labels_f1():
    return [
        dbc.DropdownMenuItem('bicycle', id='bicycle_f1'),
        dbc.DropdownMenuItem('bus', id='bus_f1'),
        dbc.DropdownMenuItem('car', id='car_f1'),
        dbc.DropdownMenuItem('motorbike', id='motorbike_f1'),
        dbc.DropdownMenuItem('person', id='person_f1'),
        dbc.DropdownMenuItem('train', id='train_f1')
        ]

def items_labels_f2():
    return [
        dbc.DropdownMenuItem('bicycle', id='bicycle_f2'),
        dbc.DropdownMenuItem('bus', id='bus_f2'),
        dbc.DropdownMenuItem('car', id='car_f2'),
        dbc.DropdownMenuItem('motorbike', id='motorbike_f2'),
        dbc.DropdownMenuItem('person', id='person_f2'),
        dbc.DropdownMenuItem('train', id='train_f2')
        ]
