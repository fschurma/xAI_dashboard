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


def items_models_filter_section():
    return [
        dbc.DropdownMenuItem('FCN ResNet50', id='fcn-resnet50_f'),
        dbc.DropdownMenuItem('SegNet', id='SegNet_f'),
        dbc.DropdownMenuItem('DeepLabV3 ResNet50', id='deeplabv3-resnet50_f'),
        dbc.DropdownMenuItem('DeepLabV3 ResNet101', id='deeplabv3-resnet101_f'),
        dbc.DropdownMenuItem('DeepLabV3 MobileNetV3-Large', id='deeplabv3-mobilenetv3-large_f'),
        dbc.DropdownMenuItem('LR-ASPP MobileNetV3-Large', id='lr-aspp-mobilenetv3-large_f')
        ]

def items_windows():
    return [
        dbc.DropdownMenuItem('Result', id='result'),
        dbc.DropdownMenuItem('Difference', id='difference')
        ]

def items_method():
    return [
        dbc.DropdownMenuItem('Method 1', id='method_1'),
        dbc.DropdownMenuItem('Method 2', id='method_2'),
        dbc.DropdownMenuItem('Method 3', id='method_3'),
        dbc.DropdownMenuItem('Method 4', id='method_4'),
        dbc.DropdownMenuItem('Method 5', id='method_5'),
        dbc.DropdownMenuItem('Method 6', id='method_6')
        ]

def items_labels():
    return [
        dbc.DropdownMenuItem('Label 1', id='label_1'),
        dbc.DropdownMenuItem('Label 2', id='label_2'),
        dbc.DropdownMenuItem('Label 3', id='label_3'),
        dbc.DropdownMenuItem('Label 4', id='label_4'),
        dbc.DropdownMenuItem('Label 5', id='label_5'),
        dbc.DropdownMenuItem('Label 6', id='label_6')
        ]
