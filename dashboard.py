# Importing Libraries
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import items
from predict_models import predict_fcn_resnet101, predict_fcn_resnet50, predict_deeplabv3_resnet50, predict_deeplabv3_resnet101, predict_deeplabv3_mobilenetv3_large
import torch
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import io
from methods import grad_cam, feature_ablation, saliency_maps, lime
import models
import platform

# Initialize the Dash app with the Bootstrap theme for dash_bootstrap_components
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Layout

app.layout = html.Div(children=[
        html.Div(children=[
            # Top Bar
            html.P('File'),
            # Show Demo Dropdown Menu
            dbc.DropdownMenu(label='Show Demo',
                            children = [
                                dbc.DropdownMenuItem('Show Demo', id='show_demo', n_clicks=0)
                            ],
                            direction='down',
                            toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                            style={'margin': '5px'}
            ),
            # Add Window Dropdown Menu
            dbc.DropdownMenu(label='Add Window',
                            children = items.items_windows(),
                            direction='down',
                            toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                            style={'margin': '5px'}
            ),
            # Choose Model Dropdown Menu
            dbc.DropdownMenu(label='Choose Model',
                            children=items.items_models_top_bar(),
                            direction='down',
                            toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                            style={'margin': '5px'}
            ),
            # Import Image Dropdown Menu
            dbc.DropdownMenu(label='Import Image', children=[
                            # Import Image Section
                            dbc.DropdownMenuItem(dcc.Upload(html.P('Import Image'), accept='.jpg, .png, .tiff', id='import_image_1'))],
                            direction='down',
                            toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                            style={'margin': '5px'})
                
        ], id='top-bar'),
        # Card Container
        html.Div([
            # This is the right card
            html.Div(id='card_1', children=[
                html.P('Choose Model'),
                html.Img(src='assets/images/neural_network.png', alt='Model Pictogram')
            ]),
            # This is the left card
            html.Div(id='card_2', children=[
                html.P('Import Image'),
                html.Img(src='assets/images/image.png', alt='Image Pictogram', n_clicks=0)      
            ])
        ], id='card_container'),
        # Row Container, where the user can choose the model and import the image
        html.Div([
            # Dropdown Container on the right below the card
            html.Div(id='dropdown_container_1', children=[
                # Dropdown Menu to choose the model
                dbc.DropdownMenu( label='Choose Model',
                children=items.items_models_card(),
                direction='down',
                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0.5px solid black'}
                )]
            ),
            # Dropdown Container on the left below the card
            html.Div(id='dropdown_container_2', children=[
                # Dropdown Menu to import the image
                dbc.DropdownMenu(label='Import Image', children=[
                            dbc.DropdownMenuItem(dcc.Upload(html.P('Import Image'), accept='.jpg, .png, .tiff', id='import_image_2'))],
                            direction='down',
                            toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'})
                    ]),
            
                ], id='row_container'),
            
            # Result Container, that displays if the show demo button is clicked
            html.Div([
                # Result frame on the top left
                html.Div(id='result_1_div', children=[
                    # Filter Container on the top left
                    html.Div(id='filter_container_1', children=[
                        # Dropdown Menu to choose the model inte the filter section on the top left
                        dbc.DropdownMenu(label='Model selection',
                                size='sm',
                                children=items.items_models_filter_section1(),
                                direction='down',
                                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                                style={'margin': '5px'}),
                        # Display the model name if a model is selected
                        html.P(id='model_name_1'),
                        # Dropdown Menu to choose the label in the filter section on the top left
                        dbc.DropdownMenu(label='Label selection',
                                size='sm',
                                children=items.items_labels_f1(),
                                direction='down',
                                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                                style={'margin': '5px'}),
                        # Display the label name if a label is selected
                        html.P(id='label_1'),
                        # Dropdown Menu to choose the method in the filter section on the top left
                        dbc.DropdownMenu(label='Method selection',
                                size='sm',
                                children=items.items_method_f1(),
                                direction='down',
                                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                                style={'margin': '5px'}),
                        # Display the method name if a method is selected
                        html.P(id='method_left')
                                  ]),
                    # Container for the demo picture, model segmentation and the methods heatmaps on the top left
                    html.Div(id='image-upload-container_1', children=[
                        # Loading bar if the results is loading
                        dcc.Loading(id='loading-1', children=[
                            #html.Div(id='output-image-upload_1'),
                            html.Div(id='demo_upload_1'),
                            html.Div(id='output-segmentation-1'),
                            html.Div(id='layer_grad_cam_1'),
                            html.Div(id='fa_1'),
                            html.Div(id='saliency_1'),
                            html.Div(id='lime_1')
                    ], type='default')
                        
                        
                        ]),
                ]),
                # Result frame on the top right
                html.Div(id='result_2_div', children=[
                    # Filter Container on the top right
                    html.Div(id='filter_container_2', children=[
                        # Dropdown Menu to choose the model in the filter section on the top right
                        dbc.DropdownMenu(label='Model selection',
                                size='sm',
                                children=items.items_models_filter_section2(),
                                direction='down',
                                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                                style={'margin': '5px'}),
                        # Display the model name if a model is selected
                        html.P(id='model_name_2'),
                        # Dropdown Menu to choose the label in the filter section on the top right
                        dbc.DropdownMenu(label='Label selection',
                                size='sm',
                                children=items.items_labels_f2(),
                                direction='down',
                                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                                style={'margin': '5px'}),
                        # Display the label name if a label is selected
                        html.P(id='label_2'),
                        # Dropdown Menu to choose the method in the filter section on the top right
                        dbc.DropdownMenu(label='Method selection',
                                size='sm',
                                children=items.items_method_f2(),
                                direction='down',
                                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                                style={'margin': '5px'}),
                        # Display the method name if a method is selected
                        html.P(id='method_right')
                            ]),
                    # Container for the demo picture, model segmentation and the methods heatmaps on the top right
                    html.Div(id='image-upload-container_2', children=[
                        # Loading bar if the results are loading
                        dcc.Loading(id='loading-2', children=[
                            #html.Div(id='output-image-upload_2'),
                            html.Div(id='demo_upload_2'),
                            html.Div(id='output-segmentation-2'),
                            html.Div(id='layer_grad_cam_2'),
                            html.Div(id='fa_2'),
                            html.Div(id='saliency_2'),
                            html.Div(id='lime_2')
                        ], type='default')
                ]),     
            ])
                ], id='result_container'),
            # Difference Container, that displays the difference between the two models
            html.Div([
                # Boundary around the difference container
                html.Div(id='difference_result', children=[
                    # Display the name of both models that are selected
                    html.H4(id='difference_model_names'),
                    # Loading bar if the results are loading
                    dcc.Loading(id='loading-3', children=[
                        # Display the difference between the two models
                        html.Div(
                             id='difference'
                            )
                    ], type='default')
            ]),
                # Container, that displays the difference between the two xAI methods
            html.Div(id='performance_div', children=[
                # Display the name of both xAI methods that are selected
                html.H4(id='difference_method_names'),
                # Loading bar if the results are loading
                dcc.Loading(id='loading-4', children=[
                    # Display the difference between the two xAI methods
                    html.Div(
                        id='difference_xAI'
                        )
                ], type='default')
            ])
        ],id='difference_container')
    ])






#------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Demo, Upload and Results

# Callback to show the demo picture if the dropdown menu is clicked
@app.callback(
    Output('result_1_div', 'style'),
    Output('result_2_div', 'style'),
    Output('demo_upload_1','children'),
    Output('demo_upload_2', 'children'),
    Output('difference_result', 'style'),
    Output('performance_div', 'style'),
    Output('card_container', 'style'),
    Output('row_container', 'style'),
    Input('result', 'n_clicks'),
    # Placeholder for the image upload
    Input('import_image_1', 'contents'),
    Input('import_image_2', 'contents'),
    Input('show_demo', 'n_clicks'),
    allow_duplicate=True
)
def show_result_window_div(n_clicks_1, contents_1, contents_2, n_clicks_demo):
    '''
    Function to show the result window if the show demo button is clicked. Additionally, the placeholder to upload an image is also in this function.
    '''
    # This is the placeholder function to displayed if an individual image is uploaded
    if n_clicks_1 and n_clicks_1 > 0 or contents_1 is not None or contents_2 is not None:
        result_1_div_style = {'display': 'block'}
        result_2_div_style = {'display': 'block'}
        difference_result_style = {'display': 'block'}
        performance_div_style = {'display': 'block'}
        card_container_style = {'display': 'none'}
        row_container_style = {'display': 'none'}

        img1 = ''
        img2 = ''

        return result_1_div_style, result_2_div_style, img1, img2, difference_result_style, performance_div_style, card_container_style, row_container_style

    # This is the part of the function that define what is displayed if the show demo button is clicked
    elif n_clicks_demo and n_clicks_demo > 0:
        result_1_div_style = {'display': 'block'}
        result_2_div_style = {'display': 'block'}
        difference_result_style = {'display': 'block'}
        performance_div_style = {'display': 'block'}
        card_container_style = {'display': 'none'}
        row_container_style = {'display': 'none'}

        # Load the demo picture dependig on the platform the dashboard is running on
        if platform.system() == 'Darwin' or platform.system() == 'Linux':
            img1 = html.Img(src=f'assets/images/demo_picture.png', alt='Image 1')
            img2 = html.Img(src=f'assets/images/demo_picture.png', alt='Image 2')
        elif platform.system() == 'Windows':
            img1 = html.Img(src = f'assets\images\demo_picture.png', alt= 'Image 1')
            img2 = html.Img(src = f'assets\images\demo_picture.png', alt= 'Image 2')

        return result_1_div_style, result_2_div_style, img1, img2, difference_result_style, performance_div_style, card_container_style, row_container_style
    
    else:
        result_1_div_style = {'display': 'none'}
        result_2_div_style = {'display': 'none'}
        difference_result_style = {'display': 'none'}
        performance_div_style = {'display': 'none'}
        card_container_style = {'opacity': 1}
        row_container_style = {'opacity': 1}
        img1 = ''
        img2 = ''

    return result_1_div_style, result_2_div_style, img1, img2, difference_result_style, performance_div_style, card_container_style, row_container_style



#------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Segmentation

# Callback to display the image segmentation results on the left side
@app.callback(
    Output('output-segmentation-1', 'children'),
    Input('fcn-resnet101_f1', 'n_clicks'),
    Input('fcn-resnet50_f1', 'n_clicks'),
    Input('deeplabv3-resnet50_f1', 'n_clicks'),
    Input('deeplabv3-resnet101_f1', 'n_clicks'),
    Input('deeplabv3-mobilenetv3-large_f1', 'n_clicks'),
    allow_duplicate = True
)

def image_segmentation_filter_left(n_clicks_1,  n_clicks_2, n_clicks_3, n_clicks_4, n_clicks_5):
    '''
    Function to display the image segmentation results on the left side depending on the model that is selected in the filter section.
    '''

    # Get the id of the dropdown menu that is clicked
    change_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    
    # Check which model is selected in the filter section
    if 'fcn-resnet101_f1' in change_id:
            
            # Load the image and the segmentation results
            input_image, output_predictions = predict_fcn_resnet101('assets/images/demo_picture.png')
            palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

            # Define the colors for the segmentation results
            colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
            colors = (colors % 255).numpy().astype("uint8")

            # Resize the segmentation results to the size of the input image
            r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
            r.putpalette(colors)

            # Display the segmentation results
            children = html.Img(src=r, alt='Segmented Image')

            return children
    

    # Check which model is selected in the filter section
    elif 'fcn-resnet50_f1' in change_id:
            
            # Load the image and the segmentation results
            input_image, output_predictions = predict_fcn_resnet50('assets/images/demo_picture.png')

            # Define the colors for the segmentation results
            palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
            colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
            colors = (colors % 255).numpy().astype("uint8")

            # Resize the segmentation results to the size of the input image
            r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
            r.putpalette(colors)

            # Display the segmentation results
            children = html.Img(src=r, alt='Segmented Image')

            return children
    
    # Check which model is selected in the filter section
    elif 'deeplabv3-resnet50_f1' in change_id:
                
            # Load the image and the segmentation results
            input_image, output_predictions = predict_deeplabv3_resnet50('assets/images/demo_picture.png')
            
            # Define the colors for the segmentation results
            palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
            colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
            colors = (colors % 255).numpy().astype("uint8")
    
            # Resize the segmentation results to the size of the input image
            r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
            r.putpalette(colors)
            
            # Display the segmentation results
            children = html.Img(src=r, alt='Segmented Image')
    
            return children
    
    # Check which model is selected in the filter section
    elif 'deeplabv3-resnet101_f1' in change_id:
                    
            # Load the image and the segmentation results
            input_image, output_predictions = predict_deeplabv3_resnet101('assets/images/demo_picture.png')
            
            # Define the colors for the segmentation results
            palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
            colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
            colors = (colors % 255).numpy().astype("uint8")
        
            # Resize the segmentation results to the size of the input image
            r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
            r.putpalette(colors)
            
            # Display the segmentation results
            children = html.Img(src=r, alt='Segmented Image')
        
            return children
    
    # Check which model is selected in the filter section
    elif 'deeplabv3-mobilenetv3-large_f1' in change_id:
                            
            # Load the image and the segmentation results
            input_image, output_predictions = predict_deeplabv3_mobilenetv3_large('assets/images/demo_picture.png')
                
            # Define the colors for the segmentation results
            palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
            colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
            colors = (colors % 255).numpy().astype("uint8")
                
            # Resize the segmentation results to the size of the input image
            r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
            r.putpalette(colors)

            # Display the segmentation results
            children = html.Img(src=r, alt='Segmented Image')
                
            return children
    

# Callback to display the image segmentation results on the right side
@app.callback(
    Output('output-segmentation-2', 'children'),
    Input('fcn-resnet101_f2', 'n_clicks'),
    Input('fcn-resnet50_f2', 'n_clicks'),
    Input('deeplabv3-resnet50_f2', 'n_clicks'),
    Input('deeplabv3-resnet101_f2', 'n_clicks'),
    Input('deeplabv3-mobilenetv3-large_f2', 'n_clicks'),
    allow_duplicate = True
)

def image_segmentation_filter_right(n_clicks_1, n_clicks_2, n_clicks_3, n_clicks_4, n_clicks_5):

    # Get the id of the dropdown menu that is clicked
    change_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    # Check which model is selected in the filter section
    if 'fcn-resnet101_f2' in change_id:
            
        # Load the image and the segmentation results
        input_image, output_predictions = predict_fcn_resnet101('assets/images/demo_picture.png')

        # Define the colors for the segmentation results
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

        # Resize the segmentation results to the size of the input image
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
        r.putpalette(colors)

        # Display the segmentation results
        children = html.Img(src=r, alt='Segmented Image')

        return children
    
    # Check which model is selected in the filter section
    elif 'fcn-resnet50_f2' in change_id:
            
        # Load the image and the segmentation results
        input_image, output_predictions = predict_fcn_resnet50('assets/images/demo_picture.png')

        # Define the colors for the segmentation results
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

        # Resize the segmentation results to the size of the input image
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
        r.putpalette(colors)

        # Display the segmentation results
        children = html.Img(src=r, alt='Segmented Image')

        return children
    
    # Check which model is selected in the filter section
    elif 'deeplabv3-resnet50_f2' in change_id:
            
        # Load the image and the segmentation results           
        input_image, output_predictions = predict_deeplabv3_resnet50('assets/images/demo_picture.png')
        
        # Define the colors for the segmentation results
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")
        
        # Resize the segmentation results to the size of the input image
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
        r.putpalette(colors)

        # Display the segmentation results
        children = html.Img(src=r, alt='Segmented Image')
        
        return children
    
    # Check which model is selected in the filter section
    elif 'deeplabv3-resnet101_f2' in change_id:

        # Load the image and the segmentation results                 
        input_image, output_predictions = predict_deeplabv3_resnet101('assets/images/demo_picture.png')
            
        # Define the colors for the segmentation results
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")
                
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
        r.putpalette(colors)

        children = html.Img(src=r, alt='Segmented Image')
                
        return children
    
    elif 'deeplabv3-mobilenetv3-large_f2' in change_id:
            
        # Load the image and the segmentation results                       
        input_image, output_predictions = predict_deeplabv3_mobilenetv3_large('assets/images/demo_picture.png')
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
                        
        # Define the colors for the segmentation results
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")
                        
        # Resize the segmentation results to the size of the input image
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
        r.putpalette(colors)

        # Display the segmentation results
        children = html.Img(src=r, alt='Segmented Image')

        return children
    
                        
    
# Callback to display the name model, that is chosen in the left filter section, in the title of the difference container
@app.callback(
    Output('model_name_1', 'children'),
    Input('fcn-resnet50_f1', 'n_clicks'),
    Input('fcn-resnet101_f1', 'n_clicks'),
    Input('deeplabv3-resnet50_f1', 'n_clicks'),
    Input('deeplabv3-resnet101_f1', 'n_clicks'),
    Input('deeplabv3-mobilenetv3-large_f1', 'n_clicks'),
    allow_duplicate = True
)

def show_model_name_left (n_clicks_1, n_clicks_2, n_clicks_3, n_clicks_4, n_clicks_5):

    '''
    Function to display the name of the model that is selected in the filter section on the left side.
    '''

    change_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'fcn-resnet50_f1' in change_id:
        return 'FCN ResNet50'
    elif 'fcn-resnet101_f1' in change_id:
        return 'FCN ResNet101'
    elif 'deeplabv3-resnet50_f1' in change_id:
        return 'DeepLabV3 ResNet50'
    elif 'deeplabv3-resnet101_f1' in change_id:
        return 'DeepLabV3 ResNet101'
    elif 'deeplabv3-mobilenetv3-large_f1' in change_id:
        return 'DeepLabV3 MobileNetV3-Large'
    else:
        return None
    
 # Callback to display the name model, that is chosen in the right filter section, in the title of the difference container   
@app.callback(
    Output('model_name_2', 'children'),
    Input('fcn-resnet50_f2', 'n_clicks'),
    Input('fcn-resnet101_f2', 'n_clicks'),
    Input('deeplabv3-resnet50_f2', 'n_clicks'),
    Input('deeplabv3-resnet101_f2', 'n_clicks'),
    Input('deeplabv3-mobilenetv3-large_f2', 'n_clicks'),
    allow_duplicate = True
)

def show_model_name_left (n_clicks_1, n_clicks_2, n_clicks_3, n_clicks_4, n_clicks_5):
    
    '''
    Function to display the name of the model that is selected in the filter section on the right side.
    '''

    change_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'fcn-resnet50_f2' in change_id:
        return 'FCN ResNet50'
    elif 'fcn-resnet101_f2' in change_id:
        return 'FCN ResNet101'
    elif 'deeplabv3-resnet50_f2' in change_id:
        return 'DeepLabV3 ResNet50'
    elif 'deeplabv3-resnet101_f2' in change_id:
        return 'DeepLabV3 ResNet101'
    elif 'deeplabv3-mobilenetv3-large_f2' in change_id:
        return 'DeepLabV3 MobileNetV3-Large'
    else:
        return None
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
## xAI

# Callback to display the xAI methods results on the left side
@app.callback(
     Output('layer_grad_cam_1', 'children'),
     Input('layer_grad_cam_f1', 'n_clicks'),
     Input('fa_f1', 'n_clicks'),
     Input('saliency_f1', 'n_clicks'),
    Input('lime_f1', 'n_clicks'),
     Input('model_name_1', 'children'),
     Input('label_1', 'children'))

def show_xAI_results_left (n_clicks_1, n_clicks_2, n_clicks_3, n_clicks_4, children_1, children_2):

    '''
    Function to display the xAI methods results on the left side depending on the method, label and model that are selected in the filter section.
    '''

    # Get the id of the dropdown menu that is clicked
    change_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    # Check which model and label is selected in the filter section
    if ('layer_grad_cam_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Bicycle':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.fcn_resnet50(), 2)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Bicycle':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.fcn_resnet50(), 2)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Bicycle':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.fcn_resnet50(), 2)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Bicycle':
        # Load the LIME results
        lime_50 = lime(models.fcn_resnet50(), 2)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    

    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Bus':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.fcn_resnet50(), 6)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Bus':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.fcn_resnet50(), 6)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Bus':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.fcn_resnet50(), 6)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Bus':
        # Load the LIME results
        lime_50 = lime(models.fcn_resnet50(), 6)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
        
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Car':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.fcn_resnet50(), 7)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Car':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.fcn_resnet50(), 7)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Car':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.fcn_resnet50(), 7)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Car':
        # Load the LIME results
        lime_50 = lime(models.fcn_resnet50(), 7)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children

    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Motorbike':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.fcn_resnet50(), 14)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Motorbike':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.fcn_resnet50(), 14)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Motorbike':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.fcn_resnet50(), 14)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Motorbike':
        # Load the LIME results
        lime_50 = lime(models.fcn_resnet50(), 14)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children

    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Person':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.fcn_resnet50(), 15)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Person':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.fcn_resnet50(), 15)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Person':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.fcn_resnet50(), 15)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Person':
        # Load the LIME results
        lime_50 = lime(models.fcn_resnet50(), 15)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children

    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Train':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.fcn_resnet50(), 19)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Train':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.fcn_resnet50(), 19)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Train':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.fcn_resnet50(), 19)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'FCN ResNet50' and children_2 == 'Train':
        # Load the LIME results
        lime_50 = lime(models.fcn_resnet50(), 19)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Bicycle':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.fcn_resnet101(),2)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Bicycle':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.fcn_resnet101(),2)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Bicycle':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.fcn_resnet101(),2)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Bicycle':
        # Load the LIME results
        lime_101 = lime(models.fcn_resnet101(),2)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Bus':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.fcn_resnet101(),6)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Bus':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.fcn_resnet101(),6)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Bus':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.fcn_resnet101(),6)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Bus':
        # Load the LIME results
        lime_101 = lime(models.fcn_resnet101(),6)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Car':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.fcn_resnet101(),7)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Car':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.fcn_resnet101(),7)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Car':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.fcn_resnet101(),7)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Car':
        # Load the LIME results
        lime_101 = lime(models.fcn_resnet101(),7)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Motorbike':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.fcn_resnet101(),14)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Motorbike':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.fcn_resnet101(),14)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Motorbike':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.fcn_resnet101(),14)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Motorbike':
        # Load the LIME results
        lime_101 = lime(models.fcn_resnet101(),14)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Person':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.fcn_resnet101(),15)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Person':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.fcn_resnet101(),15)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Person':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.fcn_resnet101(),15)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Person':
        # Load the LIME results
        lime_101 = lime(models.fcn_resnet101(),15)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Train':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.fcn_resnet101(),19)

        # Callback to display the xAI methods results on the right side
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Train':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.fcn_resnet101(),19)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Train':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.fcn_resnet101(),19)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'FCN ResNet101' and children_2 == 'Train':
        # Load the LIME results
        lime_101 = lime(models.fcn_resnet101(),19)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Bicycle':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.deeplabv3_resnet50(),2)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Bicycle':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.deeplabv3_resnet50(),2)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Bicycle':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.deeplabv3_resnet50(),2)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Bicycle':
        # Load the LIME results
        lime_50 = lime(models.deeplabv3_resnet50(),2)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Bus':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.deeplabv3_resnet50(),6)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Bus':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.deeplabv3_resnet50(),6)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Bus':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.deeplabv3_resnet50(),6)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Bus':
        # Load the LIME results
        lime_50 = lime(models.deeplabv3_resnet50(),6)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Car':
        grad_cam_50 = grad_cam(models.deeplabv3_resnet50(),7)

        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        return children
    
    elif ('fa_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Car':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.deeplabv3_resnet50(),7)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Car':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.deeplabv3_resnet50(),7)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Car':
        # Load the LIME results
        lime_50 = lime(models.deeplabv3_resnet50(),7)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Motorbike':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.deeplabv3_resnet50(),14)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Motorbike':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.deeplabv3_resnet50(),14)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Motorbike':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.deeplabv3_resnet50(),14)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Motorbike':
        # Load the LIME results
        lime_50 = lime(models.deeplabv3_resnet50(),14)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Person':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.deeplabv3_resnet50(),15)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Person':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.deeplabv3_resnet50(),15)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Person':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.deeplabv3_resnet50(),15)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Person':
        # Load the LIME results
        lime_50 = lime(models.deeplabv3_resnet50(),15)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Train':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.deeplabv3_resnet50(),19)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Train':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.deeplabv3_resnet50(),19)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Train':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.deeplabv3_resnet50(),19)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Train':
        # Load the LIME results
        lime_50 = lime(models.deeplabv3_resnet50(),19)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) > 0 and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Bicycle':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.deeplabv3_resnet101(),2)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Bicycle':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.deeplabv3_resnet101(),2)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('salinecy_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Bicycle':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.deeplabv3_resnet101(),2)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif('lime_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Bicycle':
        # Load the LIME results
        lime_101 = lime(models.deeplabv3_resnet101(),2)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Bus':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.deeplabv3_resnet101(),6)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Bus':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.deeplabv3_resnet101(),6)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Bus':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.deeplabv3_resnet101(),6)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Bus':
        # Load the LIME results
        lime_101 = lime(models.deeplabv3_resnet101(),6)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Car':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.deeplabv3_resnet101(),7)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Car':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.deeplabv3_resnet101(),7)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Car':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.deeplabv3_resnet101(),7)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Car':
        # Load the LIME results
        lime_101 = lime(models.deeplabv3_resnet101(),7)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Motorbike':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.deeplabv3_resnet101(),14)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Motorbike':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.deeplabv3_resnet101(),14)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Motorbike':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.deeplabv3_resnet101(),14)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Motorbike':
        # Load the LIME results
        lime_101 = lime(models.deeplabv3_resnet101(),14)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Person':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.deeplabv3_resnet101(),15)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Person':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.deeplabv3_resnet101(),15)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Person':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.deeplabv3_resnet101(),15)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Person':
        # Load the LIME results
        lime_101 = lime(models.deeplabv3_resnet101(),15)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Train':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.deeplabv3_resnet101(),19)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Train':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.deeplabv3_resnet101(),19)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Train':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.deeplabv3_resnet101(),19)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Train':
        # Load the LIME results
        lime_101 = lime(models.deeplabv3_resnet101(),19)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Bicycle':
        # Load the Grad-CAM results
        grad_cam_mobilenet = grad_cam(models.deeplabv3_mobilenetv3_large(),2)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_mobilenet, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Bicycle':
        # Load the Feature Ablation results
        fa_mobilenet = feature_ablation(models.deeplabv3_mobilenetv3_large(),2)

        # Display the Feature Ablation results
        children = html.Img(src=fa_mobilenet, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Bicycle':
        # Load the Saliency Maps results
        saliency_mobilenet = saliency_maps(models.deeplabv3_mobilenetv3_large(),2)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_mobilenet, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Bicycle':
        # Load the LIME results
        lime_mobilenet = lime(models.deeplabv3_mobilenetv3_large(),2)

        # Display the LIME results
        children = html.Img(src=lime_mobilenet, alt='Grad-CAM Image')

        # Return the LIME results
        return children

    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Bus':
        # Load the Grad-CAM results
        grad_cam_mobilenet = grad_cam(models.deeplabv3_mobilenetv3_large(),6)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_mobilenet, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Bus':
        # Load the Feature Ablation results
        fa_mobilenet = feature_ablation(models.deeplabv3_mobilenetv3_large(),6)

        # Display the Feature Ablation results
        children = html.Img(src=fa_mobilenet, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Bus':
        # Load the Saliency Maps results
        saliency_mobilenet = saliency_maps(models.deeplabv3_mobilenetv3_large(),6)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_mobilenet, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Bus':
        # Load the LIME results
        lime_mobilenet = lime(models.deeplabv3_mobilenetv3_large(),6)

        # Display the LIME results
        children = html.Img(src=lime_mobilenet, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Car':
        # Load the Grad-CAM results
        grad_cam_mobilenet = grad_cam(models.deeplabv3_mobilenetv3_large(),7)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_mobilenet, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Car':
        # Load the Feature Ablation results
        fa_mobilenet = feature_ablation(models.deeplabv3_mobilenetv3_large(),7)

        # Display the Feature Ablation results
        children = html.Img(src=fa_mobilenet, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Car':
        # Load the Saliency Maps results
        saliency_mobilenet = saliency_maps(models.deeplabv3_mobilenetv3_large(),7)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_mobilenet, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Car':
        # Load the LIME results
        lime_mobilenet = lime(models.deeplabv3_mobilenetv3_large(),7)

        # Display the LIME results
        children = html.Img(src=lime_mobilenet, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Motorbike':
        # Load the Grad-CAM results
        grad_cam_mobilenet = grad_cam(models.deeplabv3_mobilenetv3_large(),14)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_mobilenet, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Motorbike':
        # Load the Feature Ablation results
        fa_mobilenet = feature_ablation(models.deeplabv3_mobilenetv3_large(),14)

        # Display the Feature Ablation results
        children = html.Img(src=fa_mobilenet, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Motorbike':
        # Load the Saliency Maps results
        saliency_mobilenet = saliency_maps(models.deeplabv3_mobilenetv3_large(),14)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_mobilenet, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Motorbike':
        # Load the LIME results
        lime_mobilenet = lime(models.deeplabv3_mobilenetv3_large(),14)

        # Display the LIME results
        children = html.Img(src=lime_mobilenet, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Person':
        # Load the Grad-CAM results
        grad_cam_mobilenet = grad_cam(models.deeplabv3_mobilenetv3_large(),15)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_mobilenet, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Person':
        # Load the Feature Ablation results
        fa_mobilenet = feature_ablation(models.deeplabv3_mobilenetv3_large(),15)

        # Display the Feature Ablation results
        children = html.Img(src=fa_mobilenet, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Person':
        # Load the Saliency Maps results
        saliency_mobilenet = saliency_maps(models.deeplabv3_mobilenetv3_large(),15)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_mobilenet, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Person':
        # Load the LIME results
        lime_mobilenet = lime(models.deeplabv3_mobilenetv3_large(),15)

        # Display the LIME results
        children = html.Img(src=lime_mobilenet, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Train':
        # Load the Grad-CAM results
        grad_cam_mobilenet = grad_cam(models.deeplabv3_mobilenetv3_large(),19)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_mobilenet, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Train':
        # Load the Feature Ablation results
        fa_mobilenet = feature_ablation(models.deeplabv3_mobilenetv3_large(),19)

        # Display the Feature Ablation results
        children = html.Img(src=fa_mobilenet, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Train':
        # Load the Saliency Maps results
        saliency_mobilenet = saliency_maps(models.deeplabv3_mobilenetv3_large(),19)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_mobilenet, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f1' in change_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Train':
        # Load the LIME results
        lime_mobilenet = lime(models.deeplabv3_mobilenetv3_large(),19)

        # Display the LIME results
        children = html.Img(src=lime_mobilenet, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
# Callback to display the xAI results for the right image
@app.callback(
     Output('layer_grad_cam_2', 'children'),
     Input('layer_grad_cam_f2', 'n_clicks'),
     Input('fa_f2', 'n_clicks'),
     Input('saliency_f2', 'n_clicks'),
     Input('lime_f2', 'n_clicks'),
     Input('model_name_2', 'children'),
     Input('label_2', 'children'))

def show_xAI_results_right (n_clicks_1, n_clicks_2, n_clicks_3, n_clicks_4, children_1, children_2):
    """
    This function displays the xAI results for the right image based on the selected model and label.
    """
    
    # Get the ID of the clicked button
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    # Check which model and label is selected in the filter section
    if ('layer_grad_cam_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Bicycle':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.fcn_resnet50(),2)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Bicycle':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.fcn_resnet50(), 2)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Bicycle':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.fcn_resnet50(), 2)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Bicycle':
        # Load the LIME results
        lime_50 = lime(models.fcn_resnet50(), 2)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Bus':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.fcn_resnet50(),6)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Bus':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.fcn_resnet50(),6)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Bus':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.fcn_resnet50(),6)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Bus':
        # Load the LIME results
        lime_50 = lime(models.fcn_resnet50(),6)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Car':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.fcn_resnet50(),7)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Car':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.fcn_resnet50(),7)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Car':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.fcn_resnet50(),7)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Car':
        # Load the LIME results
        lime_50 = lime(models.fcn_resnet50(),7)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Motorbike':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.fcn_resnet50(),14)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Motorbike':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.fcn_resnet50(),14)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Motorbike':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.fcn_resnet50(),14)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Motorbike':
        # Load the LIME results
        lime_50 = lime(models.fcn_resnet50(),14)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Person':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.fcn_resnet50(),15)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Person':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.fcn_resnet50(),15)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Person':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.fcn_resnet50(),15)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Person':
        # Load the LIME results
        lime_50 = lime(models.fcn_resnet50(),15)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Train':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.fcn_resnet50(),19)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Train':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.fcn_resnet50(),19)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Train':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.fcn_resnet50(),19)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'FCN ResNet50' and children_2 == 'Train':
        # Load the LIME results
        lime_50 = lime(models.fcn_resnet50(),19)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Bicycle':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.fcn_resnet101(),2)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Bicycle':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.fcn_resnet101(),2)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Bicycle':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.fcn_resnet101(),2)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Bicycle':
        # Load the LIME results
        lime_101 = lime(models.fcn_resnet101(),2)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Bus':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.fcn_resnet101(),6)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Bus':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.fcn_resnet101(),6)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Bus':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.fcn_resnet101(),6)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Bus':
        # Load the LIME results
        lime_101 = lime(models.fcn_resnet101(),6)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Car':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.fcn_resnet101(),7)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Car':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.fcn_resnet101(),7)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Car':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.fcn_resnet101(),7)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Car':
        # Load the LIME results
        lime_101 = lime(models.fcn_resnet101(),7)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Motorbike':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.fcn_resnet101(),14)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Motorbike':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.fcn_resnet101(),14)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Motorbike':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.fcn_resnet101(),14)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Motorbike':
        # Load the LIME results
        lime_101 = lime(models.fcn_resnet101(),14)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Person':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.fcn_resnet101(),15)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Person':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.fcn_resnet101(),15)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Person':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.fcn_resnet101(),15)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Person':
        # Load the LIME results
        lime_101 = lime(models.fcn_resnet101(),15)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Train':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.fcn_resnet101(),19)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Train':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.fcn_resnet101(),19)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Train':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.fcn_resnet101(),19)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'FCN ResNet101' and children_2 == 'Train':
        # Load the LIME results
        lime_101 = lime(models.fcn_resnet101(),19)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Bicycle':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.deeplabv3_resnet50(),2)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Bicycle':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.deeplabv3_resnet50(),2)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Bicycle':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.deeplabv3_resnet50(),2)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Bicycle':
        # Load the LIME results
        lime_50 = lime(models.deeplabv3_resnet50(),2)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Bus':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.deeplabv3_resnet50(),6)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Bus':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.deeplabv3_resnet50(),6)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Bus':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.deeplabv3_resnet50(),6)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Bus':
        # Load the LIME results
        lime_50 = lime(models.deeplabv3_resnet50(),6)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Car':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.deeplabv3_resnet50(),7)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Car':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.deeplabv3_resnet50(),7)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Car':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.deeplabv3_resnet50(),7)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Car':
        # Load the LIME results
        lime_50 = lime(models.deeplabv3_resnet50(),7)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Motorbike':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.deeplabv3_resnet50(),14)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Motorbike':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.deeplabv3_resnet50(),14)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Motorbike':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.deeplabv3_resnet50(),14)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Motorbike':
        # Load the LIME results
        lime_50 = lime(models.deeplabv3_resnet50(),14)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Person':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.deeplabv3_resnet50(),15)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Person':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.deeplabv3_resnet50(),15)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Person':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.deeplabv3_resnet50(),15)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Person':
        # Load the LIME results
        lime_50 = lime(models.deeplabv3_resnet50(),15)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Train':
        # Load the Grad-CAM results
        grad_cam_50 = grad_cam(models.deeplabv3_resnet50(),19)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_50, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Train':
        # Load the Feature Ablation results
        fa_50 = feature_ablation(models.deeplabv3_resnet50(),19)

        # Display the Feature Ablation results
        children = html.Img(src=fa_50, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Train':
        # Load the Saliency Maps results
        saliency_50 = saliency_maps(models.deeplabv3_resnet50(),19)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_50, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet50' and children_2 == 'Train':
        # Load the LIME results
        lime_50 = lime(models.deeplabv3_resnet50(),19)

        # Display the LIME results
        children = html.Img(src=lime_50, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Bicycle':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.deeplabv3_resnet101(),2)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Bicycle':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.deeplabv3_resnet101(),2)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Bicycle':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.deeplabv3_resnet101(),2)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Bicycle':
        # Load the LIME results
        lime_101 = lime(models.deeplabv3_resnet101(),2)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Bus':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.deeplabv3_resnet101(),6)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Bus':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.deeplabv3_resnet101(),6)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Bus':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.deeplabv3_resnet101(),6)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Bus':
        # Load the LIME results
        lime_101 = lime(models.deeplabv3_resnet101(),6)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Car':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.deeplabv3_resnet101(),7)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Car':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.deeplabv3_resnet101(),7)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Car':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.deeplabv3_resnet101(),7)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Car':
        # Load the LIME results
        lime_101 = lime(models.deeplabv3_resnet101(),7)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Motorbike':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.deeplabv3_resnet101(),14)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Motorbike':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.deeplabv3_resnet101(),14)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Motorbike':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.deeplabv3_resnet101(),14)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Motorbike':
        # Load the LIME results
        lime_101 = lime(models.deeplabv3_resnet101(),14)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Person':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.deeplabv3_resnet101(),15)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Person':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.deeplabv3_resnet101(),15)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Person':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.deeplabv3_resnet101(),15)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Person':
        # Load the LIME results
        lime_101 = lime(models.deeplabv3_resnet101(),15)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Train':
        # Load the Grad-CAM results
        grad_cam_101 = grad_cam(models.deeplabv3_resnet101(),19)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_101, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Train':
        # Load the Feature Ablation results
        fa_101 = feature_ablation(models.deeplabv3_resnet101(),19)

        # Display the Feature Ablation results
        children = html.Img(src=fa_101, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Train':
        # Load the Saliency Maps results
        saliency_101 = saliency_maps(models.deeplabv3_resnet101(),19)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_101, alt='Grad-CAM Image')

        # Return the Saliency Maps results  
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'DeepLabV3 ResNet101' and children_2 == 'Train':
        # Load the LIME results
        lime_101 = lime(models.deeplabv3_resnet101(),19)

        # Display the LIME results
        children = html.Img(src=lime_101, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Bicycle':
        # Load the Grad-CAM results
        grad_cam_mobilenet = grad_cam(models.deeplabv3_mobilenetv3_large(),2)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_mobilenet, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Bicycle':
        # Load the Feature Ablation results
        fa_mobilenet = feature_ablation(models.deeplabv3_mobilenetv3_large(),2)

        # Display the Feature Ablation results
        children = html.Img(src=fa_mobilenet, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Bicycle':
        # Load the Saliency Maps results
        saliency_mobilenet = saliency_maps(models.deeplabv3_mobilenetv3_large(),2)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_mobilenet, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Bicycle':
        # Load the LIME results
        lime_mobilenet = lime(models.deeplabv3_mobilenetv3_large(),2)

        # Display the LIME results
        children = html.Img(src=lime_mobilenet, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Bus':
        # Load the Grad-CAM results
        grad_cam_mobilenet = grad_cam(models.deeplabv3_mobilenetv3_large(),6)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_mobilenet, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Bus':
        # Load the Feature Ablation results
        fa_mobilenet = feature_ablation(models.deeplabv3_mobilenetv3_large(),6)

        # Display the Feature Ablation results
        children = html.Img(src=fa_mobilenet, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Bus':
        # Load the Saliency Maps results
        saliency_mobilenet = saliency_maps(models.deeplabv3_mobilenetv3_large(),6)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_mobilenet, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Bus':
        # Load the LIME results
        lime_mobilenet = lime(models.deeplabv3_mobilenetv3_large(),6)

        # Display the LIME results
        children = html.Img(src=lime_mobilenet, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Car':
        # Load the Grad-CAM results
        grad_cam_mobilenet = grad_cam(models.deeplabv3_mobilenetv3_large(),7)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_mobilenet, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Car':
        # Load the Feature Ablation results
        fa_mobilenet = feature_ablation(models.deeplabv3_mobilenetv3_large(),7)

        # Display the Feature Ablation results
        children = html.Img(src=fa_mobilenet, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Car':
        # Load the Saliency Maps results
        saliency_mobilenet = saliency_maps(models.deeplabv3_mobilenetv3_large(),7)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_mobilenet, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Car':
        # Load the LIME results
        lime_mobilenet = lime(models.deeplabv3_mobilenetv3_large(),7)

        # Display the LIME results
        children = html.Img(src=lime_mobilenet, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Motorbike':
        # Load the Grad-CAM results
        grad_cam_mobilenet = grad_cam(models.deeplabv3_mobilenetv3_large(),14)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_mobilenet, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Motorbike':
        # Load the Feature Ablation results
        fa_mobilenet = feature_ablation(models.deeplabv3_mobilenetv3_large(),14)

        # Display the Feature Ablation results
        children = html.Img(src=fa_mobilenet, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Motorbike':
        # Load the Saliency Maps results
        saliency_mobilenet = saliency_maps(models.deeplabv3_mobilenetv3_large(),14)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_mobilenet, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Motorbike':
        # Load the LIME results
        lime_mobilenet = lime(models.deeplabv3_mobilenetv3_large(),14)

        # Display the LIME results
        children = html.Img(src=lime_mobilenet, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Person':
        # Load the Grad-CAM results
        grad_cam_mobilenet = grad_cam(models.deeplabv3_mobilenetv3_large(),15)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_mobilenet, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Person':
        # Load the Feature Ablation results
        fa_mobilenet = feature_ablation(models.deeplabv3_mobilenetv3_large(),15)

        # Display the Feature Ablation results
        children = html.Img(src=fa_mobilenet, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Person':
        # Load the Saliency Maps results
        saliency_mobilenet = saliency_maps(models.deeplabv3_mobilenetv3_large(),15)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_mobilenet, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Person':
        # Load the LIME results
        lime_mobilenet = lime(models.deeplabv3_mobilenetv3_large(),15)

        # Display the LIME results
        children = html.Img(src=lime_mobilenet, alt='Grad-CAM Image')

        # Return the LIME results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('layer_grad_cam_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Train':
        # Load the Grad-CAM results
        grad_cam_mobilenet = grad_cam(models.deeplabv3_mobilenetv3_large(),19)

        # Display the Grad-CAM results
        children = html.Img(src=grad_cam_mobilenet, alt='Grad-CAM Image')

        # Return the Grad-CAM results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('fa_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Train':
        # Load the Feature Ablation results
        fa_mobilenet = feature_ablation(models.deeplabv3_mobilenetv3_large(),19)

        # Display the Feature Ablation results
        children = html.Img(src=fa_mobilenet, alt='Grad-CAM Image')

        # Return the Feature Ablation results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('saliency_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Train':
        # Load the Saliency Maps results
        saliency_mobilenet = saliency_maps(models.deeplabv3_mobilenetv3_large(),19)

        # Display the Saliency Maps results
        children = html.Img(src=saliency_mobilenet, alt='Grad-CAM Image')

        # Return the Saliency Maps results
        return children
    
    # Check which model and label is selected in the filter section
    elif ('lime_f2' in changed_id) and children_1 == 'DeepLabV3 MobileNetV3-Large' and children_2 == 'Train':
        # Load the LIME results
        lime_mobilenet = lime(models.deeplabv3_mobilenetv3_large(),19)

        # Display the LIME results
        children = html.Img(src=lime_mobilenet, alt='Grad-CAM Image')

        # Return the LIME results
        return children

    


#------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Show Difference
    

# Calculate the difference between two segmentation images
@app.callback(
    Output('difference', 'children'),
    [Input('output-segmentation-1', 'children'),
     Input('output-segmentation-2', 'children')]
)
def calculate_segmentation_difference(children_1, children_2):
    
    '''
    This function calculates the difference between two segmentation images.
    '''

    # Check if both images are available
    if children_1 is not None and children_2 is not None:
        # Load the base64-encoded image data
        img1_data = children_1['props']['src'].split(',')[1]
        img2_data = children_2['props']['src'].split(',')[1]
        
        # Convert the base64-encoded image data to a numpy array
        img1_array = np.array(Image.open(io.BytesIO(base64.b64decode(img1_data))))
        img2_array = np.array(Image.open(io.BytesIO(base64.b64decode(img2_data))))

        # Calculate the absolute difference between the two images
        difference1 = np.abs(img1_array - img2_array)
        difference2 = np.abs(img2_array - img1_array)

        # Calculate the difference between the two differences
        difference = difference1 - difference2
        
        # Convert the difference to an image
        diff_image = Image.fromarray(difference.astype('uint8'))
        
        # Convert the difference image to a base64-encoded string
        buffered = BytesIO()
        diff_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Create an HTML element to display the difference image
        diff_img_html = html.Img(src='data:image/png;base64,' + img_str)
        
        # Return the difference image
        return diff_img_html
    
    # Check if no images are available
    elif children_1 is None and children_2 is None:
        # Return a message that no images are available
        return "No segmentation images available to compare."
    
    # Check if only one image is available
    elif children_1 is None and children_2 is not None:
        # Return a message that only one image is available
        return "No segmentation image available in the left filter."
    # Check if only one image is available
    elif children_1 is not None and children_2 is None:
        # Return a message that only one image is available
        return "No segmentation image available in the right filter."

# Show the model names in the difference image container  
@app.callback(
     Output('difference_model_names', 'children'),
        [Input('model_name_1', 'children'),
        Input('model_name_2', 'children')])

def show_model_names(model_name_1, model_name_2):

    '''
    This function shows the model names in the difference image container.
    '''

    if model_name_1 is not None and model_name_2 is None:
        return f'Model difference between {model_name_1} and Model right'
    elif model_name_1 is None and model_name_2 is not None:
        return f'Model difference between Model left and {model_name_2}'
    elif model_name_1 is not None and model_name_2 is not None:
        return f'Model difference between {model_name_1} and {model_name_2}'
    elif model_name_1 is None and model_name_2 is None:
        return 'Model difference between Model left and Model right'


#------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Label Selection


# Show the label names in the difference image container
@app.callback(
    Output('label_1', 'children'),
    Output('label_2', 'children'),
    Input('bicycle_f1', 'n_clicks'),
    Input('bus_f1', 'n_clicks'),
    Input('car_f1', 'n_clicks'),
    Input('motorbike_f1', 'n_clicks'),
    Input('person_f1', 'n_clicks'),
    Input('train_f1', 'n_clicks'),
    Input('bicycle_f2', 'n_clicks'),
    Input('bus_f2', 'n_clicks'),
    Input('car_f2', 'n_clicks'),
    Input('motorbike_f2', 'n_clicks'),
    Input('person_f2', 'n_clicks'),
    Input('train_f2', 'n_clicks'),
    allow_duplicate = True
)

def show_labelname (n_clicks_1, n_clicks_2, n_clicks_3, n_clicks_4, n_clicks_5, n_clicks_6, n_clicks_7, n_clicks_8, n_clicks_9, n_clicks_10, n_clicks_11, n_clicks_12):

    """
    This function shows the label names in the difference image container.
    """

    change_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'bicycle_f1' in change_id:
        return 'Bicycle', 'Bycicle'
    elif 'bus_f1' in change_id:
        return 'Bus', 'Bus'
    elif 'car_f1' in change_id:
        return 'Car', 'Car'
    elif 'motorbike_f1' in change_id:
        return 'Motorbike'
    elif 'person_f1' in change_id:
        return 'Person', 'Person'
    elif 'train_f1' in change_id:
        return 'Train', 'Train'
    elif 'bicycle_f2' in change_id:
        return 'Bicycle', 'Bycicle'
    elif 'bus_f2' in change_id:
        return 'Bus',  'Bus'
    elif 'car_f2' in change_id:
        return 'Car', 'Car'
    elif 'motorbike_f2' in change_id:
        return 'Motorbike', 'Motorbike'
    elif 'person_f2' in change_id:
        return 'Person', 'Person'
    elif 'train_f2' in change_id:
        return 'Train', 'Train'
    else:
        return None, None
    


#------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Difference of x-AI-methods
    
# Calculate the difference between two xAI-methods
@app.callback(
    Output('difference_xAI', 'children'),
    [Input('layer_grad_cam_1', 'children'),
     Input('layer_grad_cam_2', 'children')]
)

def calculate_xAI_difference(children_1, children_2):
    
    '''
    This function calculates the difference between two xAI-methods.
    '''


    # Check if both images are available
    if children_1 is not None and children_2 is not None:
        
        # Load the base64-encoded image data
        img1_data = children_1['props']['src'].split(',')[1]
        img2_data = children_2['props']['src'].split(',')[1]
        
        # Convert the base64-encoded image data to a numpy array
        img1_array = np.array(Image.open(io.BytesIO(base64.b64decode(img1_data))))
        img2_array = np.array(Image.open(io.BytesIO(base64.b64decode(img2_data))))

        # Calculate the absolute difference between the two images
        difference1 = np.abs(img1_array - img2_array)
        difference2 = np.abs(img2_array - img1_array)

        difference = difference1 - difference2
        
        # Convert the difference to an image
        diff_image = Image.fromarray(difference.astype('uint8'))
        
        # Convert the difference image to a base64-encoded string
        buffered = BytesIO()
        diff_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Create an HTML element to display the difference image
        diff_img_html = html.Img(src='data:image/png;base64,' + img_str)
        
        # Return the difference image
        return diff_img_html
    
    # Check if no images are available
    elif children_1 is None and children_2 is None:
        # Return a message that no images are available
        return "No xAI-method is chosen in the left and right filter."
    
    # Check if only one image is available
    elif children_1 is None and children_2 is not None:
        # Return a message that only one image is available
        return "No xAI-method is chosen in the left filter."
    
    # Check if only one image is available
    elif children_1 is not None and children_2 is None:
        # Return a message that only one image is available
        return "No xAI-method is chosen in the right filter."
    
    

# Show the xAI-method names in the difference image container
@app.callback(
    Output('method_left', 'children'),
    Input('layer_grad_cam_f1', 'n_clicks'),
    Input('fa_f1', 'n_clicks'),
    Input('saliency_f1', 'n_clicks'),
    Input('lime_f1', 'n_clicks'),
    allow_duplicate = True
)

def show_method_name_left (n_clicks_1, n_clicks_2, n_clicks_3, n_clicks_4):

    """
    This function shows the xAI-method names in the difference image container.
    """

    change_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'layer_grad_cam_f1' in change_id:
        return 'Layer Grad-CAM'
    
    elif 'fa_f1' in change_id:
        return 'Feature Ablation'
    elif 'saliency_f1' in change_id:
        return 'Saliency Maps'
    elif 'lime_f1' in change_id:
        return 'LIME'

    else:
        return None
    

# Show the xAI-method names in the difference image container
@app.callback(
    Output('method_right', 'children'),
    Input('layer_grad_cam_f2', 'n_clicks'),
    Input('fa_f2', 'n_clicks'),
    Input('saliency_f2', 'n_clicks'),
    Input('lime_f2', 'n_clicks'),
    allow_duplicate = True
)

def show_method_name_right (n_clicks_1, n_clicks_2, n_clicks_3, n_clicks_4):

    """
    This function shows the xAI-method names in the difference image container.
    """

    change_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'layer_grad_cam_f2' in change_id:
        return 'Layer Grad-CAM'
    elif 'fa_f2' in change_id:
        return 'Feature Ablation'
    elif 'saliency_f2' in change_id:
        return 'Saliency Maps'
    elif 'lime_f2' in change_id:
        return 'LIME'
    else:
        return None


# Show the xAI-method names in the difference image container
@app.callback(
     Output('difference_method_names', 'children'),
        [Input('method_left', 'children'),
        Input('method_right', 'children')])

def show_method_names(method_1, method_2):
    if method_1 is not None and method_2 is not None:
        return f'Method difference between {method_1} and {method_2}'
    elif method_1 is None and method_2 is not None:
        return f'Method difference between Method left and {method_2}'
    elif method_1 is not None and method_2 is None:
        return f'Method difference between {method_1} and Method right'
    elif method_1 is None and method_2 is None:
        return 'Method difference between Method left and Method right'
    

# Include CSS file
app.css.append_css({
    'external_url': 'assets/styles.css'
})

if __name__ == '__main__':
    app.run_server(debug=True, port=1718)