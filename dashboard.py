import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

items_models_top_bar = [
    dbc.DropdownMenuItem('FCN ResNet50', id='fcn-resnet50_t'),
    dbc.DropdownMenuItem('FCN ResNet101', id='fcn-resnet101_t'),
    dbc.DropdownMenuItem('DeepLabV3 ResNet50', id='deeplabv3-resnet50_t'),
    dbc.DropdownMenuItem('DeepLabV3 ResNet101', id='deeplabv3-resnet101_t'),
    dbc.DropdownMenuItem('DeepLabV3 MobileNetV3-Large', id='deeplabv3-mobilenetv3-large_t'),
    dbc.DropdownMenuItem('LR-ASPP MobileNetV3-Large', id='lr-aspp-mobilenetv3-large_t')
    ]

items_models_card = [
    dbc.DropdownMenuItem('FCN ResNet50', id='fcn-resnet50_c'),
    dbc.DropdownMenuItem('FCN ResNet101', id='fcn-resnet101_c'),
    dbc.DropdownMenuItem('DeepLabV3 ResNet50', id='deeplabv3-resnet50_c'),
    dbc.DropdownMenuItem('DeepLabV3 ResNet101', id='deeplabv3-resnet101_c'),
    dbc.DropdownMenuItem('DeepLabV3 MobileNetV3-Large', id='deeplabv3-mobilenetv3-large_c'),
    dbc.DropdownMenuItem('LR-ASPP MobileNetV3-Large', id='lr-aspp-mobilenetv3-large_c')
    ]

items_models_filter_section = [
    dbc.DropdownMenuItem('FCN ResNet50', id='fcn-resnet50_f'),
    dbc.DropdownMenuItem('FCN ResNet101', id='fcn-resnet101_f'),
    dbc.DropdownMenuItem('DeepLabV3 ResNet50', id='deeplabv3-resnet50_f'),
    dbc.DropdownMenuItem('DeepLabV3 ResNet101', id='deeplabv3-resnet101_f'),
    dbc.DropdownMenuItem('DeepLabV3 MobileNetV3-Large', id='deeplabv3-mobilenetv3-large_f'),
    dbc.DropdownMenuItem('LR-ASPP MobileNetV3-Large', id='lr-aspp-mobilenetv3-large_f')
    ]

items_windows = [
    dbc.DropdownMenuItem('Result', id='result'),
    dbc.DropdownMenuItem('Difference', id='difference')
    ]

items_method = [
    dbc.DropdownMenuItem('Method 1', id='method_1'),
    dbc.DropdownMenuItem('Method 2', id='method_2'),
    dbc.DropdownMenuItem('Method 3', id='method_3'),
    dbc.DropdownMenuItem('Method 4', id='method_4'),
    dbc.DropdownMenuItem('Method 5', id='method_5'),
    dbc.DropdownMenuItem('Method 6', id='method_6')
    ]

items_labels = [
    dbc.DropdownMenuItem('Label 1', id='label_1'),
    dbc.DropdownMenuItem('Label 2', id='label_2'),
    dbc.DropdownMenuItem('Label 3', id='label_3'),
    dbc.DropdownMenuItem('Label 4', id='label_4'),
    dbc.DropdownMenuItem('Label 5', id='label_5'),
    dbc.DropdownMenuItem('Label 6', id='label_6')
    ]

performance_model_1 = '0.95'
performance_model_2 = '0.85'

model_1 = 'FCN ResNet50'
model_2 = 'DeepLabV3 ResNet101'



app.layout = html.Div([
    html.Div([
        html.Div([
            html.P('File'),
            dbc.DropdownMenu(label='Show Demo',
                            children = items_models_top_bar,
                            direction='down',
                            toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                            style={'margin': '5px'}
            ),
            dbc.DropdownMenu(label='Add Window',
                            children = items_windows,
                            direction='down',
                            toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                            style={'margin': '5px'}
            ),
            dbc.DropdownMenu(label='Choose Model',
                            children=items_models_top_bar,
                            direction='down',
                            toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                            style={'margin': '5px'}
            ),
            dbc.DropdownMenu(label='Import Image', children=[
                            dbc.DropdownMenuItem(dcc.Upload(html.P('Import Image'), accept='.jpg, .png, .tiff'), id='import_image')],
                            direction='down',
                            toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                            style={'margin': '5px'})
                
        ], id='top-bar'),

        html.Div([
            html.Div(id='card', children=[
                html.P('Choose Model'),
                html.Img(src='assets/images/neural_network.png', alt='Model Pictogram')
            ]),
            html.Div(id='card', children=[
                html.P('Import Image'),
                html.Img(src='assets/images/image.png', alt='Image Pictogram', n_clicks=0)      
            ])
        ], id='card_container'),

        html.Div([
            html.Div(id='dropdown_container', children=[
                dbc.DropdownMenu( label='Choose Model',
                children=items_models_card,
                direction='down',
                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0.5px solid black'}
                )]
            ),
            html.Div(id='dropdown_container', children=[
                dbc.DropdownMenu(label='Import Image', children=[
                            dbc.DropdownMenuItem(dcc.Upload(html.P('Import Image'), accept='.jpg, .png, .tiff'), id='import_image')],
                            direction='down',
                            toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'})
                    ])
                ], id='row_container'),
        
            html.Div([
                html.Div(id='result_1_div', children=[
                    html.Div(id='filter_container', children=[
                        dbc.DropdownMenu(label='Model selection',
                                size='sm',
                                children=items_models_filter_section,
                                direction='down',
                                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                                style={'margin': '5px'}),
                        dbc.DropdownMenu(label='Method selection',
                                size='sm',
                                children=items_method,
                                direction='down',
                                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                                style={'margin': '5px'}),
                        dbc.DropdownMenu(label='Model selection',
                                size='sm',
                                children=items_labels,
                                direction='down',
                                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                                style={'margin': '5px'})           
                    ])
                ]),
                html.Div(id='result_2_div', children=[
                    html.Div(id='filter_container', children=[
                        dbc.DropdownMenu(label='Model selection',
                                size='sm',
                                children=items_models_filter_section,
                                direction='down',
                                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                                style={'margin': '5px'}),
                        dbc.DropdownMenu(label='Method selection',
                                size='sm',
                                children=items_method,
                                direction='down',
                                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                                style={'margin': '5px'}),
                        dbc.DropdownMenu(label='Model selection',
                                size='sm',
                                children=items_labels,
                                direction='down',
                                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                                style={'margin': '5px'})

                ])
            ])
                ], id='result_container'),

            html.Div([
                html.Div(id='difference_result', children=[
                        html.H4('Difference')
            ]),

                html.Div(id='performance_div', children=[
                    html.H4('Performance'),
                    html.Div([
                        html.P(f'Performance of the {model_1}: ' + performance_model_1),
                        html.P(f'Performance of the {model_2}: ' + performance_model_2)
                ])
            ])
            ],id='difference_container')
        ])
 ])


@app.callback(
    Output('result_1_div', 'style'),
    Output('result_2_div', 'style'),
    Output('difference_result', 'style'),
    Output('performance_div', 'style'),
    Output('card_container', 'style'),
    Output('row_container', 'style'),
    Input('result', 'n_clicks')
)
def show_result_div(n_clicks):
    if n_clicks and n_clicks > 0:  # Überprüfen, ob das Element geklickt wurde
        result_1_div_style = {'display': 'block'}
        result_2_div_style = {'display': 'block'}
        difference_result_style = {'display': 'block'}
        performance_div_style = {'display': 'block'}
        card_container_style = {'display': 'none'}
        row_container_style = {'display': 'none'}
    else:
        result_1_div_style = {'display': 'none'}
        result_2_div_style = {'display': 'none'}
        difference_result_style = {'display': 'none'}
        performance_div_style = {'display': 'none'}
        card_container_style = {'opacity': 1}
        row_container_style = {'opacity': 1}


    return result_1_div_style, result_2_div_style, difference_result_style, performance_div_style, card_container_style, row_container_style


"""@app.callback(
    Output('dropdown_1', 'style'),
    Output('dropdown_2', 'style'),
    Output('dropdown_3', 'style'),
    [Input('p_element_1', 'n_clicks'),
     Input('p_element_2', 'n_clicks'),
     Input('p_element_3', 'n_clicks')])
    
def open_dropdown(p1_clicks, p2_clicks, p3_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return [{'display': 'none'}] * 3
    else:
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if prop_id == 'p_element_1':
            return [{'display': 'block'}, {'display': 'none'}, {'display': 'none'}]
        elif prop_id == 'p_element_2':
            return [{'display': 'none'}, {'display': 'block'}, {'display': 'none'}]
        elif prop_id == 'p_element_3':
            return [{'display': 'none'}, {'display': 'none'}, {'display': 'block'}]"""

# Include CSS file
app.css.append_css({
    'external_url': 'assets/styles.css'
})

if __name__ == '__main__':
    app.run_server(debug=True)
