import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import items
from models import SegNet

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


performance_model_1 = '0.95'
performance_model_2 = '0.85'

model_1 = 'FCN ResNet50'
model_2 = 'DeepLabV3 ResNet101'


app.layout = html.Div(children=[
        html.Div(children=[
            html.P('File'),
            dbc.DropdownMenu(label='Show Demo',
                            children = items.items_models_top_bar(),
                            direction='down',
                            toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                            style={'margin': '5px'}
            ),
            dbc.DropdownMenu(label='Add Window',
                            children = items.items_windows(),
                            direction='down',
                            toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                            style={'margin': '5px'}
            ),
            dbc.DropdownMenu(label='Choose Model',
                            children=[items.items_models_top_bar()],
                            direction='down',
                            toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                            style={'margin': '5px'}
            ),
            dbc.DropdownMenu(label='Import Image', children=[
                            dbc.DropdownMenuItem(dcc.Upload(html.P('Import Image'), accept='.jpg, .png, .tiff', id='import_image_1'))],
                            direction='down',
                            toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                            style={'margin': '5px'})
                
        ], id='top-bar'),

        html.Div([
            html.Div(id='card_1', children=[
                html.P('Choose Model'),
                html.Img(src='assets/images/neural_network.png', alt='Model Pictogram')
            ]),
            html.Div(id='card_2', children=[
                html.P('Import Image'),
                html.Img(src='assets/images/image.png', alt='Image Pictogram', n_clicks=0)      
            ])
        ], id='card_container'),

        html.Div([
            html.Div(id='dropdown_container_1', children=[
                dbc.DropdownMenu( label='Choose Model',
                children=items.items_models_card(),
                direction='down',
                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0.5px solid black'}
                )]
            ),
            html.Div(id='dropdown_container_2', children=[
                dbc.DropdownMenu(label='Import Image', children=[
                            dbc.DropdownMenuItem(dcc.Upload(html.P('Import Image'), accept='.jpg, .png, .tiff', id='import_image_2'))],
                            direction='down',
                            toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'})
                    ]),
            
                ], id='row_container'),
            


            html.Div([ 
                html.Div(id='result_1_div', children=[
                    html.Div(id='filter_container_1', children=[
                        dbc.DropdownMenu(label='Model selection',
                                size='sm',
                                children=items.items_models_filter_section(),
                                direction='down',
                                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                                style={'margin': '5px'}),
                        dbc.DropdownMenu(label='Method selection',
                                size='sm',
                                children=items.items_method(),
                                direction='down',
                                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                                style={'margin': '5px'}),
                        dbc.DropdownMenu(label='Model selection',
                                size='sm',
                                children=items.items_labels(),
                                direction='down',
                                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                                style={'margin': '5px'})           
                    ]),
                    html.Div(id='image-upload-container_1', children=[
                        html.Div(id='output-image-upload_1')]
                        ),
                ]),
                html.Div(id='result_2_div', children=[
                    html.Div(id='filter_container_2', children=[
                        dbc.DropdownMenu(label='Model selection',
                                size='sm',
                                children=items.items_models_filter_section(),
                                direction='down',
                                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                                style={'margin': '5px'}),
                        dbc.DropdownMenu(label='Method selection',
                                size='sm',
                                children=items.items_method(),
                                direction='down',
                                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                                style={'margin': '5px'}),
                        dbc.DropdownMenu(label='Model selection',
                                size='sm',
                                children=items.items_labels(),
                                direction='down',
                                toggle_style={'color': 'black', 'background-color': 'grey', 'border': '0px solid black'},
                                style={'margin': '5px'})
                        ]),
                    html.Div(id='image-upload-container_2', children=[
                        html.Div(id='output-image-upload_2')]
                        ),     
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




def parse_contents(contents, filename, date,):
    print("Contents:", contents)  # Debugging-Ausdruck für den Inhalt des Bildes
    if contents is not None:
        # HTML-Element für das Bild erstellen
        image_element = html.Img(src=contents)
        # Einzelnes Div-Element mit allen Inhalten erstellen
        return html.Div([
            image_element
        ])

@app.callback(
    [Output('output-image-upload_1', 'children'),
    Output('output-image-upload_2', 'children')],
    [Input('import_image_1', 'contents'),
    Input('import_image_1', 'filename'),
    Input('import_image_1', 'last_modified'),
    Input('import_image_2', 'contents'),
    Input('import_image_2', 'filename'),
    Input('import_image_2', 'last_modified')]

)
def update_output_1(contents_1, filename_1, last_modified_1, contents_2, filename_2, last_modified_2):
    # Überprüfen, ob contents eine Liste ist, wenn nicht, eine Liste daraus machen
    if not isinstance(contents_1, list) or isinstance(contents_2, list):
        if contents_1 is not None:
            contents = [contents_1]
        elif contents_2 is not None:
            contents = [contents_2]
    
    if not isinstance(filename_1, list) or isinstance(filename_2, list):
        if filename_1 is not None:
            filename = [filename_1]
        elif filename_2 is not None:
            filename = [filename_2]
    
    if not isinstance(last_modified_1, list) or isinstance(last_modified_2, list):
        if last_modified_1 is not None:
            last_modified = [last_modified_1]
        elif last_modified_2 is not None:
            last_modified = [last_modified_2]
        
    # Hier kannst du nun sicher sein, dass alle Eingabeparameter Listen sind
    if contents:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(contents, filename, last_modified)]
        return children, children
    

@app.callback(
    Output('result_1_div', 'style'),
    Output('result_2_div', 'style'),
    Output('difference_result', 'style'),
    Output('performance_div', 'style'),
    Output('card_container', 'style'),
    Output('row_container', 'style'),
    Input('result', 'n_clicks'),
    Input('import_image_1', 'contents'),
    Input('import_image_2', 'contents')

)

def show_result_window_div(n_clicks, contents_1, contents_2):
    if (n_clicks and n_clicks > 0) or contents_1 is not None or contents_2 is not None:  # Überprüfen, ob das Element geklickt wurde
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


# Include CSS file
app.css.append_css({
    'external_url': 'assets/styles.css'
})

if __name__ == '__main__':
    app.run_server(debug=True, port=1718)