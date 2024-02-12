from dash import Dash, html, dcc, callback_context
from dash.dependencies import Input, Output, State

app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Div([
            html.P('File'),
            html.P('Show Demo'),
            html.P('Add Window', id='p_element'),
            dcc.Dropdown(['FCN ResNet50', 'FCN ResNet101', 'DeepLabV3 ResNet50','DeepLabV3 ResNet101', 'DeepLabV3 MobileNetV3-Large', 'LR-ASPP MobileNetV3-Large'
                         ],'FCN ResNet50', id='dropdown'),
            dcc.Upload(html.P('Choose Model'), accept='.h5'),
            dcc.Upload(html.P('Import Image'), accept='.jpg, .png, .tiff')

        ], id='top-bar'),

        html.Div([
            html.Div(className='card', children=[
                html.P('Choose Model'),
                html.Img(src='assets/images/neural_network.png', alt='Model Pictogram'),
            ]),
            html.Div(className='card', children=[
                html.P('Import Image'),
                html.Img(src='assets/images/image.png', alt='Image Pictogram', n_clicks=0),      
            ]),
        ], className='row-container'),

        html.Div([
            dcc.Upload(
                id='upload-model', 
                children=html.Button('Upload Model'), accept='.h5'
            ),
            dcc.Upload(
                id='upload-image',
                children=html.Button('Upload Image'), accept='.jpg, .png, .tiff'
            )
        ], className='row-container'),
    ]),
])

@app.callback(
    Output('dropdown', 'style'),
    [Input('p_element', 'n_clicks')]
)
def open_dropdown(n_clicks):
    if n_clicks is not None and n_clicks > 0:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

# Include CSS file
app.css.append_css({
    'external_url': 'assets/styles.css'
})

if __name__ == '__main__':
    app.run_server(debug=True)
