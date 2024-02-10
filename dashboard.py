from dash import Dash, html, dcc, callback_context
from dash.dependencies import Input, Output

app = Dash(__name__)

app.layout = html.Div([
    html.Div(
        children=[
            html.P('File'),
            html.P('Show Demo'),
            html.P('Add Window'),
            html.P('Choose Model'),
            html.P('Import Image', id='import-image'),
            html.Div(id='upload-container')
        ],
        id='top-bar'
    ),

    html.Div([
        html.Div(className='card', children=[
            html.P('Choose Model'),
            html.Img(src='assets/images/neural_network.png', alt='Model Pictogram')
        ]),
        html.Div(className='card', children=[
            html.P('Import Image'),
            html.Img(src='assets/images/image.png', alt='Image Pictogram')
            
        ],
        id='import-image'
        ),
        
    ], className='row-container')
])

app.css.append_css({
    'external_url': 'assets/styles.css'
})


@app.callback(
    Output('upload-container', 'children'),
    [Input('import-image', 'n_clicks')]
)
def toggle_upload(n_clicks):
    if n_clicks is None:
        return []
    else:
        return dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        )

@app.callback(
    Output('upload-container', 'children'),
    [Input('top-bar', 'n_clicks')],
    prevent_initial_call=True
)
def close_upload(n_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return []
    elif ctx.triggered[0]['prop_id'] != 'top-bar.n_clicks':
        return []
    else:
        return []

if __name__ == '__main__':
    app.run_server(debug=True)
