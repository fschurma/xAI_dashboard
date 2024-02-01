from dash import Dash, html


app = Dash(__name__,)

app.layout = html.Div([
    html.Div(
        children=[
                html.P('File'),
                html.P('Example'),
                html.P('Add Window'),
                html.P('Choose Model'),
                html.P('Import Image'),
                ],

             id='top-bar'
             ),
    html.Div([
        html.Div(className='card', children=[
            html.P('Image') ]),
        html.Div(className='card', children=[
            html.P('Image') ]),
        html.Div(className ='card', children=[
            html.P('Image') ]),
        ], 
            className='row-container')
        ])

app.css.append_css({
    'external_url': 'assets/styles.css'
    })
        




if __name__ == '__main__':
    app.run(debug=True)
