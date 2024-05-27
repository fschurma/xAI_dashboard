# In this file the future work for importing the image and process it to display it in the web app is done.

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Image Upload


def parse_contents(contents):
    print("Contents:", contents)  # Debugging-Ausdruck für den Inhalt des Bildes
    if contents is not None:
        # HTML-Element für das Bild erstellen
        image_element = html.Img(src=contents)
        # Einzelnes Div-Element mit allen Inhalten erstellen
        return image_element


@app.callback(
    Output('output-image-upload_1', 'children'),
    Output('output-image-upload_2', 'children'),
    Input('import_image_1', 'contents'),
    Input('import_image_1', 'filename'),
    Input('import_image_1', 'last_modified'),
    Input('import_image_2', 'contents'),
    Input('import_image_2', 'filename'),
    Input('import_image_2', 'last_modified')

)
def update_output_1(contents_1, filename_1, last_modified_1, contents_2, filename_2, last_modified_2):
    
    contents = []


    # Überprüfen, ob contents eine Liste ist, wenn nicht, eine Liste daraus machen
    if not (isinstance(contents_1, list) or isinstance(contents_2, list)):
        if contents_1 is not None:
            contents = [contents_1]
        elif contents_2 is not None:
            contents = [contents_2]
    
    if not (isinstance(filename_1, list) or isinstance(filename_2, list)):
        if filename_1 is not None:
            filename = [filename_1]
        elif filename_2 is not None:
            filename = [filename_2]
    
    if not (isinstance(last_modified_1, list) or isinstance(last_modified_2, list)):
        if last_modified_1 is not None:
            last_modified = [last_modified_1]
        elif last_modified_2 is not None:
            last_modified = [last_modified_2]
        
    if contents:
        children = [
            parse_contents(c) for c in contents]
        
        img = children[0]


        return img, img
    
    else:
        print ("No image uploaded. Please upload an image.")