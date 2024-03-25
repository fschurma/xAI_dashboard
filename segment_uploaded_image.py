data_transforms_256 = transforms.Compose([  # Resize images
    transforms.ToTensor()  # Convert to tensor
])

def load_model_segnet(model_file_path = 'assets/models/net_epoch24.pth'):
    model = SegNet(3,20) #one additional class for pixel ignored
    model.load_state_dict(torch.load(model_file_path,map_location=torch.device('cpu')))
    
    return model

def predict_segnet(image, model_file_path='assets/models/net_epoch24.pth'):
    model = load_model_segnet(model_file_path)
    input_tensor = data_transforms_256(image).unsqueeze(0)

    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
    _, predicted = torch.max(output.data, 1) 
    
    print(predicted.shape)
    print(predicted)
    predicted = predicted.squeeze().cpu().numpy() 

    # Map predicted class indices to random colors
    num_classes = 20  # Number of classes
    height, width = predicted.shape
    print(predicted.shape)
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate random colors for each class
    np.random.seed(42)  # Set a seed for reproducibility
    random_colors = np.random.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)

    for label in range(num_classes):
        color_image[predicted == label] = random_colors[label]

    input_image_transformed = np.transpose(input_tensor.squeeze(0).cpu().numpy(), (1, 2, 0))



    return input_image_transformed, color_image

def image_to_base64(image):
    # Konvertiere das Bild in ein Bytes-Objekt
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    # Encode das Bild als Base64-codierte Zeichenfolge
    img_base64 = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')
    return img_base64


@app.callback(
    Output('output-segmentation-1', 'children'),
    Input('import_image_1', 'contents'),
    Input('import_image_1', 'filename'),
    Input('import_image_1', 'last_modified')
)

def image_segmentation(contents, filename, last_modified):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        try:
            if 'png' in filename:
                image = Image.open(io.BytesIO(decoded)).convert('RGB')
            elif 'jpg' in filename:
                image = Image.open(io.BytesIO(decoded)).convert('RGB')
            
            elif 'jpeg' in filename:
                image = Image.open(io.BytesIO(decoded)).convert('RGB')

        except Exception as e:
            print(e)
            return html.Div(['There was an error processing this file.'])

        input_image_transformed, color_image =  predict_segnet(image)

        color_image_base64 = image_to_base64(color_image)

        
        children = html.Div(children[html.Img(src=f'data:image/png;base64,{color_image_base64}')])

        return children