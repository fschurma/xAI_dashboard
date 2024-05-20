# xAI dashboard for image segmentation

## Implementation

1. Clone this directory
2. Install following modules

    - dash
    - dash_bootstrap_components
    - torch
    - pillow
    - torchvision
    - captum (If it doesn't work in conda it should be installed with `pip install captum`.)
    - opencv

The pytorch_grad_cam module has to be installed with `pip install grad-cam`.

Else you can install the requirments with `pip install -r requirements.txt`

## Use of the app
1. Let run dashboard.py (if no IDE is available just let run `python dashboard.py` in command line)
2. Open the demo app from topbar
3. Choose a segmentation model in both filter sections to compare
![PNG](/assets/images/model_selection.png)
4. Choose a label
![PNG](/assets/images/label_selection.png)
5. Choose a xAI method in both filter sections to compare
![PNG](/assets/images/label_selection.png)

