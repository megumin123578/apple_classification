import torch
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict, List
import torchvision
from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer



# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dir = "Training"
test_dir = "Test"

# Write transform for image
data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
])


# creating training set

train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform = data_transform, # transform to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root = test_dir,
                                 transform = data_transform)

# Get the class names from the target directory
class_names_found = sorted([entry.name for entry in list(os.scandir(train_dir))])



# Make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class TinyVGG(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=2,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
        # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion

torch.manual_seed(42)
# Load the model
model_0 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(train_data.classes))

# Load the model's state
model_0.load_state_dict(torch.load("model0.pth"))

def kq(input_img) -> Tuple[Dict, float]:
    # Start the timer
    start_time = timer()

    # Setup custom image path
    custom_image = input_img


    # Load in custom image and convert the tensor values to float32
    image = torchvision.io.read_image(str(custom_image)).type(torch.float32)

    # Divide the image pixel values by 255 to get them between [0, 1]
    custom_image = image / 255.

    # Create transform pipeline to resize image
    custom_image_transform = transforms.Compose([
        transforms.Resize((64, 64), antialias=True),
    ])

    # Transform target image
    custom_image_transformed = custom_image_transform(custom_image)

    # Add batch dimension to the image
    custom_image_transformed = custom_image_transformed.unsqueeze(0)

    model_0.eval()
    # Make a prediction on image
    with torch.inference_mode():
        custom_image_pred = model_0(custom_image_transformed)
            # Make a prediction on image with an extra dimension



    class_names = train_data.classes

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)

    # Find the predicted label
    custom_image_pred_class = str(class_names[custom_image_pred_label.cpu().item()])  # put pred label to CPU, otherwise will error

    #Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(custom_image_pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time
    

from rembg import remove
import cv2
import numpy as np
import gradio as gr
import random

def rm(input_img):
    img_0 = cv2.cvtColor(input_img, cv2.COLOR_RGBA2BGR)
    cv2.imwrite('bruh_0.jpg',img_0)
    
    # Remove background
    img = remove(input_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    # Save the processed image
    cv2.imwrite('bruh.jpg', img)

    resize_picture_after_rm = cv2.imread('bruh.jpg')
    resize_picture_after_rm = cv2.resize(resize_picture_after_rm, (200, 200))
    cv2.imwrite('bruh.jpg', resize_picture_after_rm)

    picture_after_remove_background = 'bruh.jpg'

     # Call the kq function and unpack its return values
    pred_labels_and_probs, pred_time = kq('bruh_0.jpg')
    
    return picture_after_remove_background, pred_labels_and_probs, pred_time



def sepia(input_img):
    img_1 = rm(input_img)
    
    return img_1

new_width = 200
new_height = 200

a = cv2.imread('tao_chin_internet.jpg')
# Resize the image
resized_image = cv2.resize(a, (new_width, new_height))
cv2.imwrite('a.jpg', resized_image)

b = cv2.imread('tao_do_tren_cay_internet.jpg')
# Resize the image
resized_image = cv2.resize(b, (new_width, new_height))
cv2.imwrite('b.jpg', resized_image)

c = cv2.imread('tao_hong_internet.jpg')
# Resize the image
resized_image = cv2.resize(c, (new_width, new_height))
cv2.imwrite('c.jpg', resized_image)

d = cv2.imread('tao_xanh_giao_dien.jpg')
# Resize the image
resized_image = cv2.resize(d, (new_width, new_height))
cv2.imwrite('d.jpg', resized_image)

e = cv2.imread('tao_xanh_internet.jpg')
# Resize the image
resized_image = cv2.resize(e, (new_width, new_height))
cv2.imwrite('e.jpg', resized_image)

f = cv2.imread('tao_thoi_1.jpg')
# Resize the image
resized_image = cv2.resize(f, (new_width, new_height))
cv2.imwrite('f.jpg', resized_image)

g = cv2.imread('AppleSootyBlotch-423x322.jpg')
# Resize the image
resized_image = cv2.resize(g, (new_width, new_height))
cv2.imwrite('g.jpg', resized_image)


# Create a list of example inputs to our Gradio demo
example_list = ['a.jpg','b.jpg','c.jpg','d.jpg','e.jpg','f.jpg','g.jpg']

cv2.imread('tao_xanh_giao_dien.jpg')

title = "Apple's quality classification using AI 🍏🍎"
description = "How it works? Click the Upload button to upload image from your device or choose camera to capture a picture , click the Submit button to get the result."
article = article = (
    "<body>"
"    <p>Apples, the crisp and succulent fruit synonymous with health, not only satisfy our taste buds but also pack a punch when it comes to nutritional benefits. This article delves into the various essential nutrients found in apples, highlighting why this fruit has earned its reputation as a symbol of well-being.<p>"

"    <h2>Vitamins:</h2>"
"    <p>Apples are a rich source of vitamins, particularly vitamin C. This vital nutrient acts as a powerful antioxidant, playing a crucial role in strengthening the immune system and protecting cells from damage. Additionally, apples contain a variety of B-vitamins, contributing to energy metabolism and overall vitality.</p>"

"    <h2>Dietary Fiber:</h2>"
"    <p>One of the standout nutritional features of apples is their high fiber content. Apples boast both soluble and insoluble fiber, promoting digestive health. The soluble fiber helps manage cholesterol levels, while the insoluble fiber aids in maintaining regular bowel movements. Including apples in your diet is a delicious way to support a healthy gut.</p>"

"    <h2>Minerals:</h2>"
"    <p>Apples contain an array of minerals that are essential for various bodily functions. Potassium, for instance, plays a crucial role in maintaining proper heart function and blood pressure. Additionally, apples provide small amounts of minerals like calcium and magnesium, contributing to bone health and overall well-being.</p>"

"    <h2>Phytonutrients:</h2>"
"    <p>Beyond traditional vitamins and minerals, apples are rich in phytonutrients. These natural compounds, such as flavonoids and polyphenols, have potent antioxidant properties. They help combat oxidative stress in the body, potentially reducing the risk of chronic diseases and promoting longevity.</p>"

"    <h2>Low in Calories, High in Nutrition:</h2>"
"    <p>Apples are a nutrient-dense food, meaning they provide a substantial amount of nutrients while being relatively low in calories. This makes them an excellent choice for those aiming to maintain a healthy weight without compromising nutritional intake.</p>"
"</body>"

)




demo = gr.Interface(
    sepia,
    inputs=gr.Image(),
    outputs=["image", gr.Label(num_top_classes=3, label="Predictions"), gr.Number(label="Prediction time (s)")],
    title=title,
    description=description,
    article=article,
    allow_flagging='never',
    examples=example_list,
    css="""
/* Your custom CSS styles here */
body {
    background-color: #f2f2f2;
    font-family: Arial, sans-serif;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;  /* Three columns, equal width */
    gap: 20px;  /* Adjust the gap between the columns */
}

article {
    max-width: 800px;
    line-height: 1.6;
    margin-left: 20px;  /* Adjust the margin for the left column */
}

h1 {
    color: #fef65b;
    font-size: 50px;
    margin-bottom: 16px;
    text-align: center;  /* Canh giữa title */
}

h2 {
    color: #ff9900;
    font-size: 27px;
    margin-bottom: 16px;
    font-weight: bold;
}

p {
    color: #ffffff;
    font-size: 20px;
    margin-bottom: 12px;
    padding-right: 10px;
}

img {
    height: auto;
    width: auto;   /* Chiều rộng tự động điều chỉnh */
    margin: auto;  /* Canh giữa ảnh */
    object-fit: contain; /* Đảm bảo ảnh không bị méo */
}

"""
)

demo.launch(debug=False, # print errors locally
            share=True) # generate a publically shareable URL?