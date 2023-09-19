import streamlit as st
import os
import requests
from PIL import Image
import numpy as np
import json
import torch
import tempfile
import torchvision
import torchvision.transforms as transforms

def main():
    st.title("Image Submission App")
    
    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        # Convert uploaded file to a numpy array
        image_array = process_uploaded_file(uploaded_file)
        st.write("Shape of the image array:", image_array.shape)
    
def process_uploaded_file(uploaded_file):
    # Open the uploaded image using PIL
    image = Image.open(uploaded_file)
    
    # Convert uploaded file to a numpy array
    result = process_uploaded_file(uploaded_file)
    st.write("Image Label:", result["label"])
    st.write("Label Probability:", result["probability"])
    
def process_uploaded_file(uploaded_file):
    # Open the uploaded image using PIL
    image = Image.open(uploaded_file)
    # define test dataset DataLoaders
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    image = transform(image).float()
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    data = image.numpy()
    input_data = json.dumps({"data": data.tolist()})


    with tempfile.TemporaryDirectory() as tmpdir:
        # use the temporary directory here
        file = os.path.join(tmpdir, "request.json")
        with open(file, "w") as outfile:
            outfile.write(input_data)
        
        result = {"label": "ship", "probability": "0.9999"}
    return result

    
if __name__ == "__main__":
    main()
