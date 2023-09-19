import streamlit as st
import os
import urllib
import requests
from PIL import Image
import numpy as np
import json
import torch
import tempfile
import torchvision
import torchvision.transforms as transforms

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
    r_data = image.numpy()
    r_data = json.dumps({"data": r_data.tolist()})

    body = str.encode(r_data)


    url = 'https://pytorch-model-endpoint.eastus.inference.ml.azure.com/score'
    # Replace this with the primary/secondary key or AMLToken for the endpoint
    api_key = os.getenv("ENDPOINT_KEY")
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    # Remove this header to have the request observe the endpoint traffic rules
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'pytorch-deployment' }

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = json.loads(response.read().decode())
        print(result)
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))

    return result

def main():
    st.title("Image Submission App")
    
    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        # Convert uploaded file to a numpy array
        result = process_uploaded_file(uploaded_file)
        # st.write("Shape of the image array:", image_array.shape)
        st.write("Image Label:", result["label"])
        st.write("Label Probability:", result["probability"])
    

    
if __name__ == "__main__":
    main()
