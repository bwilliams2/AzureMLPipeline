import os
import torch
import json
import logging
import pandas as pd
# import
import os
import mlflow
import argparse
# from azureml.core import Workspace, Dataset
# from azureml.core.authentication import MsiAuthentication
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()

subscription_id=os.getenv("SUBSCRIPTION_ID")
resource_group=os.getenv("RESOURCE_GROUP")
workspace_name=os.getenv("WORKSPACE_NAME")
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# TODO - add mlflow logging


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """

    global model, data, data_version
    # load current data
    data_asset = ml_client.data.get("LocationTestData", label="latest")
    data_version = data_asset.version
    print("Loading data")
    print("Current data version is " + str(data_version))
    data = pd.read_csv(data_asset.path)


    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model", "data", "model.pth")
    # deserialize the model file back into a sklearn model

    # load model
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # model = model.to(device)
    logging.info("Init complete")


def run(input_data):
    data_asset = ml_client.data.get("LocationTestData", label="latest")
    data_version = data_asset.version

    print("Checking data version")
    print("Current data version is " + str(data_version))
    print("New data version is " + str(data_asset.version))
    if data_version != data_asset.version:
        # Refresh data
        data_version = data_asset.version
        data = pd.read_csv(data_asset.path)

    input_data = torch.tensor(json.loads(input_data)["data"])

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # model.eval()
    with torch.no_grad():
        output = model(input_data)
        softmax = nn.Softmax(dim=1)
        pred_probs = softmax(output).numpy()[0]
        index = torch.argmax(output, 1)

    return {"label": classes[index], "probability": str(pred_probs[index])}