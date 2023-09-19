import os
import torch
import json
import logging
# imports
import os
import mlflow
import argparse

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
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model", "data", "model.pth")
    # deserialize the model file back into a sklearn model

    # load model
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # model = model.to(device)
    logging.info("Init complete")


def run(input_data):
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