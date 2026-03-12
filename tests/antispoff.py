import cv2
import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np

from models.Antispoofing.features.antispoof import get_full_feature_vector


def genData(path):
    feature = []

    image_extensions = {".jpg", ".png", ".jpeg", ".webp"}

    if os.path.splitext(path)[1].lower() in image_extensions:
        features = get_full_feature_vector(path, flg=True)
        feature.append(features)

    else:
        cap = cv2.VideoCapture(path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            offs = get_full_feature_vector(frame, flg=False)
            feature.append(offs)

        cap.release()

    return feature


path = r"D:\Storeage-1\Main\ML-Model\data_set\pandu.mp4"

gendata = genData(path)

E_data = []
gamma_data = []

for vec in gendata:

    E_data.append(vec[0])
    gamma_data.append(vec[1])


# -----------------------------
# Prepare model input
# -----------------------------

X = np.column_stack((E_data, gamma_data))

X = torch.tensor(X, dtype=torch.float32)


# -----------------------------
# Load model
# -----------------------------

model = torch.load(r"D:\Storeage-1\Main\ML-Model\models\Antispoofing\Markov_Mark0.pth")

model.eval()


# -----------------------------
# Inference
# -----------------------------

with torch.no_grad():

    outputs = model(X)

    preds = torch.argmax(outputs, dim=1)

print("Predictions:", preds)