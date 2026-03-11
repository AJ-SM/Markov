
import cv2 
import torch
import  torch.optim as optim 
from torch.nn import nn 
import pandas as pd 

path = r"D:\Storeage-1\Main\ML-Model\data_set\pandu.mp4"

model = torch.load("../models/Antispoofing/Markov_Mark0.pth")
