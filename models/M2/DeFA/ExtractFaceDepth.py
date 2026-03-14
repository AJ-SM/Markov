import cv2
import numpy as np
from api import PRN

# initialize PRNet
prn = PRN(is_dlib=True)

# load image
image = cv2.imread(r"D:\Storeage-1\Main\ML-Model\models\M2\RDJ.jpg")

# run reconstruction
pos = prn.process(image)

# extract vertices
vertices = prn.get_vertices(pos)

# extract depth
depth = vertices[:,2]

print(vertices.shape)