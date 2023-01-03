import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

file_path = 'D:/wp/data/xray_artknife_a_/crop'

file = os.listdir(file_path)

for f in file:
    img1 = cv2.imread(file_path + '/' + f, -1)
    if(f != 'Thumbs.db'):
        print(img1.shape)
        x, y = img1.shape
        
        dst = cv2.pyrUp()