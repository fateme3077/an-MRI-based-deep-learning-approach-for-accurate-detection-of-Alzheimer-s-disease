# an-MRI-based-deep-learning-approach-for-accurate-detection-of-Alzheimer-s-disease
# imports
import cv2
import numpy as np
import os
import cv2
import scipy.io

###############################################################################
# Define Variables
files = ['MildDemented',
         'ModerateDemented',
         'NonDemented',
         'VeryMildDemented'];

k = 0

# Creat Tensors
Data = np.arange(7216*100*100*1)
Data = np.reshape(Data, (7216,100,100))

t = []

# Load Data
for i in range(0,4): # i = 0, 1, 2, 3
    res = os.listdir('./MRI_Alzheimer/' + files[i])
    print(files[i] + ' : ' + str(len(res)) + ' images')

    for j in range(0,len(res)):
        img = cv2.imread('./MRI_Alzheimer/' + files[i] + '/' + res[j])
        img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize Data
        img_ = cv2.resize(img_,(100,100))
