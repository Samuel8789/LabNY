# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 10:45:09 2021

@author: sp3660
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caiman as cm
import os
import tifffile
from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift

file_1=r'C:\Users\sp3660\Desktop\210615_SPHV_FOV1_10minspont_50024_narrored_920_with-000_d1_256_d2_256_d3_1_order_F_frames_33901_MCkalmanstd_projection.tif'
file_2=r'C:\Users\sp3660\Desktop\210615_SPHV_FOV1_1050tomato_50024_narrored_920_with-000_d1_256_d2_256_d3_1_order_F_frames_5_average_projection.tif'
with tifffile.TiffFile(file_1) as tffl:
    input_arr1 = tffl.asarray()
with tifffile.TiffFile(file_2) as tffl:
    input_arr2 = tffl.asarray()
    

input_arr1=np.expand_dims(input_arr1, 0)
input_arr2=np.expand_dims(input_arr2, 0)
stack=np.concatenate((input_arr1,input_arr2),axis=0)             
movie1=cm.movie(stack)
regist=movie1.motion_correct()

plt.imshow(input_arr1,  vmax=255, cmap='inferno')
plt.imshow(input_arr2,  vmax=6000, cmap='inferno')





# Open the image files.
img1_color = cv2.imread(file_1)  # Image to be aligned.
img2_color = cv2.imread(file_2)    # Reference image.
image=Image.open(file_1)



# Convert to grayscale.
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
height, width = img2.shape
 
# Create ORB detector with 5000 features.
orb_detector = cv2.ORB_create(5000)
 
# Find keypoints and descriptors.
# The first arg is the image, second arg is the mask
#  (which is not required in this case).
kp1, d1 = orb_detector.detectAndCompute(img1, None)
kp2, d2 = orb_detector.detectAndCompute(img2, None)
 
# Match features between the two images.
# We create a Brute Force matcher with
# Hamming distance as measurement mode.
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
 
# Match the two sets of descriptors.
matches = matcher.match(d1, d2)
 
# Sort matches on the basis of their Hamming distance.
matches.sort(key = lambda x: x.distance)
 
# Take the top 90 % matches forward.
matches = matches[:int(len(matches)*0.9)]
no_of_matches = len(matches)
 
# Define empty matrices of shape no_of_matches * 2.
p1 = np.zeros((no_of_matches, 2))
p2 = np.zeros((no_of_matches, 2))
 
for i in range(len(matches)):
  p1[i, :] = kp1[matches[i].queryIdx].pt
  p2[i, :] = kp2[matches[i].trainIdx].pt
 
# Find the homography matrix.
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
 
# Use this matrix to transform the
# colored image wrt the reference image.
transformed_img = cv2.warpPerspective(img1_color,
                    homography, (width, height))
 
# Save the output.
cv2.imwrite('output.jpg', transformed_img)