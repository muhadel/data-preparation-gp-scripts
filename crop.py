import glob # for the path and reeegex
import os
import math
import random
import cv2
import numpy as np
import pandas as pd
#import natsort
  
labelDir= 'datasetImage'
newdir= 'E:\dataset\AdelImages'
labels = os.listdir(labelDir)
numOfFrames = 3
rate= 15 / numOfFrames
print('List is :' )
count2 = 0

for label in labels:
  listing = os.listdir(labelDir + '/' + label)
  for video_ in listing:
    images = os.listdir(labelDir + '/' + label + '/' + video_)
    count = 1

    for image in images:
#       if count == 1:
      filename = labelDir + '/' + label + '/' + video_ + '/' + image 
      newImgName = newdir + '/' + label + '/' + video_ + '/' + image 
      #print(filename)
      img = cv2.imread(filename)
      img = np.array(img)
      crop_img = img[0:317, :]
      #print(crop_img.shape)
      image=cv2.resize(crop_img,(304,317), interpolation = cv2.INTER_AREA)
      cv2.imwrite(filename,crop_img)
#         cv2.imshow("cropped", crop_img)
#         cv2.waitKey(0)
  count2 += 1
  print(str(int(count2)) + '-Done Crop Label: '+ label)

print("Dataset Successfuly Croped")