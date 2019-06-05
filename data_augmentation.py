#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 04:10:42 2019

@author: adel
"""

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
#import scipy.misc
import os
import cv2
import numpy as np
#import mxnet as mx


def multiply_image(image,R,G,B , loc):
    print("LOCATION", loc)
    image=image*[R,G,B]
    cv2.imwrite(loc, image)


def black_hat_image(image, shift , loc):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    cv2.imwrite(loc, image)


def sharpen_image(image , loc):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(loc, image)

def emboss_image(image , loc):
    kernel_emboss_1=np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
    image = cv2.filter2D(image, -1, kernel_emboss_1)+128
    cv2.imwrite(loc, image)

def edge_image(image,ksize , loc):
    image = cv2.Sobel(image,cv2.CV_16U,1,0,ksize=ksize)
    cv2.imwrite(loc, image)

def addeptive_gaussian_noise(image , loc):
    h,s,v=cv2.split(image)
    s = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    h = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    v = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    image=cv2.merge([h,s,v])
    cv2.imwrite(loc, image)



def generate_observations(augmentation_type, images , label_path, video_name, new_augmented_video_num):

    for image in images:
        # image location
        image_loc = label_path + '/' +video_name +'/'+ image
        img_filename = image
        # load the image
        img = load_img(image_loc)
        #pyplot.imshow(img)
        #pyplot.show()
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        #,brightness_range=[0.2,1.0], width_shift_range=[-100,100] zoom_range=[0.5,1.0]
       
        datagen = ImageDataGenerator(zoom_range=[0.7,1.0])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(1):
            # define subplot
            #pyplot.subplot(330 + 1 + 5)
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            #print('image>>>>>>>', image)
       
            #creating directories of new augmented data
            output_dir =label_path + '/adel_'+ str(new_augmented_video_num)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            #augmented image location
            remove_img_extention = img_filename[:-4]
            augmented_image_location = output_dir+'/'+ remove_img_extention +'_' +'.jpg'
            #save the augmented image to location
            #scipy.misc.imsave(augmented_image_location, image)
            if(augmentation_type == 0):
                multiply_image(image,1.5,1.5,1.5, augmented_image_location)
            elif (augmentation_type == 1):
                sharpen_image(image, augmented_image_location)
            elif (augmentation_type == 2):
                edge_image(image,3, augmented_image_location)
            elif (augmentation_type == 3):
                addeptive_gaussian_noise(image, augmented_image_location)
            elif (augmentation_type == 4):
                emboss_image(image, augmented_image_location)
            elif (augmentation_type == 5):
                black_hat_image(image,500, augmented_image_location)
           

def main():
    rootdir = 'testdir'
    labels = os.listdir(rootdir)
    for label in labels:
      videos = os.listdir(rootdir + '/' + label)
      videos_count = len(videos)
      # videos couter
      videos_counter = 0
      augmentation_types_counter=0
      print('************************************')
      print('Label:',label,'-- videos_count:',videos_count)
      if 4 <= videos_count < 12:
          for i in range(videos_count , 12):
              print('Label: ',label,'-- Augmented video_'+str(i+1))
              images = os.listdir(rootdir + '/' + label + '/' + videos[videos_counter])
              # generate augmented images
              generate_observations(augmentation_types_counter, images , rootdir + '/' + label , videos[videos_counter], i+1 )
              videos_counter+=1
              augmentation_types_counter = (augmentation_types_counter+1) % 5
              #to handle out of range ERROR
              videos_counter = videos_counter % videos_count
      else:
            print('Label:',label,'will be escaped!')
  
  
if __name__ == '__main__':
    main()