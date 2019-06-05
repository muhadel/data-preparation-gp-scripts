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
import scipy.misc
import os



def generate_observations(augmentation_type, images , label_path, video_name, new_augmented_video_num):

    for image in images:
        # image location
        image_loc = label_path + '/' +video_name +'/'+ image
        img_filename = image
        # load the image
        img = load_img(image_loc)
        pyplot.imshow(img)
        #pyplot.show()
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        #,brightness_range=[0.2,1.0], width_shift_range=[-100,100] zoom_range=[0.5,1.0]
        if(augmentation_type == 0):
            datagen = ImageDataGenerator(featurewise_center=True,
                                         featurewise_std_normalization=True,
                                         horizontal_flip=True)
        elif(augmentation_type == 1):
            datagen = ImageDataGenerator(zca_whitening=True, samplewise_std_normalization=True)
        elif (augmentation_type == 2):
                datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
        elif (augmentation_type == 3):
                datagen = ImageDataGenerator(horizontal_flip=True)
        elif (augmentation_type == 4):
                datagen = ImageDataGenerator(samplewise_center=True)
        elif (augmentation_type == 5):
                datagen = ImageDataGenerator(shear_range=0.5, cval=0.3,
                                             featurewise_std_normalization=True,
                                             channel_shift_range=0.8,
                                             rotation_range=5 )
        
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(1):
            # define subplot
            pyplot.subplot(330 + 1 + 5)
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            #creating directories of new augmented data
            output_dir =label_path + '/adel_'+ str(new_augmented_video_num)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            #augmented image location
            remove_img_extention = img_filename[:-4]
            augmented_image_location = output_dir+'/'+ remove_img_extention +'_' +'.jpg'
            #save the augmented image to location
            scipy.misc.imsave(augmented_image_location, image)


    



'''
AUGMENTATION TYPE *********** NUMBER
-------------------------------------
zoom_range=[0.8,0.5]           0
samplewise_std_normalization   1
brightness_range=[0.2,1.0]     2
horizontal_flip=True           3
samplewise_center=True         4 labany
Rotation                       5

'''
def main():
    rootdir = 'testdir'
    labels = os.listdir(rootdir)
    for label in labels:
      videos = os.listdir(rootdir + '/' + label)
      
      videos_count = len(videos)
      counter = 0
      augmentation_types_counter=0
      print('************************************')
      print('Label:',label,'-- videos_count:',videos_count)
      if 4 <= videos_count < 12:
          for i in range(videos_count , 12):
              print('Label: ',label,'-- Augmented video_'+str(i+1))
              images = os.listdir(rootdir + '/' + label + '/' + videos[counter])
              generate_observations(augmentation_types_counter, images , rootdir + '/' + label , videos[counter], i+1 )
              
              counter+=1
              augmentation_types_counter = (augmentation_types_counter+1) %5
              #to handle out of range ERROR
              counter = counter % videos_count
      else:
            print('Label:',label,'will be escaped!')
  
  
if __name__ == '__main__':
    main()