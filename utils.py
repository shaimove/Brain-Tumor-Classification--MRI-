# utils.py
import os
import shutil
import random
import numpy as np
import cv2

#%% 
def CalculateStats(data_root):
    # This function calculate the mean and std of the dataset
    # the function read every image, and save the number of pixels, mean, std
    # at the end it return the mean and std of all dataset
    mean = []
    std = []
    resoution = []
    
    # for all sub folders (classes)
    for sub_class in os.listdir(data_root):
        # choose sub folder
        sub_folder = os.path.join(data_root, sub_class)
        
        # for every image in subfolder
        for img in os.listdir(sub_folder):
            # create image path
            img_path = os.path.join(data_root,sub_folder,img)
            
            # read image
            image = np.array(cv2.imread(img_path,0))
            
            # calculate stats
            mean.append(np.mean(image))
            std.append(np.std(image))
            resoution.append(image.shape[0]*image.shape[1])
    
    # After reading all the dataset, we need to calculate the stats
    mean,std,resoution = np.array(mean),np.array(std),np.array(resoution)
    
    # weight according to sum of pixels
    weights = resoution/np.sum(resoution)
    
    # calculate weighted std and mean 
    mean_pixel = np.sum(mean * weights)
    std_pixel = np.sum(std * weights)
    
    
    return mean_pixel,std_pixel
            
#%% The dataset came only with train and test set. we need to create validation set ahead

def CreateValidationSet(training_folder,validation_folder):
    
    precent_of_images = 0.2
    
    # for every sub folder
    for sub_class in os.listdir(training_folder):
        # choose sub folder 
        sub_folder = os.path.join(training_folder, sub_class)
        destintaion = os.path.join(validation_folder,sub_class)
        
        # create list of images 
        list_of_images = os.listdir(sub_folder)
        
        # create random list of images to move
        num_of_image = int(precent_of_images * len(list_of_images))
        images_to_move = random.sample(list_of_images,num_of_image)
        
        # move from training to source
        for img in images_to_move:
            file = os.path.join(sub_folder,img)
            shutil.move(file,destintaion)
            
    
    return print('Done!')
    




            
