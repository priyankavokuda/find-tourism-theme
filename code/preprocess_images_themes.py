# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 18:31:18 2016
@author: priyanka

This code is used to pre-process the the images of all themes present in "themes" folder. "themes" folder contains folders with name of the themes, these folders have the images of respective themes.
The preprocessed images are stored in .npy format. 
"""

import numpy as np
import glob
from scipy import misc
import os
from skimage.feature import hog
from skimage import  color
from matplotlib import pyplot as plt
import scipy.ndimage
import os

img_size=50
hog_feat_size=288
num_training_ings=1000

rgb_feat_size=img_size*img_size*3

total_feat_size=(hog_feat_size+rgb_feat_size)


os.chdir("themes/")
themes=['architecture','beach','desert','nature','wildlife','winterlandscape']
for path in themes:
    pwd=os.getcwd()
    os.chdir(path)
    print path
    feature=np.zeros((num_training_ings,total_feat_size))
    img_cnt=0
    for fileName in glob.glob("*.jpg"):
        img = misc.imread(fileName)
        mFIleName=fileName.replace(".jpg","")
        
        resized_img=misc.imresize(img,(img_size,img_size))
        
        resized_img_gray = color.rgb2gray(resized_img)
        hog_features, hog_image = hog(resized_img_gray, orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualise=True)   
        #hog_image=misc.toimage(hog_image)
        #misc.imsave(mFIleName+"hog.png", hog_image)
        #resized_img_luv = color.rgb2luv(resized_img)
        rgb_features=np.reshape(resized_img,(1,rgb_feat_size))
    #    rgb_feature_img=misc.toimage(rgb_features)
    #    misc.imsave(mFIleName+"rgb.png", rgb_feature_img)    
        hog_features=np.reshape(hog_features,(1,hog_feat_size))
        
        total_features=np.column_stack((rgb_features,hog_features))
        total_features=np.reshape(total_features,(1,total_feat_size))
        feature[img_cnt,:]=total_features
        img_cnt=img_cnt+1
        #misc.imsave(fileName+'resize.p', resized_img)
    os.chdir(pwd)
    #print cnt
    print feature.shape 
    #os.chdir("../")
    npy_name=path.replace("/","")  
    np.save(npy_name,feature)


