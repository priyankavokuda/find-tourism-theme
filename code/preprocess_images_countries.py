# -*- coding: utf-8 -*-
"""
Created on Sat May 21 19:50:24 2016
@author: priyanka

This code is used to pre-process the the images of all countries present in "countries" folder. "countries" folder contains folders with name visit_*country name*, these folders have the images of respective countries. 
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

rgb_feat_size=img_size*img_size*3

total_feat_size=(hog_feat_size+rgb_feat_size)


os.chdir("countries/")
countries=['argentina',
'australia',
'austria',
'bangladesh',
'belgium',
'bhutan',
'bolivia',
'bosnia',
'botswana',
'brazil',
'cambodia',
'canada',
'chile',
'china',
'croatia',
'czechrepublic',
'denmark',
'ecuador',
'egypt',
'estonia',
'fiji',
'finland',
'france',
'germany',
'greece',
'guatemala',
'holland',
'honduras',
'hungary',
'iceland',
'india',
'indonesia',
'iran',
'italy',
'malaysia',
'maldives',
'morocco',
'myanmar',
'nepal',
'newzealand',
'norway',
'russia',
'tanzania',
'uae']
for path in countries:
    pwd=os.getcwd()
    os.chdir("visit_"+path)
    print path
    num_files=len(glob.glob('*.jpg'))
    feature=np.zeros((num_files,total_feat_size))
    img_cnt=0 
    for fileName in glob.glob("*.jpg"):
        img = misc.imread(fileName)
        mFIleName=fileName.replace(".jpg","")
        
        resized_img=misc.imresize(img,(img_size,img_size))
        
        resized_img_gray = color.rgb2gray(resized_img)
        hog_features, hog_image = hog(resized_img_gray, orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualise=True)   
        #print hog_features.shape
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


