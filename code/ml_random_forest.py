# -*- coding: utf-8 -*-
"""
Created on Sun May  8 22:43:48 2016

This code predicts the tourism theme of the country given prediction model trained on different tourism theme images using Random forests. 
And Cartopy python package designed to make drawing maps is used to visualize output in world map.
Random forest implementation from scikit-learn http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
Cartopy implementation from http://scitools.org.uk/cartopy/docs/v0.13/matplotlib/feature_interface.html
@author: priyanka
"""

import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import pickle
from matplotlib import pyplot as plt
from collections import Counter
import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs


img_size=50
hog_feat_size=288
rgb_feat_size=img_size*img_size*3
total_feat_size=(hog_feat_size+rgb_feat_size)

os.chdir("themes/")
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
themes=['architecture','beach','desert','nature','wildlife','winterlandscape']
themes_mapping={'architecture':0,'beach':1,'desert':2,'nature':3,'wildlife':4,'winterlandscape':5}
countries_mapping={}
colour_codes={0:'brown',1:'blue',2:'yellow',3:'green',4:'orange',5:'cyan'}
country_correction={'Bosnia':'Bosnia and Herz.','Czechrepublic':'Czech Republic','Holland':'Netherlands','Newzealand':'New Zealand','Tanzania':'United Republic of Tanzania','Uae':'United Arab Emirates'}

X=np.load("theme.npy")
y=np.load("label.npy")
y=np.ravel(y)
y=y.astype(int)

clf = RandomForestClassifier(n_estimators=200) 
clf.fit(X, y) 
s1 = pickle.dumps(clf)
clf = pickle.loads(s1)

os.chdir("../countries/")

for i in range(len(countries)):
    X=np.zeros((0,total_feat_size))
    y=np.zeros((0,1))
    data=np.load(countries[i]+".npy")
    X=np.row_stack((X,data))
    y_predict=clf.predict(X)
    maxcnt = Counter(y_predict)
    countries_mapping[countries[i]]=maxcnt.most_common(1)[0][0]

ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
ax.set_extent([-150, 60, -25, 60])

shpfilename = shpreader.natural_earth(resolution='110m',
                                      category='cultural',
                                      name='admin_0_countries')
reader = shpreader.Reader(shpfilename)
countries_ = reader.records()

for country in countries_:
    for mycountry in countries:
        if mycountry.title() in country_correction:
            name=country_correction[mycountry.title()]
        else:
            name=mycountry.title()
        if country.attributes['admin'] == name:
            ax.add_geometries(country.geometry, ccrs.PlateCarree(),
                              facecolor=colour_codes[countries_mapping[mycountry]],
                           label=country.attributes['adm0_a3'])
plt.show()
plt.savefig('country.png',dpi=1200) 
