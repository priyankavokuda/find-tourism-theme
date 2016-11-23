# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 12:25:08 2016

This code predicts the tourism theme of the country given prediction model trained on different tourism theme images using SVM. 
Before using SVM, to reduce the dimension of the data PCA is used and the size of the components is decreaased to 18 from 7788. 
SVM implementation from http://scikit-learn.org/stable/modules/svm.html
PCA implementation from http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
@author: priyanka
"""


import numpy as np
from sklearn import svm
import os
from sklearn.cross_validation import train_test_split
import pickle
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


img_size=50
hog_feat_size=288
rgb_feat_size=img_size*img_size*3
total_feat_size=(hog_feat_size+rgb_feat_size)


def plot_confusion_matrix(cm,clf_type, label_names=False,title='Confusion matrix(log scale)', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=90)
    plt.yticks(tick_marks, label_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(clf_type+'confusion_matrix.png',dpi=1200)

os.chdir("themes/")
themes=['architecture','beach','desert','nature','wildlife','winterlandscape']
# Uncomment to make a combined data and label file
#X=np.zeros((0,total_feat_size))
#y=np.zeros((0,1))
#for i in range(len(themes)):
#    data=np.load(themes[i]+".npy")
#    X=np.row_stack((X,data))
#    for j in range(len(data)):
#        y=np.vstack((y,themes_mapping[themes[i]]))
#    print data.shape
#
#np.save("theme",X)
#np.save("label",y)
#os.chdir("themes/")
X=np.load("theme.npy")
y=np.load("label.npy")
y=np.ravel(y)
y=y.astype(int)
# Uncomment to reduce dimensionality of the data. Run it only once.
#pca = PCA(n_components=0.8,whiten=True)
#pca.fit(X)
##train_X=pca.fit_transform(train_X)
#X=pca.transform(X)
#
#np.save('reduced_theme',X)
#plt.plot(pca.explained_variance_, linewidth=2)
#plt.axis('tight')
#plt.xlabel('n_components')
#plt.ylabel('explained_variance_')
#plt.savefig('componentsvsvariance.png',dpi=1200)
X=np.load("reduced_theme.npy")
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}


clf = svm.SVC(kernel='rbf',C=2)
#clf = svm.SVC(kernel='linear')
#clf = grid_search.GridSearchCV(clf, parameters)
#clf = RandomForestClassifier(n_estimators=200)
#clf= linear_model.BayesianRidge()
clf.fit(X_train, y_train)  
clf.fit(X_train, y_train) 
s1 = pickle.dumps(clf)
clf = pickle.loads(s1)


y_predict=clf.predict(X_test)
score=clf.score(X_test,y_test)
print "Score after applying SVM on the data is: ",score,"%"
print ""


cm=confusion_matrix(y_test,y_predict)


#label_names=['architecture','beach','desert','nature','wildlife','winterlandscape']
np.set_printoptions(precision=2)
print('Confusion matrix(logscale)')
#print(cm)
plt.figure()
log_cm=np.log10(cm+1)
clf_type="logreg"
plot_confusion_matrix(log_cm,clf_type,themes)
