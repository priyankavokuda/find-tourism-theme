#We propose a framework that is able to automatically find the tourism theme of a given input country. We use supervised learning for this task with SVM and Random Forests algorithms.

#Dependencies

Python 2.7

scikit-learn: http://scikit-learn.org/stable/install.html

cartopy: conda install -c scitools cartopy

#Usage

Run preprocess_images_countries.py to pre-process images of countries in "Countries" folder.

Run preprocess_images_themes.py to pre-process images of themes "Themes" folder.

Run ml_svm.py to analyse the performance of dimensionality reduction and SVM on the data.

Run ml_random_forest.py to analyse the performance of Random forests on the data and also visualise tourism theme output on world map.

PCA, SVM and Random Forest are done using scikit-learn machine learning library.


Credits to: 
Pedregosa, Fabian, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel et al. "Scikit-learn: Machine learning in Python." Journal of Machine Learning Research 12, no. Oct (2011): 2825-2830.

and Cartopy python package designed to make drawing maps: http://scitools.org.uk/cartopy/docs/v0.13/index.html




