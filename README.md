# Find tourist theme of a country

This project was done as part of Masters curriculum which involved finding own and useful real life use-case to apply machine learning methods.

The use case was to find the tourism theme of a country.

The input to the system were images collected from  social networking site. Training images were obtained by searching using hashtags #{theme_name}.

Fixed themes such as architecture, beach, desert, wildlife, winter-landscape were selected and used as labels. The problem of finding tourist theme was reduced to supervised image classification problem with test image as a image of a country.

Random forest was used as classification method which gave a modest performance of ~50 %. The performance can be improved by removing noisy images from the training data.

# Dependencies

Python 2.7

scikit-learn: http://scikit-learn.org/stable/install.html

cartopy: conda install -c scitools cartopy

# Usage

Run preprocess_images_countries.py to pre-process images of countries in "Countries" folder.

Run preprocess_images_themes.py to pre-process images of themes "Themes" folder.

Run ml_svm.py to analyse the performance of dimensionality reduction and SVM on the data.

Run ml_random_forest.py to analyse the performance of Random forests on the data and also visualise tourism theme output on world map.

PCA, SVM and Random Forest are done using scikit-learn machine learning library.

# Results 

![](https://github.com/priyankavokuda/priyankavokuda.github.io/blob/master/images/tourist_theme.gif)


Credits to: 
Pedregosa, Fabian, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel et al. "Scikit-learn: Machine learning in Python." Journal of Machine Learning Research 12, no. Oct (2011): 2825-2830.

and Cartopy python package designed to make drawing maps: http://scitools.org.uk/cartopy/docs/v0.13/index.html




