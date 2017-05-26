## Datasets
A collection of datasets convenient for

#### cats and dogs
[https://www.kaggle.com/c/dogs-vs-cats/data](https://www.kaggle.com/c/dogs-vs-cats/data) ?
#### mnist
60 000 images containing handwritten digits in the range 0 through 9. Originally created (on this form) by [Yann LeCun](http://yann.lecun.com/exdb/mnist/)

###### Usage:
`from tfwrapper.datasets import mnist`
###### Parameters:
* size: Number of images in the dataset. Defaults to the full set
* image_size: Size of the individual images. Defaults to (28, 28)

###### Example:
An example can be seen in [custom_cnn.py](https://github.com/epigramai/tfwrapper/blob/master/examples/custom_cnn.py)
#### cifar10
50 000 small images (32x32) from the 10 classes {airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck}. Originates from [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
###### Usage:
`from tfwrapper.datasets import cifar10`
###### Parameters:
* size: Number of images in the dataset. Defaults to the full set

###### Example:
An example can be seen in [shallow_cnn.py](https://github.com/epigramai/tfwrapper/blob/master/examples/shallow_cnn.py)
#### cifar100
50 000 small images (32x32) from 100 classes. Originates from [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
###### Usage:
`from tfwrapper.datasets import cifar100`
###### Parameters:
* size: Number of images in the dataset. Defaults to the full set

###### Example:
An example can be seen in [shallow_cnn.py](https://github.com/epigramai/tfwrapper/blob/master/examples/squeezenet.py)
#### flowers
1360 images of flowers from 17 different species. Used in [A Visual Vocabulary for Flower Classification](http://www.robots.ox.ac.uk/~vgg/publications/papers/nilsback06.pdf)
###### Usage:
`from tfwrapper.datasets import flowers`
###### Parameters:
* size: Number of images in the dataset. Defaults to the full set

###### Example:
An example can be seen in [kmeans.py](https://github.com/epigramai/tfwrapper/blob/master/examples/kmeans.py)
#### wine
178 different wines, with attributes ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']. From [http://archive.ics.uci.edu/ml/datasets/Wine](http://archive.ics.uci.edu/ml/datasets/Wine)
###### Usage:
`from tfwrapper.datasets import wine`
###### Parameters:
* y_index: Describes which column should be used as a y-value. Either an int or a string. Defaults to None, which means a dataset with no y is returned
* include_headers: returns both a dataset and the list of column names

###### Example:
An example can be seen in [kmeans.py](https://github.com/epigramai/tfwrapper/blob/master/examples/linearregression.py)