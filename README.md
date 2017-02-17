# DeepNews [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Generates headline out of a given text of data.

DeepNews is a high-level news generating tool, written in Python and capable of running on top of either [Keras](https://github.com/fchollet/keras), [TensorFlow](https://github.com/tensorflow/tensorflow) or [Theano](https://github.com/Theano/Theano). It was developed for a media orgnizations or writters where they can quickly come up with headline that is short and information conveying. 

- - - -

## Getting started

- - - -

### Installing

DeepNews in written on top of Python and Keras, ThensorFlow and Theano. 

>Installing Python:

* [Anaconda](https://www.continuum.io/downloads) - Comes with prebuild libraries like Pandas, Numpy, Scipy, etc. (Recommended) 
* [Official Python website](https://www.python.org/downloads/)

>Installing Keras

* ``` sudo pip install keras ``` 
* Windows Based System can follow this steps [Stackoverflow](http://stackoverflow.com/questions/34097988/how-do-i-install-keras-and-theano-in-anaconda-python-2-7-on-windows) 

>Installing TensorFlow

* [Official site](https://www.tensorflow.org/install/)  

>Amazon AWS 
(All libraries are installed in the AMI image)
* G2 or P2 (GPU) based instances
* [Amazon Machine Image AMI](https://aws.amazon.com/marketplace/pp/B06VSPXKDX)
* GPU configuration are enabled (by default)

Neural networks are computations heavy, GPU configuration is recommended. 

- - - -

