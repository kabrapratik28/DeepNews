# DeepNews [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Generates headline out of a given text of data.

DeepNews is a high-level news generating tool, written in Python and capable of running on top of either [Keras](https://github.com/fchollet/keras), [TensorFlow](https://github.com/tensorflow/tensorflow) or [Theano](https://github.com/Theano/Theano). It was developed for media orgnizations or writters where they can quickly come up with headline that is short and information conveying. 

- - - -

## Getting started


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

### Deep News

`Import` model

```python
from deepnews import *
```

`Train` model

```python
#Fill code to train new dataset
```

`Test` model

```python
#Fill code to test new dataset
```

Using `Pretrained` model
```python
#Fill code on how to use pretrained model
```

`Evaluate` Model
```python
#Fill code on how to use evaluate results
```

`Text Preprocessing`
```python
#Fill code on how to use text preprocessing
```

In the examples folder of the repository, you will find more examples.

- - - -

### Dataset 

* [Data (zip)](https://drive.google.com/open?id=0Bw35nAjs4lJbemlpLW13U2x5RHM)  
* [Fire Dataset Website](http://fire.irsi.res.in/)
* [Seed list](https://github.com/kabrapratik28/DeepNews/blob/master/data/seed_list.txt)

### Word2Vec (Hindi Language)

## To check our Word2Vec results and click below :
[![Word2Vec Link Image](./data/label.png)](./word2vec/readme.md)

## Dataset Statistics

`Length of Article histogram`


![Length of Article Histogram](./data/Article_Length_Bar_Graph.JPG)


`Length of Headline histogram`


![Length of Headline Histogram](./data/Headline_Length_Bar_Graph.JPG)


### FIRE Dataset stats

| features                                                                                              | values              |
|-------------------------------------------------------------------------------------------------------|---------------------|
| no of articles                                                                                        | 2,97,965            |
| no of tokens                                                                                          | 85,940,081 (85.94M) |
| no of unique tokens in articles                                                                       | 3,88,449            |
| no of unique tokens in headlines                                                                      | 58,448              |
| avg length of article                                                                                 | 272                 |
| avg length of headline                                                                                | 7                   |
| size of dataset                                                                                       | 1.06GB              |
| avg. of (ratio len(article)/len(headline)) (Behind 43 words of description,  headline contain 1 word) | 43                  |


### Crawled Dataset stats

| features                                                                                              | values              |
|-------------------------------------------------------------------------------------------------------|---------------------|
| no of articles                                                                                        | 5,95,847            |
| no of tokens                                                                                          | 20,92,32,922 (209M) |
| no of unique tokens in articles                                                                       | 10,26,083           |
| no of unique tokens in headlines                                                                      | 1,24,965            |
| avg length of article                                                                                 | 316                 |
| avg length of headline                                                                                | 11                  |
| size of dataset                                                                                       | 3.70GB              |
| avg. of (ratio len(article)/len(headline)) (Behind 43 words of description,  headline contain 1 word) | 34                  |

### Number of Crawled Articles per source

| News Website      | Number of Articles | URL                                                |
|-------------------|--------------------|----------------------------------------------------|
| Aaj Tak           | 92765              | http://www.aajtak.intoday.in                       |
| ABP News          | 13654              | http://www.abpnews.abplive.in                      |
| Amar Ujala        | 181                | http://www.amarujala.com                           |
| BBC Hindi         | 28861              | http://bbc.com/hindi                               |
| Deshbandhu        | 3174               | http://deshbandhu.co.in                            |
| Economic Times    | 993                | http://hindi.economictimes.indiatimes.com          |
| Jagran            | 73290              | http://www.jagran.com                              |
| Navbharat Times   | 10329              | http://www.navbharattimes.indiatimes.com           |
| NDTV              | 92942              | http://www.khabar.ndtv.com/news/                   |
| News18            | 38833              | http://www.news18.com                              |
| Patrika           | 68288              | http://www.patrika.com                             |
| Punjab Kesari     | 15494              | http://www.punjabkesari.in                         |
| Rajasthan Patrika | 89038              | http://www.rajasthanpatrika.patrika.com            |
| Zee News          | 10463              | http://www.zeenews.india.com/hindi                 |
