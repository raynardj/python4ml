# Python for Machine Learning

> Python notebooks and experiments

## Python Tutorial
python tutorial, staging for deep/machine learning course

[1️⃣ Lesson 1](1.1_lesson1.ipynb)
[2️⃣ Lesson 2](1.2_lesson2.ipynb)
[3️⃣ Lesson 3](1.3_lesson3.ipynb)
[4️⃣ Lesson 4](1.4_lesson4.ipynb)
[5️⃣ lesson 5](1.5_lesosn5.ipynb)

## Deep Learning

* [Mighty💪 MLP(Multi-Layer Perceptron)](3.1.1_mighty_mlp.ipynb)

* [Recommender: Deep Collaborative Filtering💰💰💰](3.1.3_recommender_system.ipynb)

* [How Machine Learns To Read📚](3.1.2_how_machine_learns_to_read.ipynb)

* [From Keras to 🔥PyTorch🔥](3.5_from_keras_to_pytorch.ipynb)

#### Originally designed to teach Alex python, thank you for your constant learning enthusiasm


#### Also created for [coderbunker](http://www.coderbunker.com/co-learning) deep learning talk sessions

This course is the basic deep learning course that follows closely on Jeremy Howard's fantasic /free /life-changing [fast.ai course](course.fast.ai). 

The most of notebooks are just trails we left behind passing on their awesomeness.

[Checklist before we start and a reading list](https://raynardj.github.io/python4ml/docs/pre_checklist)

#### Experiments

* [The coord conv from the uber paper](https://raynardj.github.io/python4ml/docs/coord_conv)

<img src="docs/coord_conv.png" style="border-radius:7px;" alt="coord conv pic" width="200px"/>

* Ray's [toolbox for deep learning](https://raynardj.github.io/ray/)

<img src="https://raynardj.github.io/ray/img/Match.jpg" style="border-radius:7px;" alt="match box pic" width="200px"/>

* Fast [KMeans by batch with GPU](https://raynardj.github.io/ray/docs/kmean_torch), how to train a 60 minutes kmeans in 3 seconds, with [example here](https://raynardj.github.io/ray/docs/gowalla_preprocess)

<img src="https://raynardj.github.io/ray/img/accelerate.jpg" style="border-radius:7px;" alt="kemans accelerated" width="200px"/>

* A PyTorch [training wrapper](https://raynardj.github.io/ray/docs/matchbox) to simplify tracking

### Environment Installation

Follow the [installation instructions](https://raynardj.github.io/python4ml/docs/INSTALL)

Run the jupyter notebook on anaconda3 environment

Usual pre-requisites for the learning. 

```
python 3.6

numpy == '1.14.3'
pandas == '0.23.0'
tensorflow == '1.8.0'
keras == '2.2.0'
```

Other versions of above library will probably work.

Assuming your anaconda3 is at ```~/anaconda3/```

If you don't have any of these, try the following format in the command line:
```
~/anaconda3/bin/pip install keras==2.2.0
```
If you are on Mac:
```
~/anaconda3/bin/pip install torch torchvision
```
To install PyTorch. For other system, You'll have to visit their [homepage](https://pytorch.org/) to copy/paste the right command to install.

### Ray

In some lines of code you might see

```python
from ray import matchbox
```

or
```
from ray.lprint import lprint
```

Yeah, I listened to my friends any named my tool belt ray.

Ray is a github repository [here](https://github.com/raynardj/ray)

### Contact Us

If you want to be a contributor:

mail: ```raynard@rasenn.com```

wechat: 417848506 remark:"python4ml"

