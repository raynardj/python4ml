### Recursion Cellular Image Classification

##### Xiaochen(Raynard) Zhang

![rcic image](https://storage.googleapis.com/kaggle-competitions/kaggle/14420/logos/header.png?t=2019-06-26-02-51-18")

The codes and notebooks for [Recursion Cellular Image Classification](https://www.kaggle.com/c/recursion-cellular-image-classification/overview). It's my first kaggle medal(88/866).![bronze img](https://www.kaggle.com/static/images/medals/discussion/bronzel@1x.png)

The task is to classify siRNA image, with 6 channels stored in 6 png images

* We start our training with [this starter kernel](https://www.kaggle.com/tanlikesmath/rcic-fastai-starter). It has efficientnet, resnet, densenet, data preprocess pipeline ready. 

#### Why we should read and study data carefully?
These facts pretty much define how reckless we are this time. It's a miracle we even got the medal...
* We discover there are 2 sites 3 days after we join. Before that we only use half the data.
* We discover the plate leak only almost too late. Even if we apply the enforcing 227-hot encoding by the end. I know the fact "within each plate, there is only 1 siRNA of its class in the plate" only after the competition is over.
* We are acknowledge of "control" pictures after the competition is over.

#### Ensemble
* We ensemble the models, here are an evolution of our ensemble solutions. [v1, my first ensemble](rcic-fastai-ensemble_v1.ipynb),
[v2, I mark the public LB score](rcic-fastai-ensemble_v2.ipynb),[v3](rcic-fastai-ensemble_v3.ipynb),
[v5, final version of ensemble](rcic-fastai-ensemble_v5-b5.ipynb) with plate leak as an enforcing mask, also visualize how the plate leak works. Gradually we fade out other models, and use various versions of "EfficientNet-b5"
* The ensemble improved our public LB score at least 0.05

#### Plate Leak
* We found the [plate leak](https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/102905) 2 days before the competition closed. I kicked myself for the slopiness of not wondering "discussion" often enough.
* [This notebook](plate_leak.ipynb) expored the plate leak and allocate/save the 4 groups of siRNA.
* In the [final ensemble notebook](rcic-fastai-ensemble_v5-b5.ipynb) we apply the leak info to the model prediction output. This leak helped out Public LB score improved at least 0.1.
* On the eve of the closing (UTC+8:00), I experimented [learning from conv activations](learn_from_activiation.ipynb). But the time is too brief. Nothing prevails.

#### Thanks
* Thank my wife for the support and suffering while I'm kaggling. Thank [Nelson](https://github.com/lognat0704) for luring me into this competition, only by your sharing of the burden made so much of work possible in such short notice of time.