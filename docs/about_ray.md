# Ray's Coding
# 张晓晨主要的一些代码

* Please tolerate my naive codes years ago. 请务必容忍我多年前粗糙的代码习惯
* They are not all perfect, but they sure show the learning curve, and hope it can bring a sense of authenticity. 这是长时间以来的一些记录，有些并不完美， 但这体现了学习曲线， 并希望能给到一种真实。
* Some notebook has heavy Latex comments or images, and won't render on github page. It has to be opened in jupyter notebook to work properly. 有的是带了很多latex或图的笔记， 所以github可能无法直接render, 需要在jupyter中打开

### AI/ Machine Learning/ Deep Learning
* Kaggle competitions:
    * I go by username raynardj, sometime I change name, sometime I join team, but always the Robert D Jr avatar.learning
    * [Recursion Cellular Image Classification](https://www.kaggle.com/c/recursion-cellular-image-classification), bronze medal, public leaderboard [(88/866)](https://www.kaggle.com/c/recursion-cellular-image-classification/leaderboard), competition summary and codes [here](https://github.com/raynardj/python4ml/tree/master/experiments/rcic)
    * [Generative Dog Images](https://www.kaggle.com/c/generative-dog-images), public leaderboard [(143/924)](https://www.kaggle.com/c/generative-dog-images/leaderboard)
    * [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection), public leaderboard [(524/2943)](https://www.kaggle.com/c/aptos2019-blindness-detection/leaderboard)
    
* Experiments, papers to code. For the purpose of true representation, these are all my original coding, if any code I borrow heavily from other resource, I won't list it here.
    * [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247) from uber AI lab, code and doc [here](https://raynardj.github.io/python4ml/docs/coord_conv)
    * [A Discriminative Feature Learning Approach for Deep Face Recognition](https://ydwen.github.io/papers/WenECCV16.pdf)'s [deployment and visualization](https://github.com/raynardj/python4ml/blob/master/papers/centerloss_in_pytorch.ipynb)
    * [Language detection](https://github.com/raynardj/python4ml/blob/master/experiments/language_detection.ipynb), the entire experiment pipeline
    * Intuitive experiment/ understanding of vanilla [optimizers: SGD, Adagrad, Adam](https://github.com/raynardj/python4ml/blob/master/experiments/fun_with_optimizer_and_more_fun_with_image_reconstruction.ipynb)
    * An [experiment](https://github.com/raynardj/python4ml/blob/master/experiments/poi_reco.ipynb) of point of interest recommendation.
    * [Adversarial Attack](https://github.com/raynardj/python4ml/blob/master/experiments/Adversarial_Attack.ipynb)
    * [WGAN with deep CNN](https://github.com/raynardj/python4ml/blob/master/experiments/gan/wgan_with_deep_conv.ipynb)
    * [Code](https://github.com/raynardj/python4ml/blob/master/experiments/style_transfer_perceptual_loss.py) for paper [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1603.08155).
    * [Language Modeling](https://github.com/raynardj/python4ml/blob/master/experiments/books/language_modeling_v4_plato.ipynb)

* Machine Learning Related Repos
    * [ForgeBox](https://github.com/raynardj/forge/tree/master/forgebox), my personal machine learning/ deep learning tool box. pip installable.
    * [Forge](https://github.com/raynardj/forge), a web framework we can use to track AI experiments, in code and on UI frondend.
    * My experiments on [object detection, yolov3 with DenseNet backbone](https://github.com/raynardj/obj_detection)
    * [Sequence to sequence from scratch](https://github.com/raynardj/seqtwoseq)
    * A universal [PyTorch API docker](https://github.com/raynardj/pytorch_api)
    * [KMeans on batch](https://raynardj.github.io/ray/docs/kmean_torch), a lighting fast way to run Kmeans with 100+ centeroids. It can be accelerated on GPU.

* None-AI personal projects, I'm not looking for this line of job, the following are just showcase of engineering proficiency
    * [Read sense](http://www.rasenn.com/), an online english literature library I built up in 3 days. Simple CRUD on book, author, user and permissions. Hooked up to google reCAPTCHA. [git repository here](https://github.com/raynardj/readsense)
    * [A wedding management tool](https://github.com/raynardj/wedding) for my own wedding. Able to manage guests/tables, and randomly check our photos with our own love quotes.
    * [A raspberry pi liquid flow LED effect according to device movement](https://github.com/raynardj/rpi/blob/master/nerotears.py)
    