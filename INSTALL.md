# Environment Installation

## Linux Node

Have your own linux node, could be you local linux or online cloud linux.

Make sure you can log on linux.

Preferably, the machine will have NVIDIA GPU. GPU is not required for every deep learning experiment. But is essential to your speed for tasks like large scale CNN and RNN.

AWS could service will have GPU instance,  other providers like paperspace, ali cloud are also very common choice.

This is a huge and evolving topic.

## Install & Using Anaconda

Go download the anaconda in the following address:

```
https://www.anaconda.com/download/#linux
```

They let you choose version, Currently I use python 3.6, fast.ai part I use python 2.7 then transfer to python 3.6 in part II.

The good news is, people prepared a [python 3.6 version for the course](https://github.com/chanansh/course.fast.ai-pyhon-3-keras-2)

So you can safely choose python 3. (Though the installation steps are the same)

First, copy the link address of the download url suitable for your machine. for example:

```
https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
```

Then, in your linux machine, run wget (paste your url here):

```
wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
bash Anaconda*.sh
```

```wget``` will download the file, takes about 10s or hours, depends on if your server is in China :-)

Then follow the installation steps, and defaultly they will install anaconda3 in your home folder.

This anaconda package is basic a giant folder. Inside the folder is an isolated python virtual environment.

The reason we involve virtual environment, has following benefits:

* Inside the environment, play as crazy as you can, won't **** up the entire system.

* Inside the environment, you can pip install every python resource without using root/sudo.

* Inside the environment, most packages for data science are installed(eg. numpy pandas scipy), and their versions are coordinated. So lib with dependencies won't conflict in versions.
