# Environment Installation

## Linux Node

Have your own linux node, could be you local linux or online cloud linux.

Make sure you can log on linux.

Preferably, the machine will have NVIDIA GPU. GPU is not required for every deep learning experiment. But is essential to your speed for tasks like large scale CNN and RNN.

AWS could service will have GPU instance,  other providers like paperspace, ali cloud are also very common choice.

This is a huge and evolving topic.

## Install And Using Anaconda

### Find Anaconda Package

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

### Download

Then, in your linux machine, run wget (paste your url here):

```
wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
```

```wget``` will download the file, takes about 10s or hours, depends on if your server is in China :-)

### Install Anaconda

```
bash Anaconda*.sh
```

Then follow the installation steps, and defaultly they will install anaconda3 in your home folder.


### Virtual Environment

This anaconda package is basic a giant folder. Inside the folder is an isolated python virtual environment.

The reason we involve virtual environment, has following benefits:

* Inside the environment, play as crazy as you can, you won't mess up the entire system.

* Inside the environment, you can pip install every python resource without using root/sudo.

* Inside the environment, most packages for data science are installed(eg. numpy pandas scipy jupyter), and their versions are coordinated. So lib with dependencies won't conflict in versions.

### Activate Anaconda Environment
Activate environment, assuming you install anaconda3 in your home directory:

```
source ~/anaconda3/bin/activate
```

From now on , you can see a ```(root)``` in front of command prompt, you are in the environment

Type ```python```, enter. You can see the message shows the current python version.

Try ```import pandas as pd```, enter, no error message, you can see the pandas is already installed!

Personally, I prefer adding the following to the ~/.bashrc file, so next time I'm log on, I'will just enter ```ana3``` to activate environment instantly.

```
alias ana3='source ~/anaconda3/bin/activate'
```

### Jupyter Notebook

Generate a configuration file 

```
jupyter notebook --generate-config
```

Get the hashed password, ```python```, enter.

Run:
```
from notebook.auth import passwd; passwd()
```
Type in a password and remember. You can see python returned a hashed string, something like:
```
'sha1:451104bc6c2b:6f02f7c3184387974d6a15a495e4ad05d18cxx0f'
```
Copy this string.


### Edit the config file
```
vim ~/.jupyter/jupyter_notebook_config.py
# or 
# nano ~/.jupyter/jupyter_notebook_config.py
```

Edit the following, uncomment(remove #) the following configures, and set values.

Set the port,ip and password (the hashed string you copied)
```
## The port the notebook server will listen on.
c.NotebookApp.port = 8888

...

## The IP address the notebook server will listen on.
c.NotebookApp.ip = '*'

...

## Whether to open in a browser after starting. The specific browser used is
#  platform dependent and determined by the python standard library `webbrowser`
#  module, unless it is overridden using the --browser (NotebookApp.browser)
#  configuration option.
c.NotebookApp.open_browser = False

...

#  The string should be of the form type:salt:hashed-password.
c.NotebookApp.password = u'sha1:451104bc6c2b:6f02f7c3184387974d6a15a495e4ad05d18cxx0f'
```

Save the configuration.

### Run jupyter notebook

Have a tmux running,(optional)
```
tmux new -s jn
```

Run 
```
jupyter notebook
```

If you are using jupyter.
ctrl+'B', 'D' to detach the session. ```tmux attach -t jn``` to return to the jupyter session.

### Use the notebook

Open your local browser, open the url:

```
http://yourIpOrHost:/8888
```
Type in the password you set
