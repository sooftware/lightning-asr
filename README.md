# <h1 align="center">Lightning ASR</h1>

<div align="center">

**Modular and extensible speech recognition library leveraging [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) and [hydra](https://github.com/facebookresearch/hydra)**

  
---

<p align="center">
  <a href="https://github.com/sooftware/lightning-asr#introduction">What is Lightning ASR</a> •
  <a href="https://github.com/sooftware/lightning-asr#installation">Installation</a> •
  <a href="https://github.com/sooftware/lightning-asr#get-started">Get Started</a> •
  <a href="https://sooftware.github.io/lightning-asr/">Docs</a> •
  <a href="https://www.codefactor.io/repository/github/sooftware/lightning-asr">Codefactor</a> •
  <a href="https://github.com/sooftware/lightning-asr/blob/main/LICENSE">License</a>
</p>

---
</div>
    
## Introduction
    
[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) is the lightweight [PyTorch](https://github.com/pytorch/pytorch) wrapper for high-performance AI research. PyTorch is extremely easy to use to build complex AI models. But once the research gets complicated and things like multi-GPU training, 16-bit precision and TPU training get mixed in, users are likely to introduce bugs. PyTorch Lightning solves exactly this problem. Lightning structures your PyTorch code so it can abstract the details of training. This makes AI research scalable and fast to iterate on.   
  
This project is an example that implements the asr project with PyTorch Lightning. In this project, I trained a model consisting of a conformer encoder + LSTM decoder with Joint CTC-Attention. I hope this could be a guideline for those who research speech recognition.  
  
## Installation
  
This project recommends Python 3.7 or higher.  
I recommend creating a new virtual environment for this project (using virtual env or conda).
  

### Prerequisites
  
* numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* pytorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.   
* librosa: `conda install -c conda-forge librosa` (Refer [here](https://github.com/librosa/librosa) for problem installing librosa)
* torchaudio: `pip install torchaudio==0.6.0` (Refer [here](https://github.com/pytorch/pytorch) for problem installing torchaudio)
* sentencepiece: `pip install sentencepiece` (Refer [here](https://github.com/google/sentencepiece) for problem installing sentencepiece)
* pytorch-lightning: `pip install pytorch-lightning` (Refer [here](https://github.com/PyTorchLightning/pytorch-lightning) for problem installing pytorch-lightning)
* hydra: `pip install hydra-core --upgrade` (Refer [here](https://github.com/facebookresearch/hydra) for problem installing hydra)
  
### Install from source
Currently I only support installation from source code using setuptools. Checkout the source code and run the   
following commands:  
```
$ pip install -e .
$ ./setup.sh
```
  
### Install Apex (for 16-bit training) 
  
For faster training install NVIDIA's apex library:
  
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex

# ------------------------
# OPTIONAL: on your cluster you might need to load CUDA 10 or 9
# depending on how you installed PyTorch

# see available modules
module avail

# load correct CUDA before install
module load cuda-10.0
# ------------------------

# make sure you've loaded a cuda version > 4.0 and < 7.0
module load gcc-6.1.0

$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
  
## Get Started
  
I use [Hydra](https://github.com/facebookresearch/hydra) to control all the training configurations. If you are not familiar with Hydra I recommend visiting the [Hydra website](https://hydra.cc/). Generally, Hydra is an open-source framework that simplifies the development of research applications by providing the ability to create a hierarchical configuration dynamically.
  
### Download LibriSpeech dataset
  
You have to download [LibriSpeech](https://www.openslr.org/12) dataset that contains 1000h English speech corpus. But you can download simply by `dataset_download` option. If this option is True, download the dataset and start training. If you already have a dataset, you can set option `dataset_download` to False and specify `dataset_path`.
  
### Training Speech Recognizer
  
You can simply train with LibriSpeech dataset like below:  
  
- Example1: Train the `conformer-lstm` model with `filter-bank` features on GPU.
  
```
$ python ./bin/main.py \
    data=default \
    dataset_download=True \
    audio=fbank \
    model=conformer_lstm \
    lr_scheduler=reduce_lr_on_plateau \
    trainer=gpu
```

- Example2: Train the `conformer-lstm` model with `mel-spectrogram` features On TPU:
  
```
$ python ./bin/main.py \
    data=default \
    dataset_download=True \
    audio=melspectrogram \
    model=conformer_lstm \
    lr_scheduler=reduce_lr_on_plateau \
    trainer=tpu
```
 
## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/lightning-asr/issues) on Github.   
  
I appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  
  
### Code Style
I follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation. 
  
### License
This project is licensed under the MIT LICENSE - see the [LICENSE.md](https://github.com/sooftware/lightning-asr/blob/master/LICENSE) file for details
  
## Author
  
* Soohwan Kim [@sooftware](https://github.com/sooftware)
* Contacts: sh951011@gmail.com
