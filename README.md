# <h1 align="center">lasr</h1>

<p align="center">Lightening Automatic Speech Recognition
<p  align="center">An MIT License ASR research library, built on PyTorch-Lightning, for developing end-to-end ASR models.
  
***

<p  align="center"> 
     <a href="https://github.com/sooftware/KoSpeech/blob/latest/LICENSE">
          <img src="http://img.shields.io/badge/license-MIT-informational"> 
     </a>
     <a href="https://github.com/pytorch/pytorch">
          <img src="http://img.shields.io/badge/framework-PyTorch--Lightning-informational"> 
     </a>
     <a href="https://www.python.org/dev/peps/pep-0008/">
          <img src="http://img.shields.io/badge/codestyle-PEP--8-informational"> 
     </a>
  <a href="https://sooftware.github.io/lasr/">
          <img src="http://img.shields.io/badge/build-not tested-red">
     </a>
    
## Introduction
    
[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) is the lightweight [PyTorch](https://github.com/pytorch/pytorch) wrapper for high-performance AI research. PyTorch is extremely easy to use to build complex AI models. But once the research gets complicated and things like multi-GPU training, 16-bit precision and TPU training get mixed in, users are likely to introduce bugs. PyTorch Lightning solves exactly this problem. Lightning structures your PyTorch code so it can abstract the details of training. This makes AI research scalable and fast to iterate on. This project is an example that implements the asr project with PyTorch Lightning. In this project, I trained a model consisting of a conformer encoder + LSTM decoder with Joint CTC-Attention. The **lasr** means **l**ighthning **a**utomatic **s**peech **r**ecognition. I hope this could be a guideline for those who research speech recognition.  
  
## Installation
  
This project recommends Python 3.7 or higher.  
I recommend creating a new virtual environment for this project (using virtual env or conda).
  

### Prerequisites
  
* Numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* Pytorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.   
* librosa: `conda install -c conda-forge librosa` (Refer [here](https://github.com/librosa/librosa) for problem installing librosa)
* torchaudio: `pip install torchaudio==0.6.0` (Refer [here](https://github.com/pytorch/pytorch) for problem installing torchaudio)
* sentencepiece: `pip install sentencepiece` (Refer [here](https://github.com/google/sentencepiece) for problem installing sentencepiece)
* pytorch-lightning: `pip install pytorch-lightning` (Refer [here](https://github.com/PyTorchLightning/pytorch-lightning) for problem installing pytorch-lightning)
* hydra: `pip install hydra-core --upgrade` (Refer [here](https://github.com/facebookresearch/hydra) for problem installing hydra)
  
### Install from source
Currently we only support installation from source code using setuptools. Checkout the source code and run the   
following commands:  
```
pip install -e .
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
  
I use [Hydra](https://github.com/facebookresearch/hydra) to control all the training configurations. If you are not familiar with Hydra we recommend visiting the [Hydra website](https://hydra.cc/). Generally, Hydra is an open-source framework that simplifies the development of research applications by providing the ability to create a hierarchical configuration dynamically.
  
### Preparing LibriSpeech Dataset  
  
This procedure can take hours or more. Download the required data in the next step and create the manifest files and vocab files.  
  
- Command
  
```
$ ./dataset/prepare-libri.sh $DIR_TO_SAVE_DATA
```
  
### Training Speech Recognizer
  
You can simply train with LibriSpeech dataset like below:
```
$ python ./bin/main.py \
--data.dataset_path $DATASET_PATH \
--data.train_manifest_path $TRAIN_MANIFEST_PATH \
--data.valid_clean_manifest_path $VALID_CLEAN_MANIFEST_PATH \
--data.valid_other_manifest_path $VALID_OTHER_MANIFEST_PATH \
--data.test_clean_manifest_path $TEST_CLEAN_MANIFEST_PATH \
--data.test_other_manifest_path $TEST_OTHER_MANIFEST_PATH \
--data.vocab_path $VOCAB_PATH \
--data.vocab_model_path $VOCAB_MODEL_PATH
```
  
## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/lasr/issues) on Github.   
  
I appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  
  
### Code Style
I follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation. 
  
### License
This project is licensed under the MIT LICENSE - see the [LICENSE.md](https://github.com/sooftware/lasr/blob/master/LICENSE) file for details
  
## Author
  
* Soohwan Kim [@sooftware](https://github.com/sooftware)
* Contacts: sh951011@gmail.com
