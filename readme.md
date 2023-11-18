# CoADS: Cross attention based dual-space graph network for survival prediction of lung cancer using whole slide images

&copy; The manuscript of CoADS has been accepted for publication by CMPB.
https://www.sciencedirect.com/science/article/pii/S0169260723002249

## Introduction

This is a PyTorch implementation of the CoADS, which is proposed for the overall survival (OS) prediction of the lung cancer. 

## Usage

### Preparation

Install Pytorch and other dependencies.

- Linux
- Python 3.9
- Pytorch 1.11.0
- CUDA 11.3
- NVIDIA GPU

```shell
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

### Run

- Extract patch features using ResNet50

```shell
cd models
CUDA_VISIBLE_DEVICES=0 python extract_features.py --batch_size 128
```

- Train the survival model

```shell
cd models
CUDA_VISIBLE_DEVICES=0 python train.py --model CoADS --mode graph
```

