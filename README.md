# SCOD-Module
An implementation of Sketching Curvature for Efficient Out-of-Distribution Detection for Deep Neural Network (SCOD, [paper](https://arxiv.org/abs/2102.12567)) **regression tasks** in the form of a wrapper that can be applied to any PyTorch model.

![Figure](figures/scod_figure.png)


## Overview
Notable modifications to the [original repository](https://github.com/StanfordASL/SCOD) are made to support batched pre-processing and inference mechanics leveraging [functorch](https://pytorch.org/functorch/stable/). 
This results in processing speeds upwards of 10x, extending SCOD's use from a test-time OoD detection system to an effective train-time module.


## Setup

### System Requirements
Tested on Ubuntu 16.04, 18.04 and macOS Monterey with Python 3.7.

### Installation
This package can be installed via pip as shown below.

```bash
git clone https://github.com/agiachris/scod-module.git
cd scod-module && pip install .
```


## Usage
Description of usage.
```python 
import torch
import scod

# TODO: show basic usage
``` 


## Citation
This repository has an MIT [License](https://github.com/agiachris/scod-regression/blob/main/LICENSE). If you find this package helpful, please consider citing:
```
@inproceedings{sharma2021sketching,
  title={Sketching curvature for efficient out-of-distribution detection for deep neural networks},
  author={Sharma, Apoorva and Azizan, Navid and Pavone, Marco},
  booktitle={Uncertainty in Artificial Intelligence},
  pages={1958--1967},
  year={2021},
  organization={PMLR}
}
```
