# SCOD-Module
An implementation of Sketching Curvature for Efficient Out-of-Distribution Detection for Deep Neural Network (SCOD, [paper](https://arxiv.org/abs/2102.12567)) **regression tasks** in the form of a wrapper that can be applied to any PyTorch model.

![Figure](figures/scod_figure.png)


## Overview
Notable modifications to the [original repository](https://github.com/StanfordASL/SCOD) are made to support batched pre-processing and inference mechanics leveraging [functorch](https://pytorch.org/functorch/stable/). 
This results in **processing speeds upwards of 10x**, extending SCOD's use from a test-time OoD detection system to an effective train-time module.


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
In a nutshell, scod-regression produces two related measures of epistemic uncertainty. 
Several assumptions are currently built into the code-base that may be restrictive for some applications.
We refer the reader to the [original repository](https://github.com/StanfordASL/SCOD) should your use-case require a larger support of likelihood functions or prior distributions.

1. **Posterior Predictive Variance.** 
Our algorithms are designed for multi-dimensional regression tasks; we thereby assume a Gaussian likelihood function with unit-variance. 
Analytic solutions to the Gaussian-distributed posterior weight and predictive distributions are derived by imposing a Gaussian isotropic prior over the neural network weights.
The predictive variance quantifies the model's epistemic uncertainty.
2. **Local KL-Divergence.**
Under these guiding assumptions, we compute the expectation of the local KL-divergence in the output distribution over delta weight perturbations by integrating over the posterior distribution. 
This offers an uncertainty metric akin to the curvature of output distribution manifold under small weight perturbations - the curvature is proportional to the Fischer Information Matrix. 

```python 
import torch
from torch import nn
from scod_regression import SCOD

# Example SCOD config
config = {
  "output_dist": "NormalMeanParamLayer",
  "sketch": "SRFTSinglePassPCA",
  "num_eigs": 50,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_keys = ["states", "actions"]
dataloader_kwargs = {"batch_size": 128, "shuffle": True}

# Instantiate Q-network and wrap with SCOD
q_network = ContinuousMLPCritic().to(device)
scod = SCOD(q_network, config=config, device=device)

# Computing low-rank approximation of dataset Fischer
dataset = DatasetClass()
scod.process_dataset(dataset, input_keys, dataloader_kwargs=dataloader_kwargs)

# Add batch dimension to random sample
sample = dataset.__getitem__()
for k, v in sample.items():
    if torch.is_tensor(v):
        sample[k] = v.unsqueeze(0)

# Compute out-of-distribution quantities
q_values, post_pred_var, local_kl = scod(sample, input_keys, detach=False)

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
