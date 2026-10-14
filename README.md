______________________________________________________________________

<div align="center">

# RT-IceNet

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
[![Paper](http://img.shields.io/badge/paper.svg)](https://www.researchgate.net/publication/391500184_Rt-Icenet_A_Lightweight_Neural_Network_of_Sea_Ice_Segmentation_for_Onboard_Camera_Images)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://www.polarannual.com/sy)

</div>

## Description

#### Datasets

```bash
Links to download some of the datasets: https://drive.google.com/file/d/1ZCa_Eo0PKbRf74g87wxLaDcXbwCenQTV/view?usp=drive_link
The remainder will be published once the article has been released.
```

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/cqg99/RT-IceNet
cd RT-IceNet

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/cqg99/RT-IceNet
cd RT-IceNet

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

