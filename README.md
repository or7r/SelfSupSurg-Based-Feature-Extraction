This repo is forked from https://github.com/CAMMA-public/SelfSupSurg and used for feature extraction for "Robust Surgical Phase Recognition From Annotation Efficient Supervision"

[![arXiv](https://img.shields.io/badge/arxiv-2207.00449-red)](https://arxiv.org/abs/2406.18481)


# Get Started

## Datasets and imagenet checkpoints
Follow the steps for cholec80 dataset preparation and setting up imagenet checkpoints:

```bash
# 1. Cholec80 phase and tool labels for different splits
> git clone https://github.com/CAMMA-public/SelfSupSurg
> SelfSupSurg=$(pwd)/SelfSupSurg
> cd $SelfSupSurg/datasets/cholec80
> wget https://s3.unistra.fr/camma_public/github/selfsupsurg/ch80_labels.zip
> unzip -q ch80_labels.zip && rm ch80_labels.zip
# 2. Cholec80 frames:  
# a) Download cholec80 dataset: 
#      - Fill this google form: https://docs.google.com/forms/d/1GwZFM3-GhEduBs1d5QzbfFksKmS1OqXZAz8keYi-wKI  
#       (the link is also available on the CAMMA website: http://camma.u-strasbg.fr/datasets)
# b) Copy the videos in datasets/cholec80/videos 
# Extract frames using the following script (you need OpenCV and numpy)
> cd $SelfSupSurg
> python utils/extract_frames_ch80.py
# 3. Download Imagenet fully supervised and self-supervised weights
> cd $SelfSupSurg/checkpoints/defaults/resnet_50
> wget https://s3.unistra.fr/camma_public/github/selfsupsurg/imagenet_ckpts.zip
> unzip -q imagenet_ckpts.zip && rm imagenet_ckpts.zip

```
- Directory structure should look as follows.
```shell
$SelSupSurg/
└── datasets/cholec80/
    ├── frames/
        ├── train/
            └── video01/
            └── video02/
            ...
        ├── val/
            └── video41/
            └── video42/
            ...
        ├── test/
            └── video49/
            └── video50/
            ...
    ├── labels/
        ├── train/
            └── 1fps_12p5_0.pickle
            └── 1fps_12p5_1.pickle
            ...
        ├── val/
            └── 1fps.pickle
            └── 3fps.pickle
            ...
        ├── test/
            └── 1fps.pickle
            └── 3fps.pickle
            ...        
    └── classweights/
        ├── train/
            └── 1fps_12p5_0.pickle
            └── 1fps_12p5_1.pickle
                ...
    ...
    └── checkpoints/defaults/resnet_50/
        └── resnet50-19c8e357.pth
        └── moco_v2_800ep_pretrain.pth.tar
        └── simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1.torch
        └── swav_in1k_rn50_800ep_swav_8node_resnet_27_07_20.a0a6b676.torch
        └── dino_resnet50_pretrain.pth
```


## Installation
You need to have a [Anaconda3](https://www.anaconda.com/products/individual#linux) installed for the setup. We developed the code on the Ubuntu 20.04, Python 3.8, PyTorch 1.7.1, and CUDA 10.2 using V100 GPU.
```sh
> cd $SelfSupSurg
> conda create -n selfsupsurg python=3.8 && conda activate selfsupsurg
# install dependencies 
(selfsupsurg)>conda install -y pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.2 -c pytorch 
(selfsupsurg)>pip install opencv-python
(selfsupsurg)>pip install openpyxl==3.0.7
(selfsupsurg)>pip install pandas==1.3.2
(selfsupsurg)>pip install scikit-learn
(selfsupsurg)>pip install easydict
(selfsupsurg)>pip install apex -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/py38_cu102_pyt171/download.html
(selfsupsurg)>cd $SelfSupSurg/ext_libs
(selfsupsurg)>git clone https://github.com/facebookresearch/ClassyVision.git && cd ClassyVision
(selfsupsurg)>git checkout 659d7f788c941a8c0d08dd74e198b66bd8afa7f5 && pip install -e .
(selfsupsurg)>cd ../ && git clone --recursive https://github.com/facebookresearch/vissl.git && cd ./vissl/
(selfsupsurg)>git checkout 65f2c8d0efdd675c68a0dfb110aef87b7bb27a2b
(selfsupsurg)>pip install --progress-bar off -r requirements.txt
(selfsupsurg)>pip install -e .[dev] && cd $SelfSupSurg
(selfsupsurg)>cp -r ./vissl/vissl/* $SelfSupSurg/ext_libs/vissl/vissl/
```
#### Modify `$SelfSupSurg/ext_libs/vissl/configs/config/dataset_catalog.json` by appending the following key/value pair to the end of the dictionary
```json
"surgery_datasets": {
    "train": ["<img_path>", "<lbl_path>"],
    "val": ["<img_path>", "<lbl_path>"],
    "test": ["<img_path>", "<lbl_path>"]
}
```


## Pre-training
Run the folllowing code for the pre-training of MoCo v2, SimCLR, SwAV, and DINO methods on the Cholec80 dataset with 4 GPUS.
```sh
# MoCo v2
(selfsupsurg)>cfg=hparams/cholec80/pre_training/cholec_to_cholec/series_01/h001.yaml
(selfsupsurg)>python main.py -hp $cfg -m self_supervised
# SimCLR
(selfsupsurg)>cfg=hparams/cholec80/pre_training/cholec_to_cholec/series_01/h002.yaml
(selfsupsurg)>python main.py -hp $cfg -m self_supervised
# SwAV
(selfsupsurg)>cfg=hparams/cholec80/pre_training/cholec_to_cholec/series_01/h003.yaml
(selfsupsurg)>python main.py -hp $cfg -m self_supervised
# DINO 
(selfsupsurg)>cfg=hparams/cholec80/pre_training/cholec_to_cholec/series_01/h004.yaml
(selfsupsurg)>python main.py -hp $cfg -m self_supervised
```

## Model Weights for the **pre-training** experiments

|   Model      |  Model Weights |
| :----------: | :-----:   |
| [MoCo V2](configs/config/hparams/cholec80/pre_training/cholec_to_cholec/series_01/h001.yaml)| [download](https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_moco_v2_surg.torch) |
| [SimCLR](configs/config/hparams/cholec80/pre_training/cholec_to_cholec/series_01/h002.yaml)| [download](https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_simclr_surg.torch) |
| [SwAV](configs/config/hparams/cholec80/pre_training/cholec_to_cholec/series_01/h003.yaml)| [download](https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_swav_surg.torch) |
| [DINO](configs/config/hparams/cholec80/pre_training/cholec_to_cholec/series_01/h004.yaml)| [download](https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_dino_surg.torch) |


## Downstream finetuning
First perform pre-training using the above scripts or download the [pre-trained weights](#model-weights-for-the-pre-training-experiments) and copy them into the appropriate directories, shown below
```sh
# MoCo v2
(selfsupsurg)>mkdir -p runs/cholec80/pre_training/cholec_to_cholec/series_01/run_001/ \
               && cp model_final_checkpoint_moco_v2_surg.torch runs/cholec80/pre_training/cholec_to_cholec/series_01/run_001/
# SimCLR
(selfsupsurg)>mkdir -p runs/cholec80/pre_training/cholec_to_cholec/series_01/run_002/ \
               && cp model_final_checkpoint_simclr_surg.torch runs/cholec80/pre_training/cholec_to_cholec/series_01/run_002/
# SwAV
(selfsupsurg)>mkdir -p runs/cholec80/pre_training/cholec_to_cholec/series_01/run_003/ \
               && cp model_final_checkpoint_swav_surg.torch runs/cholec80/pre_training/cholec_to_cholec/series_01/run_003/
# DINO 
(selfsupsurg)>mkdir -p runs/cholec80/pre_training/cholec_to_cholec/series_01/run_004/ \
               && cp model_final_checkpoint_dino_surg.torch runs/cholec80/pre_training/cholec_to_cholec/series_01/run_004/
```
### 1. Surgical phase recognition (Linear Finetuning)
The config files for the surgical phase recognition **linear finetuning** experiments are in [cholec80 pre-training init](configs/config/hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/phase) and [imagenet init](configs/config/hparams/cholec80/finetuning/imagenet_to_cholec/series_01/test/phase). The config files are organized as follows:
<details>
<summary>config_files</summary>

```sh
# config files for the proposed pre-training init from cholec80 are oraganized as follows:
├── cholec_to_cholec/series_01/test/phase
    ├── 100 #(100 % of cholec 80)
    │   └── 0 #(split 0)
    │       ├── h001.yaml # MoCo V2 Surg
    │       ├── h002.yaml # SimCLR Surg
    │       ├── h003.yaml # SwAV Surg
    │       └── h004.yaml # DINO Surg
    ├── 12.5 #(12.5 % of cholec 80 dataset)
    │   ├── 0 #(split 0)
    │   │   ├── h001.yaml # MoCo V2 Surg
    │   │   ├── h002.yaml # SimCLR Surg
    │   │   ├── h003.yaml # SwAV Surg
    │   │   └── h004.yaml # DINO Surg
    │   ├── 1 #(split 1)
    │   │   ├── h001.yaml # MoCo V2 Surg
    │   │   ├── h002.yaml # SimCLR Surg
    │   │   ├── h003.yaml # SwAV Surg
    │   │   └── h004.yaml # DINO Surg
    │   ├── 2 #(split 2)
    │   │   ├── h001.yaml # MoCo V2 Surg
    │   │   ├── h002.yaml # SimCLR Surg
    │   │   ├── h003.yaml # SwAV Surg
    │   │   └── h004.yaml # DINO Surg
    └── 25 #(25 % of cholec 80 dataset)
        ├── 0 #(split 0)
        │   ├── h001.yaml # MoCo V2 Surg
        │   ├── h002.yaml # SimCLR Surg
        │   ├── h003.yaml # SwAV Surg
        │   └── h004.yaml # DINO Surg
        ├── 1 #(split 1)
        │   ├── h001.yaml # MoCo V2 Surg
        │   ├── h002.yaml # SimCLR Surg
        │   ├── h003.yaml # SwAV Surg
        │   └── h004.yaml # DINO Surg
        ├── 2 #(split 2)
        │   ├── h001.yaml # MoCo V2 Surg
        │   ├── h002.yaml # SimCLR Surg
        │   ├── h003.yaml # SwAV Surg
        │   └── h004.yaml # DINO Surg
# config files for the baselines imagenet to cholec80 are oraganized as follows:
├── imagenet_to_cholec/series_01/test/phase
    ├── 100 #(100 % of cholec 80)
    │   └── 0 #(split 0)
    │       ├── h001.yaml # Fully-supervised imagenet
    │       ├── h002.yaml # MoCo V2 imagenet
    │       ├── h003.yaml # SimCLR imagenet
    │       ├── h004.yaml # SwAV imagenet
    │       └── h005.yaml # DINO imagenet
    ├── 12.5 #(12.5 % of cholec 80 dataset)
    │   ├── 0 #(split 0)
    │   │   ├── h001.yaml # Fully-supervised imagenet
    │   │   ├── h002.yaml # MoCo V2 imagenet
    │   │   ├── h003.yaml # SimCLR  imagenet
    │   │   ├── h004.yaml # SwAV  imagenet
    │   │   └── h005.yaml # DINO imagenet
    │   ├── 1 #(split 1)
    │   │   ├── h001.yaml # Fully-supervised imagenet 
    │   │   ├── h002.yaml # MoCo V2 imagenet
    │   │   ├── h003.yaml # SimCLR imagenet
    │   │   ├── h004.yaml # SwAV imagenet
    │   │   └── h005.yaml # DINO imagenet
    │   ├── 2 #(split 2)
    │   │   ├── h001.yaml # Fully-supervised imagenet 
    │   │   ├── h002.yaml # MoCo V2 imagenet
    │   │   ├── h003.yaml # SimCLR imagenet
    │   │   ├── h004.yaml # SwAV imagenet
    │   │   └── h005.yaml # DINO imagenet
    └── 25 #(25 % of cholec 80 dataset)
        ├── 0 #(split 0)
        │   ├── h001.yaml # Fully-supervised imagenet
        │   ├── h002.yaml # MoCo V2 imagenet
        │   ├── h003.yaml # SimCLR imagenet
        │   ├── h004.yaml # SwAV imagenet
        │   └── h005.yaml # DINO imagenet
        ├── 1 #(split 1)
        │   ├── h001.yaml # Fully-supervised imagenet
        │   ├── h002.yaml # MoCo V2 imagenet
        │   ├── h003.yaml # SimCLR imagenet
        │   ├── h004.yaml # SwAV imagenet
        │   ├── h005.yaml # DINO imagenet
        ├── 2 #(split 2)
        │   ├── h001.yaml # Fully-supervised imagenet
        │   ├── h002.yaml # MoCo V2 imagenet
        │   ├── h003.yaml # SimCLR imagenet
        │   ├── h004.yaml # SwAV imagenet
        │   └── h005.yaml # DINO imagenet
```
</details>

Examples commands for surgical phase linear fine-tuning. It uses 4 GPUS for the training
```sh
# Example 1, run the following command for linear fine-tuning, initialized with MoCO V2 weights 
# on 25% of cholec80 data (split 0).
(selfsupsurg)>cfg=hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/phase/25/0/h001.yaml
(selfsupsurg)>python main.py -hp $cfg -m supervised

# Example 2, run the following command for linear fine-tuning, initialized with SimCLR weights 
# on 12.5% of cholec80 data (split 1).
(selfsupsurg)>cfg=hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/phase/12.5/1/h002.yaml
(selfsupsurg)>python main.py -hp $cfg -m supervised

# Example 3, run the following command for linear fine-tuning, initialized with 
# imagenet MoCO v2 weights on 12.5% of cholec80 data (split 2).
(selfsupsurg)>cfg=hparams/cholec80/finetuning/imagenet_to_cholec/series_01/test/phase/12.5/2/h002.yaml
(selfsupsurg)>python main.py -hp $cfg -m supervised
```

### 2. Surgical phase recognition (TCN Finetuning)

The config files for the surgical phase recognition **TCN finetuning** experiments are in [cholec80 pre-training init](configs/config/hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/phase_tcn) and [imagenet init](configs/config/hparams/cholec80/finetuning/imagenet_to_cholec/series_01/test/phase_tcn). The config files are organized as follows:
<details>
<summary>config_files</summary>

    
```sh
# config files for the proposed pre-training init from cholec80 are oraganized as follows:
├── cholec_to_cholec/series_01/test/phase_tcn
    ├── 100 #(100 % of cholec 80)
    │   └── 0 #(split 0)
    │       ├── h001.yaml # MoCo V2 Surg
    │       ├── h002.yaml # SimCLR Surg
    │       ├── h003.yaml # SwAV Surg
    │       └── h004.yaml # DINO Surg
    ├── 12.5 #(12.5 % of cholec 80 dataset)
    │   ├── 0 #(split 0)
    │   │   ├── h001.yaml # MoCo V2 Surg
    │   │   ├── h002.yaml # SimCLR Surg
    │   │   ├── h003.yaml # SwAV Surg
    │   │   └── h004.yaml # DINO Surg
    │   ├── 1 #(split 1)
    │   │   ├── h001.yaml # MoCo V2 Surg
    │   │   ├── h002.yaml # SimCLR Surg
    │   │   ├── h003.yaml # SwAV Surg
    │   │   └── h004.yaml # DINO Surg
    │   ├── 2 #(split 2)
    │   │   ├── h001.yaml # MoCo V2 Surg
    │   │   ├── h002.yaml # SimCLR Surg
    │   │   ├── h003.yaml # SwAV Surg
    │   │   └── h004.yaml # DINO Surg
    └── 25 #(25 % of cholec 80 dataset)
        ├── 0 #(split 0)
        │   ├── h001.yaml # MoCo V2 Surg
        │   ├── h002.yaml # SimCLR Surg
        │   ├── h003.yaml # SwAV Surg
        │   └── h004.yaml # DINO Surg
        ├── 1 #(split 1)
        │   ├── h001.yaml # MoCo V2 Surg
        │   ├── h002.yaml # SimCLR Surg
        │   ├── h003.yaml # SwAV Surg
        │   └── h004.yaml # DINO Surg
        ├── 2 #(split 2)
        │   ├── h001.yaml # MoCo V2 Surg
        │   ├── h002.yaml # SimCLR Surg
        │   ├── h003.yaml # SwAV Surg
        │   └── h004.yaml # DINO Surg
# config files for the baselines imagenet to cholec80 are oraganized as follows:
├── imagenet_to_cholec/series_01/test/phase_tcn
    ├── 100 #(100 % of cholec 80)
    │   └── 0 #(split 0)
    │       ├── h001.yaml # Fully-supervised imagenet
    │       ├── h002.yaml # MoCo V2 imagenet
    │       ├── h003.yaml # SimCLR imagenet
    │       ├── h004.yaml # SwAV imagenet
    │       └── h005.yaml # DINO imagenet
    ├── 12.5 #(12.5 % of cholec 80 dataset)
    │   ├── 0 #(split 0)
    │   │   ├── h001.yaml # Fully-supervised imagenet
    │   │   ├── h002.yaml # MoCo V2 imagenet
    │   │   ├── h003.yaml # SimCLR  imagenet
    │   │   ├── h004.yaml # SwAV  imagenet
    │   │   └── h005.yaml # DINO imagenet
    │   ├── 1 #(split 1)
    │   │   ├── h001.yaml # Fully-supervised imagenet 
    │   │   ├── h002.yaml # MoCo V2 imagenet
    │   │   ├── h003.yaml # SimCLR imagenet
    │   │   ├── h004.yaml # SwAV imagenet
    │   │   └── h005.yaml # DINO imagenet
    │   ├── 2 #(split 2)
    │   │   ├── h001.yaml # Fully-supervised imagenet 
    │   │   ├── h002.yaml # MoCo V2 imagenet
    │   │   ├── h003.yaml # SimCLR imagenet
    │   │   ├── h004.yaml # SwAV imagenet
    │   │   └── h005.yaml # DINO imagenet
    └── 25 #(25 % of cholec 80 dataset)
        ├── 0 #(split 0)
        │   ├── h001.yaml # Fully-supervised imagenet
        │   ├── h002.yaml # MoCo V2 imagenet
        │   ├── h003.yaml # SimCLR imagenet
        │   ├── h004.yaml # SwAV imagenet
        │   └── h005.yaml # DINO imagenet
        ├── 1 #(split 1)
        │   ├── h001.yaml # Fully-supervised imagenet
        │   ├── h002.yaml # MoCo V2 imagenet
        │   ├── h003.yaml # SimCLR imagenet
        │   ├── h004.yaml # SwAV imagenet
        │   ├── h005.yaml # DINO imagenet
        ├── 2 #(split 2)
        │   ├── h001.yaml # Fully-supervised imagenet
        │   ├── h002.yaml # MoCo V2 imagenet
        │   ├── h003.yaml # SimCLR imagenet
        │   ├── h004.yaml # SwAV imagenet
        │   └── h005.yaml # DINO imagenet
```
</details>

Examples commands for TCN fine-tuning. We first extract the features for the `train`, `val` and `test` set and then perform the TCN fine-tuning
```sh
# Example 1, run the following command for TCN fine-tuning, initialized with MoCO V2 weights 
# on 25% of cholec80 data (split 0).
# 1) feature extraction for the train, val and test set
(selfsupsurg)>cfg=hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/phase/25/0/h001.yaml
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s train -f Trunk
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s val -f Trunk
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s test -f Trunk                            
# 2) TCN fine-tuning        
(selfsupsurg)>cfg=hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/phase_tcn/25/0/h001.yaml
(selfsupsurg)>python main_ft_phase_tcn.py -hp $cfg -t test

# Example 2, run the following command for TCN fine-tuning, initialized with SimCLR weights 
# on 12.5% of cholec80 data (split 1).
# 1) feature extraction for the train, val and test set
(selfsupsurg)>cfg=hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/phase/12.5/1/h002.yaml
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s train -f Trunk
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s val -f Trunk
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s test -f Trunk                            
# 2) TCN fine-tuning        
(selfsupsurg)>cfg=hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/phase_tcn/12.5/1/h002.yaml
(selfsupsurg)>python main_ft_phase_tcn.py -hp $cfg -t test

# Example 3, run the following command for TCN fine-tuning, initialized with imagenet MoCO v2 weights 
# on 12.5% of cholec80 data (split 2).
# 1) feature extraction for the train, val and test set
(selfsupsurg)>cfg=hparams/cholec80/finetuning/imagenet_to_cholec/series_01/test/phase/12.5/2/h002.yaml
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s train -f Trunk
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s val -f Trunk
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s test -f Trunk                            
# 2) TCN fine-tuning        
(selfsupsurg)>cfg=hparams/cholec80/finetuning/imagenet_to_cholec/series_01/test/phase_tcn/12.5/2/h002.yaml
(selfsupsurg)>python main_ft_phase_tcn.py -hp $cfg -t test
```

### 3. Surgical tool recognition

The config files for the surgical tool recognition experiments are in [cholec80 pre-training init](configs/config/hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/tools) and [imagenet init](configs/config/hparams/cholec80/finetuning/imagenet_to_cholec/series_01/test/tools). The config files are organized as follows:
<details>
<summary>config_files</summary>

    
```sh
# config files for the proposed pre-training init from cholec80 are oraganized as follows:
├── cholec_to_cholec/series_01/test/tools
    ├── 100 #(100 % of cholec 80)
    │   └── 0 #(split 0)
    │       ├── h001.yaml # MoCo V2 Surg
    │       ├── h002.yaml # SimCLR Surg
    │       ├── h003.yaml # SwAV Surg
    │       └── h004.yaml # DINO Surg
    ├── 12.5 #(12.5 % of cholec 80 dataset)
    │   ├── 0 #(split 0)
    │   │   ├── h001.yaml # MoCo V2 Surg
    │   │   ├── h002.yaml # SimCLR Surg
    │   │   ├── h003.yaml # SwAV Surg
    │   │   └── h004.yaml # DINO Surg
    │   ├── 1 #(split 1)
    │   │   ├── h001.yaml # MoCo V2 Surg
    │   │   ├── h002.yaml # SimCLR Surg
    │   │   ├── h003.yaml # SwAV Surg
    │   │   └── h004.yaml # DINO Surg
    │   ├── 2 #(split 2)
    │   │   ├── h001.yaml # MoCo V2 Surg
    │   │   ├── h002.yaml # SimCLR Surg
    │   │   ├── h003.yaml # SwAV Surg
    │   │   └── h004.yaml # DINO Surg
    └── 25 #(25 % of cholec 80 dataset)
        ├── 0 #(split 0)
        │   ├── h001.yaml # MoCo V2 Surg
        │   ├── h002.yaml # SimCLR Surg
        │   ├── h003.yaml # SwAV Surg
        │   └── h004.yaml # DINO Surg
        ├── 1 #(split 1)
        │   ├── h001.yaml # MoCo V2 Surg
        │   ├── h002.yaml # SimCLR Surg
        │   ├── h003.yaml # SwAV Surg
        │   └── h004.yaml # DINO Surg
        ├── 2 #(split 2)
        │   ├── h001.yaml # MoCo V2 Surg
        │   ├── h002.yaml # SimCLR Surg
        │   ├── h003.yaml # SwAV Surg
        │   └── h004.yaml # DINO Surg
# config files for the baselines imagenet to cholec80 are oraganized as follows:
├── imagenet_to_cholec/series_01/test/tools
    ├── 100 #(100 % of cholec 80)
    │   └── 0 #(split 0)
    │       ├── h001.yaml # Fully-supervised imagenet
    │       ├── h002.yaml # MoCo V2 imagenet
    │       ├── h003.yaml # SimCLR imagenet
    │       ├── h004.yaml # SwAV imagenet
    │       └── h005.yaml # DINO imagenet
    ├── 12.5 #(12.5 % of cholec 80 dataset)
    │   ├── 0 #(split 0)
    │   │   ├── h001.yaml # Fully-supervised imagenet
    │   │   ├── h002.yaml # MoCo V2 imagenet
    │   │   ├── h003.yaml # SimCLR  imagenet
    │   │   ├── h004.yaml # SwAV  imagenet
    │   │   └── h005.yaml # DINO imagenet
    │   ├── 1 #(split 1)
    │   │   ├── h001.yaml # Fully-supervised imagenet 
    │   │   ├── h002.yaml # MoCo V2 imagenet
    │   │   ├── h003.yaml # SimCLR imagenet
    │   │   ├── h004.yaml # SwAV imagenet
    │   │   └── h005.yaml # DINO imagenet
    │   ├── 2 #(split 2)
    │   │   ├── h001.yaml # Fully-supervised imagenet 
    │   │   ├── h002.yaml # MoCo V2 imagenet
    │   │   ├── h003.yaml # SimCLR imagenet
    │   │   ├── h004.yaml # SwAV imagenet
    │   │   └── h005.yaml # DINO imagenet
    └── 25 #(25 % of cholec 80 dataset)
        ├── 0 #(split 0)
        │   ├── h001.yaml # Fully-supervised imagenet
        │   ├── h002.yaml # MoCo V2 imagenet
        │   ├── h003.yaml # SimCLR imagenet
        │   ├── h004.yaml # SwAV imagenet
        │   └── h005.yaml # DINO imagenet
        ├── 1 #(split 1)
        │   ├── h001.yaml # Fully-supervised imagenet
        │   ├── h002.yaml # MoCo V2 imagenet
        │   ├── h003.yaml # SimCLR imagenet
        │   ├── h004.yaml # SwAV imagenet
        │   ├── h005.yaml # DINO imagenet
        ├── 2 #(split 2)
        │   ├── h001.yaml # Fully-supervised imagenet
        │   ├── h002.yaml # MoCo V2 imagenet
        │   ├── h003.yaml # SimCLR imagenet
        │   ├── h004.yaml # SwAV imagenet
        │   └── h005.yaml # DINO imagenet
```
</details>    

Examples commands for surgical tool recognition **linear fine-tuning**. It uses 4 GPUS for the training
```sh
# Example 1, run the following command for linear fine-tuning, initialized with MoCO V2 weights 
# on 25% of cholec80 data (split 0).
(selfsupsurg)>cfg=hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/tools/25/0/h001.yaml
(selfsupsurg)>python main.py -hp $cfg -m supervised

# Example 2, run the following command for linear fine-tuning, initialized with SimCLR weights 
# on 12.5% of cholec80 data (split 1).
(selfsupsurg)>cfg=hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/tools/12.5/1/h002.yaml
(selfsupsurg)>python main.py -hp $cfg -m supervised

# Example 3, run the following command for linear fine-tuning, initialized with 
# imagenet MoCO v2 weights on 12.5% of cholec80 data (split 2).
(selfsupsurg)>cfg=hparams/cholec80/finetuning/imagenet_to_cholec/series_01/test/tools/12.5/2/h002.yaml
(selfsupsurg)>python main.py -hp $cfg -m supervised
```
### 4. Evaluation

Example command to evaluate all the experiments and collect the results
```sh

# computes evaluation metrics for all the experiments and saves results in the runs/metrics_<phase/tool>.csv
(selfsupsurg)>python utils/generate_test_results.py
```


## Citation
```bibtex
@article{ramesh2023dissecting,
  title={Dissecting self-supervised learning methods for surgical computer vision},
  author={Ramesh, Sanat and Srivastav, Vinkle and Alapatt, Deepak and Yu, Tong and Murali, Aditya and Sestini, Luca and Nwoye, Chinedu Innocent and Hamoud, Idris and Sharma, Saurav and Fleurentin, Antoine and others},
  journal={Medical Image Analysis},
  pages={102844},
  year={2023},
  publisher={Elsevier}
}
```




### References
The project uses [VISSL](https://github.com/facebookresearch/vissl). We thank the authors of VISSL for releasing the library. If you use VISSL, consider citing it using the following BibTeX entry.
```bibtex
@misc{goyal2021vissl,
  author =       {Priya Goyal and Quentin Duval and Jeremy Reizenstein and Matthew Leavitt and Min Xu and
                  Benjamin Lefaudeux and Mannat Singh and Vinicius Reis and Mathilde Caron and Piotr Bojanowski and
                  Armand Joulin and Ishan Misra},
  title =        {VISSL},
  howpublished = {\url{https://github.com/facebookresearch/vissl}},
  year =         {2021}
}
```
The project also leverages following research works. We thank the authors for releasing their codes.
- [TeCNO](https://github.com/tobiascz/TeCNO)

## License
This code, models, and datasets are available for non-commercial scientific research purposes as defined in the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party codes are subject to their respective licenses.
