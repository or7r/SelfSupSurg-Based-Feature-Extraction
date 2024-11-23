This repo is used for feature extraction for "Robust Surgical Phase Recognition From Annotation Efficient Supervision".
[![arXiv](https://img.shields.io/badge/arxiv-2207.00449-red)](https://arxiv.org/abs/2406.18481)

This repo was forked from https://github.com/CAMMA-public/SelfSupSurg, which corresponds to the paper
["Dissecting Self-Supervised Learning Methods for Surgical Computer Vision"](https://www.sciencedirect.com/science/article/pii/S1361841523001044).
We thank the authors for releasing their codes. 



# Get Started

## Datasets and imagenet checkpoints

### Cholec80

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

## Multibypass140

Details regarding downloading Multibypass140 can be found [here](https://github.com/CAMMA-public/MultiBypass140).

Next, we prepare the MultiBypass140 dataset as follows:
```bash
mkdir -p datasets/MultiBypass140/raw_labels/bern
cp -r "$MBy140/labels/bern/labels_by70_splits/labels/" datasets/MultiBypass140/raw_labels/bern/

mkdir -p datasets/MultiBypass140/raw_labels/strasbourg
cp -r "$MBy140/labels/strasbourg/labels_by70_splits/labels/" datasets/MultiBypass140/raw_labels/strasbourg/

python scripts/connect_labels.py

mkdir -p datasets/MultiBypass140/frames/
python scripts/order_frames.py --input_dir $MBy140/labels/bern/labels_by70_splits/frames --output_dir datasets/MultiBypass140/frames/train/ --label_file datasets/MultiBypass140/raw_labels/bern/train/1fps_100_0.pickle
python scripts/order_frames.py --input_dir $MBy140/labels/bern/labels_by70_splits/frames --output_dir datasets/MultiBypass140/frames/val/ --label_file datasets/MultiBypass140/raw_labels/bern/val/1fps_0.pickle
python scripts/order_frames.py --input_dir $MBy140/labels/bern/labels_by70_splits/frames --output_dir datasets/MultiBypass140/frames/test/ --label_file datasets/MultiBypass140/raw_labels/bern/test/1fps_0.pickle

python scripts/order_frames.py --input_dir $MBy140/labels/strasbourg/labels_by70_splits/frames --output_dir datasets/MultiBypass140/frames/train/ --label_file datasets/MultiBypass140/raw_labels/strasbourg/train/1fps_100_0.pickle
python scripts/order_frames.py --input_dir $MBy140/labels/strasbourg/labels_by70_splits/frames --output_dir datasets/MultiBypass140/frames/val/ --label_file datasets/MultiBypass140/raw_labels/strasbourg/val/1fps_0.pickle
python scripts/order_frames.py --input_dir $MBy140/labels/strasbourg/labels_by70_splits/frames --output_dir datasets/MultiBypass140/frames/test/ --label_file datasets/MultiBypass140/raw_labels/strasbourg/test/1fps_0.pickle


```


## Installation
You need to have a [Anaconda3](https://www.anaconda.com/products/individual#linux) installed for the setup. We developed the code on the Ubuntu 20.04, Python 3.8, PyTorch 1.7.1, and CUDA 10.2 using RTX6000 GPUs.
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

## Self-supervised trainining
```sh
# DINO 
(selfsupsurg)>cfg=hparams/cholec80/pre_training/cholec_to_cholec/series_01/h004.yaml
(selfsupsurg)>python main.py -hp $cfg -m self_supervised

(selfsupsurg)>mkdir -p runs/cholec80/pre_training/cholec_to_cholec/series_01/run_004/ \
               && cp model_final_checkpoint_dino_surg.torch runs/cholec80/pre_training/cholec_to_cholec/series_01/run_004/

(selfsupsurg)>cfg=hparams/MultiBypass140/pre_training/cholec_to_cholec/series_01/h004.yaml
(selfsupsurg)>python main.py -hp $cfg -m self_supervised

(selfsupsurg)>mkdir -p runs/MultiBypass140/pre_training/cholec_to_cholec/series_01/run_004/ \
               && cp model_final_checkpoint_dino_surg.torch runs/MultiBypass140/pre_training/cholec_to_cholec/series_01/run_004/
```

## Feature extraction
```sh
(selfsupsurg)>cfg=hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/phase/100/1/h004.yaml
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s train -f Trunk
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s val -f Trunk
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s test -f Trunk   


(selfsupsurg)>cfg=hparams/MultiBypass140/finetuning/cholec_to_cholec/series_01/test/phase/100/1/h004.yaml
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s train -f Trunk
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s val -f Trunk
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s test -f Trunk   

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

