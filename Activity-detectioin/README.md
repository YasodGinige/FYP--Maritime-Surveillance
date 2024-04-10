# You Only Watch Once (YOWO)

PyTorch implementation of the article "[You Only Watch Once: A Unified CNN Architecture for Real-Time Spatiotemporal Action Localization](https://github.com/wei-tim/YOWO/blob/master/examples/YOWO_updated.pdf)". The repositry contains code for real-time spatiotemporal action localization with PyTorch on AVA, UCF101-24 and JHMDB datasets!

**Updated paper** can be accessed via [**YOWO_updated.pdf**](https://github.com/wei-tim/YOWO/blob/master/examples/YOWO_updated.pdf)

**NOTE:**
We adapted the YOWO algorithm in the thermal domain to detect suspicious activities like human trafficking and swimming (in the deep water). The algorithm contains two main sections. It captures spatial information and spatiotemporal information separately, and it combines them to do the final classification using channel fusion and an attention module. When reorganizing, we changed the last layer of the model to detect two target activities, swimming and possible human trafficking footage, in our dataset. Then, we retrained the model to detect these two activities in the thermal domain using our dataset.

To set up the algorithm, follow the below-mentioned steps.

## Installation
```bash
git clone https://github.com/YasodGinige/FYP--Maritime-Surveillance.git
cd YOWO
```

### Datasets

* Our    : Dowload from [here](https://drive.google.com/drive/folders/1MRWGZO6Qmw952WIPxi5eSfsiR52R5VLw)
* AVA	   : Download from [here](https://github.com/cvdfoundation/ava-dataset)

After downloading the dataset, rearrange it according to the [these](https://github.com/facebookresearch/SlowFast/blob/master/slowfast/datasets/DATASET.md) instructions.

### Download backbone pretrained weights

* Darknet-19 weights can be downloaded via:
```bash
wget http://pjreddie.com/media/files/yolo.weights
```
* ResNeXt ve ResNet pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing).

### Pretrained YOWO models

Pretrained models for AVA dataset can be downloaded from [here](https://drive.google.com/drive/folders/1g-jTfxCV9_uNFr61pjo4VxNfgDlbWLlb?usp=sharing).

All materials (annotations and pretrained models) are also available in Baiduyun Disk:
[here](https://pan.baidu.com/s/1yaOYqzcEx96z9gAkOhMnvQ) with password 95mm

## Running the code

* All training configurations are given in [ava.yaml](https://github.com/wei-tim/YOWO/blob/master/cfg/ava.yaml), [ucf24.yaml](https://github.com/wei-tim/YOWO/blob/master/cfg/ucf24.yaml) and [jhmdb.yaml](https://github.com/wei-tim/YOWO/blob/master/cfg/jhmdb.yaml) files.
* AVA training:
```bash
python main.py --cfg cfg/ava.yaml
```

## Validating the model

* For AVA dataset, after each epoch, validation is performed and frame-mAP score is provided.

```bash
python evaluation/Object-Detection-Metrics/pascalvoc.py --gtfolder PATH-TO-GROUNDTRUTHS-FOLDER --detfolder PATH-TO-DETECTIONS-FOLDER

```

* For video_mAP, set the pretrained model in the correct yaml file and run:
```bash
python video_mAP.py --cfg cfg/ucf24.yaml
```

### Citation
If you use this code or pre-trained models, please cite the following:

```bibtex
@InProceedings{kopuklu2019yowo,
title={You Only Watch Once: A Unified CNN Architecture for Real-Time Spatiotemporal Action Localization},
author={K{\"o}p{\"u}kl{\"u}, Okan and Wei, Xiangyu and Rigoll, Gerhard},
journal={arXiv preprint arXiv:1911.06644},
year={2019}
}
```

