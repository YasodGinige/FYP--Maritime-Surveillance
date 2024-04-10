# You Only Watch Once (YOWO)

PyTorch implementation of the article "[You Only Watch Once: A Unified CNN Architecture for Real-Time Spatiotemporal Action Localization](https://github.com/wei-tim/YOWO/blob/master/examples/YOWO_updated.pdf)". The repositry contains code for real-time spatiotemporal action localization with PyTorch on AVA, UCF101-24 and JHMDB datasets!

**Updated paper** can be accessed via [**YOWO_updated.pdf**](https://github.com/wei-tim/YOWO/blob/master/examples/YOWO_updated.pdf)

**NOTE:**
We adapted the YOWO algorithm in the thermal domain to detect suspicious activities like human trafficking and swimming (in the deep water). The algorithm contains two main sections. It captures spatial information and spatiotemporal information separately, and it combines them to do the final classification using channel fusion and an attention module. When reorganizing, we changed the last layer of the model to detect two target activities, swimming and possible human trafficking footage, in our dataset. Then, we retrained the model to detect these two activities in the thermal domain using our dataset.

To set up the algorithm, follow the below-mentioned steps.

## Installation
```bash
git clone https://github.com/wei-tim/YOWO.git
cd YOWO
```

### Datasets

* AVA	   : Download from [here](https://github.com/cvdfoundation/ava-dataset)
* UCF101-24: Download from [here](https://drive.google.com/file/d/1o2l6nYhd-0DDXGP-IPReBP4y1ffVmGSE/view?usp=sharing)
* J-HMDB-21: Download from [here](http://jhmdb.is.tue.mpg.de/challenge/JHMDB/datasets)

Use instructions [here](https://github.com/facebookresearch/SlowFast/blob/master/slowfast/datasets/DATASET.md) for the preperation of AVA dataset.

Modify the paths in ucf24.data and jhmdb21.data under cfg directory accordingly.
Download the dataset annotations from [here](https://www.dropbox.com/sh/16jv2kwzom1pmlt/AABL3cFWDfG5MuH9PwnjSJf0a?dl=0).

### Download backbone pretrained weights

* Darknet-19 weights can be downloaded via:
```bash
wget http://pjreddie.com/media/files/yolo.weights
```

* ResNeXt ve ResNet pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing).

***NOTE:*** For JHMDB-21 trainings, HMDB-51 finetuned pretrained models should be used! (e.g. "resnext-101-kinetics-hmdb51_split1.pth").

* For resource efficient 3D CNN architectures (ShuffleNet, ShuffleNetv2, MobileNet, MobileNetv2), pretrained models can be downloaded from [here](https://github.com/okankop/Efficient-3DCNNs).

### Pretrained YOWO models

Pretrained models for UCF101-24 and J-HMDB-21 datasets can be downloaded from [here](https://www.dropbox.com/sh/16jv2kwzom1pmlt/AABL3cFWDfG5MuH9PwnjSJf0a?dl=0).

Pretrained models for AVA dataset can be downloaded from [here](https://drive.google.com/drive/folders/1g-jTfxCV9_uNFr61pjo4VxNfgDlbWLlb?usp=sharing).

All materials (annotations and pretrained models) are also available in Baiduyun Disk:
[here](https://pan.baidu.com/s/1yaOYqzcEx96z9gAkOhMnvQ) with password 95mm

## Running the code

* All training configurations are given in [ava.yaml](https://github.com/wei-tim/YOWO/blob/master/cfg/ava.yaml), [ucf24.yaml](https://github.com/wei-tim/YOWO/blob/master/cfg/ucf24.yaml) and [jhmdb.yaml](https://github.com/wei-tim/YOWO/blob/master/cfg/jhmdb.yaml) files.
* AVA training:
```bash
python main.py --cfg cfg/ava.yaml
```
* UCF101-24 training:
```bash
python main.py --cfg cfg/ucf24.yaml
```
* J-HMDB-21 training:
```bash
python main.py --cfg cfg/jhmdb.yaml
```

## Validating the model

* For AVA dataset, after each epoch, validation is performed and frame-mAP score is provided.

* For UCF101-24 and J-HMDB-21 datasets, after each validation, frame detections is recorded under 'jhmdb_detections' or 'ucf_detections'. From [here](https://www.dropbox.com/sh/16jv2kwzom1pmlt/AABL3cFWDfG5MuH9PwnjSJf0a?dl=0), 'groundtruths_jhmdb.zip' and 'groundtruths_jhmdb.zip' should be downloaded and extracted to "evaluation/Object-Detection-Metrics". Then, run the following command to calculate frame_mAP.

```bash
python evaluation/Object-Detection-Metrics/pascalvoc.py --gtfolder PATH-TO-GROUNDTRUTHS-FOLDER --detfolder PATH-TO-DETECTIONS-FOLDER

```

* For video_mAP, set the pretrained model in the correct yaml file and run:
```bash
python video_mAP.py --cfg cfg/ucf24.yaml
```

## Running on a text video

* You can run AVA pretrained model on any test video with the following code:
```bash
python test_video_ava.py --cfg cfg/ava.yaml
```

***UPDATEs:*** 
* YOWO is extended for AVA dataset. 
* Old repo is deprecated and moved to [YOWO_deprecated](https://github.com/wei-tim/YOWO/tree/yowo_deprecated) branch. 

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

### Acknowledgements
We thank [Hang Xiao](https://github.com/marvis) for releasing [pytorch_yolo2](https://github.com/marvis/pytorch-yolo2) codebase, which we build our work on top. 
