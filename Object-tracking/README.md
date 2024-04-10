# Maritime Object Tracking

We addapted the TraDeS algorithm to detect and track maritime objects in the thermal domain.

[**Track to Detect and Segment: An Online Multi-Object Tracker**]&#40;http://arxiv.org/abs/2004.01177&#41;,            )
[**Track to Detect and Segment: An Online Multi-Object Tracker**](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_Track_To_Detect_and_Segment_An_Online_Multi-Object_Tracker_CVPR_2021_paper.pdf)  

## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Run Demo
Before run the demo, first download our trained models:
[CrowdHuman model](https://drive.google.com/file/d/1pljgwSecg50OhCTc2yCEhEBY3AwvPFlp/view?usp=sharing) (2D tracking),

Then, put the models in `./models/` and `cd ./src/`. **The demo result will be saved as a video in `results/`.**

### *2D Tracking Demo*

**Demo for a video clip which we randomly selected from YouTube**: Run the demo (using the [CrowdHuman model](https://drive.google.com/file/d/1pljgwSecg50OhCTc2yCEhEBY3AwvPFlp/view?usp=sharing)):

    python demo.py tracking --load_model ../models/crowdhuman.pth --num_class 1 --demo ../videos/street_2d.mp4 --pre_hm --ltrb_amodal --pre_thresh 0.5 --track_thresh 0.5 --inference --clip_len 2 --trades --save_video --resize_video --input_h 480 --input_w 864

##  Training and Evaluation

Please refer to [Data.md](readme/DATA.md) for dataset preparation.

**Train on MOT17 halftrain set:** Place the [pretrained model](https://drive.google.com/file/d/1pljgwSecg50OhCTc2yCEhEBY3AwvPFlp/view?usp=sharing) in $TraDeS_ROOT/models/ and run:

    sh experiments/mot17_train.sh

**Test on MOT17 validation set:** Place the [MOT model](https://drive.google.com/file/d/18DQi6LqFuO7_2QObvZSNK2y_F8yXT17p/view?usp=sharing) in $TraDeS_ROOT/models/ and run:

    sh experiments/mot17_test.sh

## *Train on Static Images*
We follow [CenterTrack](https://arxiv.org/pdf/2004.01177.pdf) which uses CrowdHuman to pretrain 2D object tracking model. Only the training set is used.

    sh experiments/crowdhuman.sh

The trained model is available at [CrowdHuman model](https://drive.google.com/file/d/1pljgwSecg50OhCTc2yCEhEBY3AwvPFlp/view?usp=sharing).


## Citation
If you find it useful in your research, please consider citing our paper as follows:

    @inproceedings{Wu2021TraDeS,
    title={Track to Detect and Segment: An Online Multi-Object Tracker},
    author={Wu, Jialian and Cao, Jiale and Song, Liangchen and Wang, Yu and Yang, Ming and Yuan, Junsong},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021}}

## Acknowledgment
Many thanks to [CenterTrack](https://github.com/xingyizhou/CenterTrack) authors for their great framework!
