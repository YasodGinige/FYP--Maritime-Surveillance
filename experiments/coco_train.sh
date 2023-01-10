cd src
# train
CUDA_VISIBLE_DEVICES=0,1 python main.py tracking --exp_id bt_ps_ch_twelve --dataset coco --dataset_version 17fulltrain --pre_hm --shift 0.05 --scale 0.05 --ltrb_amodal --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --num_epochs 140 --lr_step 90,120 --gpus 0,1 --load_model ../models/bt_ps_ch_five.pth --save_point 70,110,130 --max_frame_dist 10 --lr 8e-5  --batch_size 16 --clip_len 2 --trades --transferlearn 0
cd ..