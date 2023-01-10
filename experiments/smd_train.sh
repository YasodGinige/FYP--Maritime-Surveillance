cd src
# train
CUDA_VISIBLE_DEVICES=0,1 python main.py tracking --exp_id bt_ps_ch_seven --dataset smd  --pre_hm --shift 0.05 --scale 0.05 --ltrb_amodal --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --num_epochs 20 --lr_step 5,15 --gpus 0,1 --load_model ../models/bt_ps_ch_five.pth --save_point 10 --max_frame_dist 10 --lr 8e-5  --batch_size 8 --clip_len 2 --trades
cd ..


