cd src

# train
python demo_V_extract.py tracking --load_model ../models/bt_ps_ch_twelve.pth --num_class 4 --demo ../videos/weligama/V3_b10_1.mp4 --pre_hm --ltrb_amodal --pre_thresh 0.5 --track_thresh 0.5 --inference --clip_len 2 --trades --save_video --resize_video --input_h 480 --input_w 864
cd ..