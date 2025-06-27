python test.py \
    --dataset_dir "../data_radar_34_512/testData"  \
    --model_path "./save/INFO_202505172246_50_34_256_0.0001_512_8_ACC0.9952966570854187/model/epoch49_best_model.pth" \
    --batch_size 256 --window 256 --segment_length 8 --random_cut 20 \
    --random_windows --add_noise --noise_snr 20 --num_class 34 --device 3