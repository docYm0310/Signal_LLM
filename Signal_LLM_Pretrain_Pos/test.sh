python test.py \
    --dataset_dir "../data/data_radar/dataset_radar_34_512/testData/" \
    --model_path "./save/INFO_202504141954_50_34_256_0.0001_512_8_ACC0.9907851815223694/model/epoch50_best_model.pth" \
    --batch_size 256 --window 256 --segment_length 128 --random_cut 20 \
    --random_windows --add_noise --noise_snr 20 --num_class 34 --device 2