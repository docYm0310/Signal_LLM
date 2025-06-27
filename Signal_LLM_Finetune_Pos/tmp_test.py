import os
from utils import *
dataset_dir = "../data/data_radar/dataset_radar_89_512/testData/"

for i, filename in enumerate(os.listdir(dataset_dir)):
    pulse_, labels_ = loadmatWithRF_Sig(dataset_dir, filename, i)
    print(int(len(pulse_) // 0.1))


