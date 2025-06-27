import torch

import h5py
import numpy as np


datamap = h5py.File("./example.mat", 'r')
datamap_ = datamap['IFPDW']['Wave']

for i in range(len(datamap_)):
    idx = datamap_[i][0]
    tmp = datamap[idx][:].T
    tmp = tmp[0]

    real_part = np.array([x[0] for x in tmp])  # 提取实部（I）
    imag_part = np.array([x[1] for x in tmp])  # 提取虚部（Q）
    print(type(real_part))

    exit()
# for i in range(len(datamap)):
# print(idx)
# tmp0 = np.array(datamap(idx))
# print(tmp0)

# for i, idx in enumerate(idx):
#     tmp = np.array(datamap[i][idx])
#     print(tmp)
#     # real = tmp.real  # 取实部
#     # imag = tmp.imag  # 取虚部    
#     # print(real)

