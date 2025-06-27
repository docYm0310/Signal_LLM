import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, AdamW
import torch.nn.functional as F
import torch
import scipy.io
import os
import json  # 导入json库
import numpy as np
import matplotlib.pyplot as plt
import random

def random_mask(length, ratio=0.15, mean=None, std=None):
    """
    在给定长度的数组中，按正态分布选择不重复的索引，将这些位置置为1，其余为0。
    
    参数:
        length: 数组长度
        ratio: 设置为1的比例
        mean: 正态分布的均值，默认居中
        std: 标准差，默认 length / 6
    返回:
        mask: 一个0-1数组，指定索引处为1
    """
    num_ones = int(length * ratio)
    if mean is None:
        mean = length / 2
    if std is None:
        std = length / 6

    mask = np.zeros(length, dtype=int)

    # 计算最大尝试次数
    max_trials = num_ones * 10
    sampled = []

    # 保证总是采样到 num_ones 个有效索引
    while len(sampled) < num_ones:
        # 多次采样，直到完成
        for _ in range(max_trials):
            candidate = int(np.round(np.random.normal(loc=mean, scale=std)))
            if 0 <= candidate < length and candidate not in sampled:
                sampled.append(candidate)
            if len(sampled) >= num_ones:
                break

        # 如果采样不完整，补充缺失的部分
        if len(sampled) < num_ones:
            # 补充缺失的索引，采用随机选择的方法
            remaining = num_ones - len(sampled)
            available_indexes = set(range(length)) - set(sampled)
            sampled.extend(np.random.choice(list(available_indexes), size=remaining, replace=False))

    mask[sampled] = 1
    return mask

# add AWGN noise
def add_awgn(signal, SNR):
    noise = torch.randn_like(signal)
    signal_power = torch.mean((signal - torch.mean(signal)) ** 2)
    noise_variance = signal_power / (10 ** (SNR / 10))
    noise *= (torch.sqrt(noise_variance) / torch.std(noise, unbiased=False))
    signal_noise = signal + noise
    return signal_noise

def create_collate_fn_IQ_mode(random_windows, add_noise, random_cut, window, snr, segment_length, mask_ratio=0.15):
    def collate_fn_IQ(batch):
        data, flag, mask_idxs = [], [], []
        for i in range(batch.__len__()):
            tmp = torch.tensor(batch[i])
            # add noise or random cut
            if random_windows:
                start = random.randint(1, random_cut)
                tmp = tmp[:, start:start + window]

            # Construct label before adding noise
            label = tmp.unfold(dimension=1, size=segment_length, step=segment_length).permute(1, 0, 2)  # (32, 8, 2)
            tmp = add_awgn(tmp, SNR=snr) if add_noise else tmp
            data.append(tmp)
            # MLM
            mask_idxs.append(torch.tensor(random_mask(window // segment_length, ratio=mask_ratio)))
            flag.append(label)
        flag = torch.stack(flag)
        data = torch.stack(data)
        mask_idxs = torch.stack(mask_idxs)
        return {
            'inputs': data, # (bs, 2, 256)
            'labels': flag, # (bs, 32, 8, 2)
            'mask': mask_idxs # (bs, 32)
        }
    return collate_fn_IQ

def loadmatWithRF_Sig(path, file):
    dataset = []
    datamap = scipy.io.loadmat(path + '/' + file)

    # 根据不同数据格式加载
    if 'sigmat' in datamap.keys():
        datamap = datamap['sigmat'][0]
        dataColumn = 0
    elif 'Data' in datamap.keys():
        datamap = datamap['Data'][0]
        dataColumn = 0
    elif 'allData_Filter_new' in datamap.keys():  # 8Sig
        datamap = datamap['allData_Filter_new'].squeeze(1)
        dataColumn = 4
    elif 'IFPDW' in datamap.keys():
        datamap = datamap['IFPDW'].squeeze(1)
        dataColumn = 4
    elif 'SigMat' in datamap.keys():
        datamap = datamap['SigMat'].squeeze(1)
        dataColumn = 5
    else:
        print("No match loadmat method!")
        exit()

    # 遍历每个样本
    for i in range(len(datamap)):
    # for i in range(50):
        
        tmp = datamap[i][dataColumn]
        real = tmp.real  # 取实部
        imag = tmp.imag  # 取虚部

        # 最大幅度归一化
        # Amp = np.abs(tmp)  # 求幅度
        # phi = np.angle(tmp)  # 求相位
        # Amp_Max = np.max(Amp)
        # real = real / Amp_Max
        # imag = imag / Amp_Max
        # phi = phi / 3.14

        # 数据归一化
        real = (real - real.mean()) / real.std()
        imag = (imag - imag.mean()) / imag.std()

        # 整合数据格式
        real = real[0]
        imag = imag[0]

        # 将实部和虚部拼接
        data2 = np.vstack((real, imag))
        data2 = torch.tensor(data2, dtype=torch.float32)  # ✅ 转换为 Tensor
        dataset.append(data2)

    return dataset # [tensor(2, 512), tensor(2, 512), ...], 每个tensor包含2个通道，512个采样点


class PulseDataset(Dataset):
    def __init__(self, pulse_data):
        """
        :param segment_length: 每个token包含的原始信号点数
        """
        self.pulse_data = pulse_data
        self.train_split = 0.8
        self.valid_split = 0.2


    def __len__(self):
        return len(self.pulse_data)

    def __getitem__(self, idx):
        pulse = self.pulse_data[idx]

        return pulse

    def split_dataset(self):
        data = self.pulse_data
        random.Random(42).shuffle(data)

        size = len(data)
        train_split_end = int(self.train_split * size)
        valid_split_end = int((self.train_split + self.valid_split) * size)

        train_dataset = PulseDataset(data[:train_split_end])
        valid_dataset = PulseDataset(data[train_split_end:valid_split_end])

        return train_dataset, valid_dataset


def show_plot(train_losses_history, valid_losses_history, save_dir_path):
    plt.plot(train_losses_history, label='Training Loss', color='blue')
    plt.plot(valid_losses_history, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    save_loss_path = save_dir_path + "/loss"
    os.makedirs(save_loss_path)
    plt.savefig(save_loss_path + "/train_valid_loss.png")