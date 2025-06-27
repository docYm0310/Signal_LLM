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

# 添加AWGN噪声
def add_awgn(signal, SNR):
    noise = torch.randn_like(signal)
    signal_power = torch.mean((signal - torch.mean(signal)) ** 2)
    noise_variance = signal_power / (10 ** (SNR / 10))
    noise *= (torch.sqrt(noise_variance) / torch.std(noise, unbiased=False))
    signal_noise = signal + noise
    return signal_noise


def create_collate_fn_IQ_mode(random_windows, add_noise, random_cut, window, snr):
    def collate_fn_IQ(batch):
        data0, flag = [], []
        for i in range(batch.__len__()):
            flag.append(torch.tensor(batch[i]['labels']))
            # data0 截断和加随机数
            tmp = torch.tensor(batch[i]['inputs_embeds'])
            if random_windows:
                start = random.randint(1, random_cut)
                tmp = tmp[:, start:start + window]
            tmp = add_awgn(tmp, SNR=snr) if add_noise else tmp
            data0.append(tmp)

        # 组成N*300
        flag = torch.stack(flag)
        data0 = torch.stack(data0)
        return {
            'inputs_embeds': data0,
            'labels': flag
        }
    return collate_fn_IQ

def loadmatWithRF_Sig(path, file, label):
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
    # for i in range(100):
        tmp = datamap[i][dataColumn]
        real = tmp.real  # 取实部
        imag = tmp.imag  # 取虚部

        # 最大幅度归一化
        Amp = np.abs(tmp)  # 求幅度
        phi = np.angle(tmp)  # 求相位
        Amp_Max = np.max(Amp)
        real = real / Amp_Max
        imag = imag / Amp_Max
        phi = phi / 3.14

        # 整合数据格式
        real = real[0]
        imag = imag[0]

        # 将实部和虚部拼接
        data2 = np.vstack((real, imag))
        data2 = data2.tolist()

        dataset.append(data2) # data2 保持数组类型，方便后面数据处理
        labels = [label] * len(dataset)

    return dataset, labels



class PulseDataset(Dataset):
    def __init__(self, pulse_data, labels, max_len=256, segment_length=8):
        """
        :param segment_length: 每个token包含的原始信号点数
        """
        self.pulse_data = pulse_data
        self.labels = labels
        self.segment_length = segment_length
        self.max_token_length = max_len // segment_length


    def __len__(self):
        return len(self.pulse_data)

    def __getitem__(self, idx):
        pulse = self.pulse_data[idx]
        label = self.labels[idx]
        pulse = torch.tensor(pulse)

        return {
            'inputs_embeds': pulse,
            'labels': torch.tensor(label, dtype=torch.long)
        }

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