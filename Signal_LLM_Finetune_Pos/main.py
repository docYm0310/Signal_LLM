import torch
import random
import numpy as np
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from models.bert import PulseClassifier
from utils import *
from train import finetune
import datetime
import pprint

# 设置全局随机种子
def set_seed(seed):
    random.seed(seed)  # Python随机数生成器
    np.random.seed(seed)  # Numpy随机数生成器
    torch.manual_seed(seed)  # CPU上的随机数生成器
    torch.cuda.manual_seed(seed)  # 当前GPU上的随机数生成器
    torch.cuda.manual_seed_all(seed)  # 所有GPU上的随机数生成器
    torch.backends.cudnn.deterministic = True  # 保证卷积操作的确定性
    torch.backends.cudnn.benchmark = False  # 禁用cudnn自动寻找最优算法，以保证结果可复现

# 主函数
def main(args):
    # 设置随机种子
    set_seed(args.seed)
    # 读入脉冲数据，格式为：[(signal_source_1), (signal_source_2), ...]，每个信号源包含16个采样点
    pulse_data = []
    labels = []
    for i, filename in enumerate(os.listdir(args.dataset_dir)):
        # 加载数据
        pulse_, labels_ = loadmatWithRF_Sig(args.dataset_dir, filename, i)
        print(filename.encode('utf-8', errors='ignore').decode('utf-8'),len(pulse_))
        pulse_data.extend(pulse_)
        labels.extend(labels_)
    pprint.pprint(vars(args))

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(pulse_data, labels, test_size=0.2, stratify=labels) 

    # 初始化tokenizer和数据集
    train_dataset = PulseDataset(X_train, y_train, max_len=args.window,segment_length=args.segment_length)
    val_dataset = PulseDataset(X_val, y_val, max_len=args.window,segment_length=args.segment_length)

    # 数据加载器 args.random_cut, args.random_windows, args.window, args.add_noise, args.noise_snr
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=create_collate_fn_IQ_mode(args.random_windows,args.add_noise,args.random_cut,args.window,args.noise_snr))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=create_collate_fn_IQ_mode(args.random_windows,args.add_noise,args.random_cut,args.window,args.noise_snr))

    # 初始化模型和优化器
    model = PulseClassifier(pretrained_model_name=args.bert_model_path,num_classes=args.num_class,model_dim=768,segment_length=args.segment_length)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
                                                            T_max = args.num_epochs)


    # 选择设备
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 开始训练
    train_loss_history, valid_loss_history, latest_model, best_model, best_epoch, max_valid_acc = finetune(model, train_loader, val_loader, optimizer, device, num_epochs=args.num_epochs, scheduler=scheduler)

    # 存储信息
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d%H%M")
    save_dir_path = args.save_path + f"/INFO_{formatted_time}_{args.num_epochs}_{args.num_class}_{args.batch_size}_{args.lr}_{args.window * 2}_{args.segment_length}_ACC{max_valid_acc}"
    os.makedirs(save_dir_path)
    save_model_path = save_dir_path + f"/model"
    os.makedirs(save_model_path)
    torch.save(best_model, save_model_path + f"/epoch{best_epoch}_best_model.pth")
    torch.save(latest_model, save_model_path + "/latest_model.pth")
    show_plot(train_loss_history, valid_loss_history, save_dir_path)


if __name__ == "__main__":
    # 创建解析器对象
    parser = argparse.ArgumentParser(description="Pulse signal classification with BERT")

    # 环境参数配置
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dataset_dir', type=str, default='../Signal_llm_task_2/data/data_radar/dataset_radar_34_512/allData', help='Directory for the pulse data')
    parser.add_argument('--bert_model_path', type=str, default='../bert_pretrain', help='Path to pre-trained BERT model')
    parser.add_argument('--save_path', type=str, default='./save', help='Path to model loss picture')
    parser.add_argument('--device', type=int, default=0, help='Device number')

    # 训练参数配置
    parser.add_argument('--num_class', type=int, default=2, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--window', type=int, default=128, help='Number of window')
    parser.add_argument('--segment_length', type=int, default=8, help='Length of segment')

    # 可选择参数配置
    parser.add_argument('--random_cut', type=int, default=20, help='Number of random cut range')
    parser.add_argument('--noise_snr', type=int, default=50, help='Number of noise snr')
    parser.add_argument('--random_windows', action="store_true", help='is random cut windows?')
    parser.add_argument('--add_noise', action="store_true", help='is add noise to signal?')

    # 解析命令行参数
    args = parser.parse_args()
    print(args.add_noise)
    print(args.noise_snr)
    # 运行主函数
    main(args)

'''
python main.py --num_epochs 30 --batch_size 256 --num_class 34 --lr 1e-4 --window 128 --segment_length 8 --random_cut 20 --random_windows True --add_noise True --noise_snr 23 
'''
