import torch
import random
import numpy as np
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, BertConfig
from sklearn.model_selection import train_test_split
from models.bert import BertForMaskedSignalAutoencoder
from utils import *
from train import finetune
import datetime
import pprint
from transformers import get_cosine_schedule_with_warmup


# 设置全局随机种子
def set_seed(seed):
    random.seed(seed)  # Python随机数生成器
    np.random.seed(seed)  # Numpy随机数生成器
    torch.manual_seed(seed)  # CPU上的随机数生成器
    torch.cuda.manual_seed(seed)  # 当前GPU上的随机数生成器
    torch.cuda.manual_seed_all(seed)  # 所有GPU上的随机数生成器
    torch.backends.cudnn.deterministic = True  # 保证卷积操作的确定性
    torch.backends.cudnn.benchmark = False  # 禁用cudnn自动寻找最优算法，以保证结果可复现

# 
def main(args):
    # Set seed
    set_seed(args.seed)
    pulse_data = []
    for filename in os.listdir(args.dataset_dir):
        # Read data
        pulse_ = loadmatWithRF_Sig(args.dataset_dir, filename)
        print(filename.encode('utf-8', errors='ignore').decode('utf-8'),len(pulse_))
        pulse_data.extend(pulse_)
    pprint.pprint(vars(args))

    # Construct Dataset
    dataset = PulseDataset(pulse_data)
    train_dataset, val_dataset = dataset.split_dataset()

    # Construct DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=create_collate_fn_IQ_mode(args.random_windows,args.add_noise,args.random_cut,args.window,args.noise_snr,args.segment_length,args.mask_ratio))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, collate_fn=create_collate_fn_IQ_mode(args.random_windows,args.add_noise,args.random_cut,args.window,args.noise_snr,args.segment_length,args.mask_ratio))

    # 初始化模型和优化器
    config = BertConfig.from_pretrained('./bert_pretrain')
    model = BertForMaskedSignalAutoencoder(config)
    # 定义衰减参数
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.lr)
    num_training_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # optimizer = AdamW(model.parameters(), lr=args.lr)
    # scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
    #                                                         T_max = args.num_epochs)


    # 选择设备
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # 开始训练
    train_loss_history, valid_loss_history, latest_model, best_model, best_epoch, best_valid_loss = finetune(model, train_loader, val_loader, optimizer, device, num_epochs=args.num_epochs, scheduler=scheduler, non_mask_loss=args.non_mask_loss, non_mask_loss_weight=args.non_mask_loss_weight, combined_loss_radio=args.combined_loss_radio)

    # 存储信息
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d%H%M")
    save_dir_path = args.save_path + f"/INFO_{formatted_time}_{args.num_epochs}_{args.batch_size}_{args.lr}_{args.window}_{args.segment_length}_LOSS{best_valid_loss}"
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
    parser.add_argument('--bert_model_path', type=str, default='./bert_pretrain', help='Path to pre-trained BERT model')
    parser.add_argument('--save_path', type=str, default='./save', help='Path to model loss picture')
    parser.add_argument('--device', type=int, default=0, help='Device number')

    # 训练参数配置
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Learning rate for the optimizer')
    parser.add_argument('--non_mask_loss_weight', type=float, default=0.2, help='Learning rate for the optimizer')
    parser.add_argument('--combined_loss_radio', type=float, default=0.7, help='Learning rate for the optimizer')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--window', type=int, default=128, help='Number of window')
    parser.add_argument('--segment_length', type=int, default=8, help='Length of segment')

    # 可选择参数配置
    parser.add_argument('--random_cut', type=int, default=20, help='Number of random cut range')
    parser.add_argument('--noise_snr', type=int, default=50, help='Number of noise snr')
    parser.add_argument('--random_windows', action="store_true", help='is random cut windows?')
    parser.add_argument('--add_noise', action="store_true", help='is add noise to signal?')
    parser.add_argument('--non_mask_loss', action="store_true", help='is add noise to signal?')

    # 解析命令行参数
    args = parser.parse_args()
    # 运行主函数
    main(args)
