

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


# 在test_model函数内进行调整
def test_model(model, test_dataloader, criterion, device, num_classes, label_to_filename_map):
    model.eval()
    test_loss = 0
    test_correct = 0
    test_samples = 0
    correct_per_class = [0] * num_classes  # 各个标签类别的正确预测数量
    total_per_class = [0] * num_classes  # 各个标签类别的总样本数量

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            input_ids = batch['inputs_embeds'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            test_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            test_correct += torch.sum(preds == labels).item()
            test_samples += labels.size(0)

            # 更新每个标签类别的正确预测数量和总样本数量
            for j in range(labels.size(0)):
                label = labels[j].item()
                correct_per_class[label] += (preds[j] == labels[j]).item()
                total_per_class[label] += 1

    avg_test_loss = test_loss / len(test_dataloader)
    test_accuracy = test_correct / test_samples

    # 输出各个类别的准确率，并根据映射数组输出文件名的准确率
    label_accuracies = {}
    for label in range(num_classes):
        label_accuracy = correct_per_class[label] / total_per_class[label] if total_per_class[label] > 0 else 0
        label_name = label_to_filename_map[label]  # 根据映射数组获取文件名
        label_accuracies[label_name] = label_accuracy

    return avg_test_loss, test_accuracy, label_accuracies


# 主函数修改
def main(args):
    # 设置随机种子
    set_seed(args.seed)

    # 读入脉冲数据，格式为：[(signal_source_1), (signal_source_2), ...]，每个信号源包含16个采样点
    pulse_data = []
    labels = []
    label_to_filename_map = {}  # 标签-文件名映射表 index为标签

    for i, filename in enumerate(os.listdir(args.dataset_dir)):
        # 加载数据
        pulse_, labels_ = loadmatWithRF_Sig(args.dataset_dir, filename, i)
        print(int(len(pulse_) / 0.1))
        pulse_data.extend(pulse_)
        labels.extend(labels_)
        filename = filename.encode('utf-8', errors='ignore').decode('utf-8')
        label_to_filename_map[labels_[0]] = filename

    print(label_to_filename_map)
    pprint.pprint(vars(args))

    # 构造测试数据集
    test_dataset = PulseDataset(pulse_data, labels, args.window, args.segment_length)
    # 数据加载器
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=create_collate_fn_IQ_mode(args.random_windows,args.add_noise,args.random_cut,args.window,args.noise_snr))

    # 初始化模型和优化器
    model = torch.load(args.model_path)
    # 选择设备
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()

    # 开始测试
    test_loss, test_accuracy, label_accuracies = test_model(model, test_dataloader, criterion, device, num_classes=args.num_class, label_to_filename_map=label_to_filename_map)
    print("测试集损失：", test_loss)
    print("测试集总精确率：", test_accuracy)

    # 输出每个标签对应的文件名的准确率
    for label_name, accuracy in label_accuracies.items():
        print(f"{label_name} ： {accuracy:.2f}")
        # print(f"{accuracy:.2f}")


if __name__ == "__main__":
    # 创建解析器对象
    parser = argparse.ArgumentParser(description="Pulse signal classification with BERT")

    # 环境参数配置
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dataset_dir', type=str, default='../Signal_llm_task_2/data/data_radar/dataset_radar_34_512/allData', help='Directory for the pulse data')
    parser.add_argument('--model_path', type=str, default='./save', help='Path to model loss picture')
    parser.add_argument('--device', type=int, default=0, help='Device number')
    
    # 训练参数配置
    parser.add_argument('--num_class', type=int, default=2, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation')
    parser.add_argument('--window', type=int, default=128, help='Number of window')
    parser.add_argument('--segment_length', type=int, default=8, help='Length of segment')

    # 可选择参数配置
    parser.add_argument('--random_cut', type=int, default=20, help='Number of random cut range')
    parser.add_argument('--random_windows', action="store_true", help='is random cut windows?')
    parser.add_argument('--add_noise', action="store_true", help='is add noise to signal?')
    parser.add_argument('--noise_snr', type=int, default=50, help='Number of noise snr')

    # 解析命令行参数
    args = parser.parse_args()

    # 运行主函数
    main(args)