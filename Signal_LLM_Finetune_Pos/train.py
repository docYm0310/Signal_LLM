import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, AdamW
import torch.nn.functional as F
import os
from tqdm import tqdm


# 3. 训练与评估
def finetune(model, train_loader, val_loader, optimizer, device, num_epochs=3, scheduler=None):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    train_loss_history = []
    valid_loss_history = []
    max_valid_acc = 0
    best_epoch = 0
    best_model = None
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(train_loader):
            input_ids = batch['inputs_embeds'].to(device)

            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪，出现梯度爆炸的时候使用

            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            total_correct += torch.sum(preds == labels)
            total_samples += labels.size(0)

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}, Train Accuracy: {total_correct / total_samples}")

        # Evaluate on validation set
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['inputs_embeds'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                _, preds = torch.max(logits, dim=1)
                val_correct += torch.sum(preds == labels)
                val_samples += labels.size(0)
        avg_valid_loss = val_loss / len(val_loader)
        valid_loss_history.append(avg_valid_loss)
        valid_acc = val_correct / val_samples
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            best_model = model
            best_epoch = epoch + 1
        print(f"Validation Loss: {avg_valid_loss}, Validation Accuracy: {valid_acc}")
        torch.cuda.empty_cache()
    print(f"模型第{best_epoch}轮在验证集上取得最优得分{max_valid_acc}")
    return train_loss_history, valid_loss_history, model, best_model, best_epoch, max_valid_acc