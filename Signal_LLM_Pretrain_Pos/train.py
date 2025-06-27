from tqdm import tqdm
import torch
import torch.nn.functional as F

def masked_combined_loss(outputs, labels, mask_idx, alpha=0.7):
    """
    outputs: (B, 32, 2, 8) - 模型重建结果
    labels:  (B, 32, 2, 8) - 原始信号
    mask_idx: (B, M) - 表示每个样本被 mask 的 patch 的索引
    alpha: 加权系数，alpha*MSE + (1-alpha)*MAE
    """

    B, T, C, L = labels.shape  # B=batch, T=patch数, C=通道, L=patch长度
    device = labels.device

    # 生成 mask（B, T）
    mask = torch.zeros(B, T, device=device)
    mask.scatter_(1, mask_idx, 1)

    # 扩展为与 outputs 形状相同的 mask（B, T, C, L）
    mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, L)

    # 计算逐点误差
    mse = F.mse_loss(outputs, labels, reduction='none')
    mae = F.l1_loss(outputs, labels, reduction='none')

    # 应用 mask
    masked_mse = mse * mask_expanded
    masked_mae = mae * mask_expanded

    # 归一化（只对 mask 区域平均）
    loss = alpha * masked_mse.sum() / mask_expanded.sum() + \
           (1 - alpha) * masked_mae.sum() / mask_expanded.sum()

    return loss

def masked_ab_combined_loss(outputs, labels, mask_idx, alpha=0.7):
    """
    outputs: (B, 32, 2, 8) - 模型重建结果
    labels:  (B, 32, 2, 8) - 原始信号
    mask_idx: (B, M) - 表示每个样本被 mask 的 patch 的索引
    alpha: 加权系数，alpha*MSE + (1-alpha)*MAE
    """

    B, T, C, L = labels.shape  # B=batch, T=patch数, C=通道, L=patch长度
    device = labels.device

    # 生成 mask（B, T）
    mask = torch.zeros(B, T, device=device)
    mask.scatter_(1, mask_idx, 1)

    # 扩展为与 outputs 形状相同的 mask（B, T, C, L）
    mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, L)

    # 计算逐点误差
    mse = F.mse_loss(outputs, labels, reduction='none')
    mae = F.l1_loss(outputs, labels, reduction='none')

    # 应用 mask
    masked_mse = mse * (1 - mask_expanded)
    masked_mae = mae * (1 - mask_expanded)

    # 归一化（只对 mask 区域平均）
    loss = alpha * masked_mse.sum() / mask_expanded.sum() + \
           (1 - alpha) * masked_mae.sum() / mask_expanded.sum()

    return loss

def finetune(model, train_loader, val_loader, optimizer, device, num_epochs=3, scheduler=None, non_mask_loss=False, non_mask_loss_weight=0.2, combined_loss_radio=0.7):
    model.to(device)
    torch.autograd.set_detect_anomaly(True)
    criterion = torch.nn.MSELoss(reduction='none')

    train_loss_history = []
    valid_loss_history = []
    best_val_loss = float('inf')
    best_epoch = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            signal = batch['inputs'].to(device)              # shape: [B, 2, 256]
            labels = batch['labels'].to(device)              # shape: [B, 32, 2, 8]
            mask_idx = batch['mask'].to(device)              # shape: [B, 32] 
            optimizer.zero_grad()
            outputs = model(signal, mask=mask_idx)
            if non_mask_loss:
                loss = masked_combined_loss(outputs,labels,mask_idx,alpha=combined_loss_radio)
                non_loss = masked_ab_combined_loss(outputs,labels,mask_idx,alpha=combined_loss_radio)
                loss = (1 - non_mask_loss_weight) * loss + non_mask_loss_weight * non_loss
            else:
                loss = masked_combined_loss(outputs,labels,mask_idx,alpha=combined_loss_radio)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()


            total_loss += loss.item()

        # if scheduler is not None:
        #     scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, 🟢 Train Loss: {avg_train_loss:.6f}")

        # === 验证 ===
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                signal = batch['inputs'].to(device)        
                labels = batch['labels'].to(device)           
                mask_idx = batch['mask'].to(device)            

                outputs = model(signal, mask=mask_idx)
                if non_mask_loss:
                    loss = masked_combined_loss(outputs,labels,mask_idx,alpha=combined_loss_radio)
                    non_loss = masked_ab_combined_loss(outputs,labels,mask_idx,alpha=combined_loss_radio)
                    loss = (1 - non_mask_loss_weight) * loss + non_mask_loss_weight * non_loss
                else:
                    loss = masked_combined_loss(outputs,labels,mask_idx,alpha=combined_loss_radio)
                val_loss += loss.item()

        avg_valid_loss = val_loss / len(val_loader)
        valid_loss_history.append(avg_valid_loss)

        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            best_model = model
            best_epoch = epoch + 1

        print(f"🔵 Validation Loss: {avg_valid_loss:.6f}")
        torch.cuda.empty_cache()

    print(f"✅ 最佳模型出现在第 {best_epoch} 轮，验证集 Loss 最小值为 {best_val_loss:.6f}")
    return train_loss_history, valid_loss_history, model, best_model, best_epoch, best_val_loss
