import torch

# 假设参数
batch_size = 2
seq_len = 32
hidden_dim = 768
masked_len = 4  # 每个样本中 mask 掩盖的位置数
valid_len = seq_len - masked_len  # 非mask位置数量，即 Encoder_outputs 的长度

# 1. 构造原始输入
X = torch.randn(batch_size, seq_len, hidden_dim)

# 2. 构造mask（随机生成若干个1）
mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
for i in range(batch_size):
    idx = torch.randperm(seq_len)[:masked_len]
    mask[i, idx] = 1

# 3. 构造Encoder_outputs（只对应 mask == 0 的位置）
Encoder_outputs = torch.randn(batch_size, valid_len, hidden_dim)

# 4. 构造mask_token
mask_token = torch.randn(1, 1, hidden_dim)

# 5. 初始化X_new
X_new = torch.zeros_like(X)

# 6. 获取非mask和mask的位置索引
non_mask_indices = ~mask  # [B, 32]
mask_indices = mask       # [B, 32]

# 7. 将 Encoder_outputs 按照非mask位置写入X_new
X_new[non_mask_indices] = Encoder_outputs.reshape(-1, hidden_dim)
print(X_new.shape)

# 8. 将 mask_token 写入mask位置
X_new[mask_indices] = mask_token.expand(batch_size, seq_len, hidden_dim)[mask_indices]

# 打印结果验证
print("X shape:", X.shape)
print("mask shape:", mask.shape)
print("Encoder_outputs shape:", Encoder_outputs.shape)
print("X_new shape:", X_new.shape)
print("是否完全替换成功：", torch.allclose(X_new[~mask], Encoder_outputs.reshape(-1, hidden_dim)))
