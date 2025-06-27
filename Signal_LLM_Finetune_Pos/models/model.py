import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BertForMaskedSignalAutoencoder(nn.Module):
    def __init__(self, config, patch_size=8, in_channels=2, signal_length=256):
        super(BertForMaskedSignalAutoencoder, self).__init__()
        self.patch_size = patch_size
        self.num_patches = signal_length // patch_size
        self.in_channels = in_channels
        self.embedding_dim = config.hidden_size

        # Patch Embedding
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(in_channels, self.embedding_dim, kernel_size=patch_size, stride=patch_size)
        )

        # Learnable [MASK] token
        self.mask_token = nn.Parameter(torch.zeros(1, self.embedding_dim))  # 1D tensor with shape (1, embedding_dim)
        nn.init.xavier_uniform_(self.mask_token)  # Apply Xavier uniform initialization

        # BERT encoder
        self.bert = BertModel(config)

        # Decoder: embedding → (2, 8)
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, in_channels * patch_size),
            nn.Unflatten(-1, (in_channels, patch_size))  # (B, 32, 2, 8)
        )


    def forward(self, signal, mask_indices=None):
        """
        :param signal: Tensor [B, 2, 256] - 原始信号输入
        :param mask_indices: Tensor [B, 4] - 掩蔽的位置索引（每个样本有 4 个需要掩蔽的位置）
        :return: Tensor [B, 2, 32, 8] - 重建后的 patch
        """
        B = signal.shape[0]  # 获取 batch size
        len_mask = mask_indices.shape[1]  # 获取每个样本掩蔽位置的数量

        # Step 1: Patchify and embed
        conv_feat = self.encoder_conv(signal)  # [B, 768, 32]
        x_embed = conv_feat.permute(0, 2, 1)   # [B, 32, 768]

        # Step 2: Masking (replace selected tokens with mask_token)
        if mask_indices is not None:
            # 利用 batch_indices 和 mask_indices 来替换特定位置
            batch_indices = torch.arange(B).unsqueeze(1).expand(-1, len_mask)  # [B, len_mask]，用于标识 batch 中的每个样本
            # 通过 batch_indices 和 mask_indices 来替换指定位置的 token 为 mask_token
            for i in range(len_mask):
                x_embed[batch_indices[:, i], mask_indices[:, i]] = self.mask_token  # 使用 mask_token 代替原值

        # Step 3: BERT encode
        x_encoded = self.bert(inputs_embeds=x_embed).last_hidden_state  # [B, 32, 768]
        # Step 4: Decode
        x_decoded = self.decoder(x_encoded)  # [B, 32, 2, 8]

        return x_decoded
