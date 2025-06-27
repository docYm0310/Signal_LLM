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
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))  # 1D tensor with shape (1, embedding_dim)
        nn.init.xavier_uniform_(self.mask_token)  # Apply Xavier uniform initialization

        # Learnable 【position embeddings】 for encoder and decoder
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embedding_dim))
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embedding_dim))

        # BERT encoder
        self.bert = BertModel(config)

        # Decoder: embedding → (2, 8)
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, in_channels * patch_size),
            nn.Unflatten(-1, (in_channels, patch_size))  # (B, num_patches, in_channels, patch_size)
        )

    def deal_decoder_inputs(self, x_embed, mask, x_encoded, mask_token):
        full_embed = torch.zeros_like(x_embed)       # [B, num_patches, embedding_dim]
        full_embed[mask == 0] = x_encoded.view(-1, x_embed.shape[-1])
        full_embed[mask == 1] = mask_token.expand(x_embed.shape[0], x_embed.shape[1], x_embed.shape[-1])[mask == 1]
        return full_embed

    def forward(self, signal, mask=None):
        """
        :param signal: Tensor [B, in_channels, signal_length] - 原始信号输入
        :param mask: Tensor [B, num_patches] - 掩蔽的位置索引（每个样本有多个需要掩蔽的位置）
        :return: Tensor [B, num_patches, in_channels, patch_size] - 重建后的 patch
        """
        # Patchify and embed
        conv_feat = self.encoder_conv(signal)  # [B, embedding_dim, num_patches]
        x_embed = conv_feat.permute(0, 2, 1)   # [B, num_patches, embedding_dim]
        # Add position embeddings to encoder input
        x_embed_with_pos = x_embed + self.pos_embed
        # Separate masked and unmasked tokens
        x_embed_nomask = x_embed_with_pos[mask == 0].view(x_embed.size(0), -1, x_embed.size(2))
        # Bert Encoder
        x_encoded = self.bert(inputs_embeds=x_embed_nomask).last_hidden_state  # [B, num_unmasked, embedding_dim]
        # Reconstruct full embedding
        full_embed = self.deal_decoder_inputs(x_embed, mask, x_encoded, self.mask_token)
        # Add position embeddings to decoder input
        full_embed_with_pos = full_embed + self.pos_embed
        # Decoder
        x_decoded = self.decoder(full_embed_with_pos)  # [B, num_patches, in_channels, patch_size]
        return x_decoded