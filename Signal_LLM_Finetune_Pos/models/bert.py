import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, AdamW, BertConfig
from models.model import BertForMaskedSignalAutoencoder



class PulseClassifier(torch.nn.Module):
    def __init__(self, pretrained_model_name="./", num_classes=2, model_dim=768, segment_length=8):
        super().__init__()
        
        self.model = torch.load(pretrained_model_name)

        # Patch Embedding
        self.encoder_conv = self.model.encoder_conv
        self.encoder_pos = self.model.encoder_pos_embed
        self.bert = self.model.bert
        self.fc = nn.Linear(768, num_classes)

    def forward(self, inputs_ids):
        inputs_emb = self.encoder_conv(inputs_ids)
        inputs_emb = inputs_emb.permute(0, 2, 1)
        inputs_emb = inputs_emb + self.encoder_pos

        last_hidden = self.bert(
            inputs_embeds=inputs_emb
        ).last_hidden_state
        cls = last_hidden[:, 0, :]
        logits = self.fc(cls)
        return logits