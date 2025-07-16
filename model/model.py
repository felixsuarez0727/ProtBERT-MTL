import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BertMultiTaskModel(nn.Module):
    def __init__(self, dropout=0.3, pooling_strategy='mean'):
        super(BertMultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained("Rostlab/prot_bert")
        self.hidden_size = self.bert.config.hidden_size
        self.pooling_strategy = pooling_strategy

        # Congelar/descongelar capas de BERT
        for param in self.bert.parameters():
            param.requires_grad = False
        # Solo fine-tune las últimas 4 capas
        for param in self.bert.encoder.layer[-4:].parameters():
            param.requires_grad = True

        for param in self.bert.pooler.parameters():
            param.requires_grad = True

        # Cabeza de Regresión (para RFU)
        self.regressor_head = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )

        # Cabeza de Clasificación (para CPP/non-CPP)
        self.classifier_head = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 2)
        )

        self._init_weights()

    def _init_weights(self):
        def init_module(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        self.regressor_head.apply(init_module)
        self.classifier_head.apply(init_module)

    def _mean_pooling(self, hidden_state, attention_mask):
        attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden_state)
        masked_embeddings = hidden_state * attention_mask
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        sum_mask = torch.sum(attention_mask, dim=1)
        return sum_embeddings / torch.clamp(sum_mask, min=1e-9)

    def _max_pooling(self, hidden_state, attention_mask):
        attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden_state)
        masked_embeddings = hidden_state * attention_mask
        masked_embeddings[attention_mask == 0] = -torch.inf
        return torch.max(masked_embeddings, dim=1)[0]

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.pooling_strategy == 'mean':
            pooled = self._mean_pooling(outputs.last_hidden_state, attention_mask)
        elif self.pooling_strategy == 'cls':
            pooled = outputs.last_hidden_state[:, 0, :]
        elif self.pooling_strategy == 'max':
            pooled = self._max_pooling(outputs.last_hidden_state, attention_mask)
        else:
            raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")

        rfu_prediction = self.regressor_head(pooled).squeeze(-1)
        cpp_prediction = self.classifier_head(pooled)

        return rfu_prediction, cpp_prediction

class FocalLoss(nn.Module):
    """Focal Loss para manejar desbalance de clases"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class HuberLoss(nn.Module):
    """Huber Loss para regresión"""
    def __init__(self, delta=1.0, reduction='mean'):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, input, target):
        residual = torch.abs(input - target)
        condition = residual < self.delta
        squared_loss = 0.5 * residual ** 2
        linear_loss = self.delta * residual - 0.5 * self.delta ** 2
        loss = torch.where(condition, squared_loss, linear_loss)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
