import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class MultiTaskPeptideDataset(Dataset):
    def __init__(self, sequences, rfu_labels, cpp_labels, tokenizer, max_length=256):
        self.sequences = sequences
        self.rfu_labels = rfu_labels
        self.cpp_labels = cpp_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = str(self.sequences[idx])
        rfu_label = self.rfu_labels[idx]
        cpp_label = self.cpp_labels[idx]

        # Espacios entre aminoácidos para ProtBERT
        spaced_sequence = ' '.join(sequence)

        encoding = self.tokenizer(
            spaced_sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

        # Manejador de etiquetas RFU (regresión)
        if pd.isna(rfu_label):
            item['rfu_label'] = torch.tensor(float('nan'))
            item['has_rfu'] = torch.tensor(0.0)  # Indicador para pérdida
        else:
            item['rfu_label'] = torch.tensor(rfu_label, dtype=torch.float32)
            item['has_rfu'] = torch.tensor(1.0)

        # Manejador de etiquetas CPP (clasificación)
        if pd.isna(cpp_label) or cpp_label == -100:
            item['cpp_label'] = torch.tensor(-100, dtype=torch.long)
            item['has_cpp'] = torch.tensor(0.0)  # Indicador para pérdida
        else:
            item['cpp_label'] = torch.tensor(int(cpp_label), dtype=torch.long)
            item['has_cpp'] = torch.tensor(1.0)

        return item
