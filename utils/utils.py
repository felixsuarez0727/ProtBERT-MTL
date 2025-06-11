import os
import torch
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def save_checkpoint(model, optimizer, scheduler, epoch, best_rmse, loss, filepath):
    """Guarda checkpoint del modelo"""
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_rmse': best_rmse,
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, filepath)
        return True
    except Exception as e:
        print(f"❌ Error guardando checkpoint: {e}")
        return False

def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cpu'):
    """Carga checkpoint del modelo"""
    try:
        if not os.path.exists(filepath):
            print(f"Archivo de checkpoint no encontrado: {filepath}")
            return None

        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint cargado exitosamente de época {checkpoint['epoch']}")
        return checkpoint
    except Exception as e:
        print(f"Error cargando checkpoint: {e}")
        return None

def plot_training_history(history_df, save_path="./saved_model"):
    """Visualiza el historial de entrenamiento"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Historial de Entrenamiento Multi-Tarea', fontsize=16, fontweight='bold')

        # Pérdida de entrenamiento
        axes[0, 0].plot(history_df['epoch'], history_df['train_loss'],
                       'b-', linewidth=2, label='Training Loss')
        axes[0, 0].set_title('Pérdida de Entrenamiento')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Métricas RFU
        axes[0, 1].plot(history_df['epoch'], history_df['val_rmse'],
                       'r-', linewidth=2, label='RMSE')
        axes[0, 1].plot(history_df['epoch'], history_df['val_mae'],
                       'orange', linewidth=2, label='MAE')
        axes[0, 1].set_title('Metricas RFU (Regresion)')
        axes[0, 1].set_xlabel('Epoca')
        axes[0, 1].set_ylabel('Error')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # R² Score
        axes[0, 2].plot(history_df['epoch'], history_df['val_r2'],
                       'g-', linewidth=2, label='R² Score')
        axes[0, 2].set_title('R² Score (RFU)')
        axes[0, 2].set_xlabel('Época')
        axes[0, 2].set_ylabel('R²')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()

        # Métricas CPP
        axes[1, 0].plot(history_df['epoch'], history_df['val_accuracy'],
                       'purple', linewidth=2, label='Accuracy')
        axes[1, 0].set_title('Accuracy CPP (Clasificación)')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # F1 Score
        axes[1, 1].plot(history_df['epoch'], history_df['val_f1'],
                       'brown', linewidth=2, label='F1 Score')
        axes[1, 1].set_title('F1 Score CPP')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        # Resumen combinado
        ax_twin = axes[1, 2].twinx()
        axes[1, 2].plot(history_df['epoch'], history_df['val_rmse'],
                       'r-', linewidth=2, label='RMSE')
        ax_twin.plot(history_df['epoch'], history_df['val_accuracy'],
                    'purple', linewidth=2, label='Accuracy')
        axes[1, 2].set_title('Resumen Multi-Tarea')
        axes[1, 2].set_xlabel('Época')
        axes[1, 2].set_ylabel('RMSE', color='red')
        ax_twin.set_ylabel('Accuracy', color='purple')
        axes[1, 2].grid(True, alpha=0.3)

        # Leyenda combinada
        lines1, labels1 = axes[1, 2].get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        axes[1, 2].legend(lines1 + lines2, labels1 + labels2, loc='center right')

        plt.tight_layout()
        plot_path = os.path.join(save_path, "training_history.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Gráficos de entrenamiento guardados en {plot_path}")
    except Exception as e:
        print(f"Error creando gráficos de entrenamiento: {e}")
