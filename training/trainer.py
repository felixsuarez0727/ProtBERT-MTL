import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                            r2_score, accuracy_score, f1_score, 
                            confusion_matrix, classification_report)
from model.model import BertMultiTaskModel, FocalLoss, HuberLoss
from utils.utils import save_checkpoint, load_checkpoint, plot_training_history
import torch.nn as nn
import torch.nn.functional as F

def train_improved(model, train_loader, val_loader, optimizer, scheduler,
                  loss_fn_rfu, loss_fn_cpp, device, epochs, patience,
                  alpha=1.0, beta=1.0, warmup_epochs=5, save_path="./saved_model"):
    
    os.makedirs(save_path, exist_ok=True)
    scaler = GradScaler()
    best_val_rmse = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(save_path, "best_model.pth")

    # Listas para métricas de historial
    train_losses = []
    val_rmses = []
    val_maes = []
    val_r2s = []
    val_accuracies = []
    val_f1s = []

    # Pesos adaptativos para las tareas
    task_weights = {'rfu': alpha, 'cpp': beta}

    print(f"🚀 Iniciando entrenamiento mejorado por {epochs} épocas...")
    print(f"📦 Modelo: {model.__class__.__name__}")
    print(f"🎯 Paciencia: {patience}")
    print(f"📱 Dispositivo: {device}")
    print(f"⚖️  Alpha (RFU): {alpha:.2f}, Beta (CPP): {beta:.2f}")
    print(f"🔥 Warmup epochs: {warmup_epochs}")

    for epoch in range(epochs):
        # ENTRENAMIENTO
        model.train()
        total_loss = 0
        rfu_loss_sum = 0
        cpp_loss_sum = 0
        batch_count = 0
        gradient_norms = []

        # Ajuste dinámico de learning rate para warmup
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * lr_scale

        progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}/{epochs}", leave=False)

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            rfu_labels = batch['rfu_label'].to(device)
            cpp_labels = batch['cpp_label'].to(device)
            has_rfu = batch['has_rfu'].to(device)
            has_cpp = batch['has_cpp'].to(device)

            optimizer.zero_grad()

            with autocast():
                rfu_predictions, cpp_predictions = model(input_ids, attention_mask)

                # Pérdida RFU (solo para muestras con etiquetas válidas)
                current_rfu_loss = torch.tensor(0.0, device=device)
                if has_rfu.sum() > 0:
                    rfu_mask = has_rfu.bool()
                    current_rfu_loss = loss_fn_rfu(
                        rfu_predictions[rfu_mask],
                        rfu_labels[rfu_mask]
                    )

                # Pérdida CPP (solo para muestras con etiquetas válidas)
                current_cpp_loss = torch.tensor(0.0, device=device)
                if has_cpp.sum() > 0:
                    cpp_mask = has_cpp.bool()
                    current_cpp_loss = loss_fn_cpp(
                        cpp_predictions[cpp_mask],
                        cpp_labels[cpp_mask]
                    )

                # Pérdida total ponderada
                loss = task_weights['rfu'] * current_rfu_loss + task_weights['cpp'] * current_cpp_loss

                # Regularización L2 adicional
                l2_reg = torch.tensor(0.0, device=device)
                for param in model.parameters():
                    if param.requires_grad:
                        l2_reg += torch.norm(param, 2)
                loss += 1e-5 * l2_reg

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  Skipping batch due to NaN/Inf loss")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            gradient_norms.append(grad_norm.item())

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            rfu_loss_sum += current_rfu_loss.item()
            cpp_loss_sum += current_cpp_loss.item()
            batch_count += 1

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rfu': f'{current_rfu_loss.item():.4f}',
                'cpp': f'{current_cpp_loss.item():.4f}',
                'grad': f'{grad_norm.item():.3f}'
            })

        if batch_count == 0:
            print("Error: No se procesaron batches válidos")
            break

        avg_train_loss = total_loss / batch_count
        avg_rfu_loss = rfu_loss_sum / batch_count
        avg_cpp_loss = cpp_loss_sum / batch_count
        avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0

        # VALIDACIÓN
        val_rmse, val_mae, val_r2, val_accuracy, val_f1 = evaluate_improved(
            model, val_loader, device, loss_fn_rfu, loss_fn_cpp
        )

        # Guardar métricas
        train_losses.append(avg_train_loss)
        val_rmses.append(val_rmse)
        val_maes.append(val_mae)
        val_r2s.append(val_r2)
        val_accuracies.append(val_accuracy)
        val_f1s.append(val_f1)

        # Ajuste dinámico de pesos de tareas
        if epoch > 5:  # Después de algunas épocas
            # Si RFU no mejora, aumentar su peso
            if len(val_rmses) > 2 and val_rmses[-1] >= val_rmses[-2]:
                task_weights['rfu'] = min(task_weights['rfu'] * 1.1, 2.0)
            # Si CPP no mejora, aumentar su peso
            if len(val_accuracies) > 2 and val_accuracies[-1] <= val_accuracies[-2]:
                task_weights['cpp'] = min(task_weights['cpp'] * 1.1, 2.0)

        print(f"📊 Época {epoch+1:3d} | "
              f"Loss: {avg_train_loss:.4f} | "
              f"RFU_L: {avg_rfu_loss:.4f} | "
              f"CPP_L: {avg_cpp_loss:.4f} | "
              f"Val RMSE: {val_rmse:.4f} | "
              f"Val R²: {val_r2:.4f} | "
              f"Val Acc: {val_accuracy:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"Grad: {avg_grad_norm:.3f}")

        # Actualizar scheduler después del warmup
        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step(val_rmse)
            current_lr = optimizer.param_groups[0]['lr']
            if epoch > warmup_epochs and hasattr(optimizer.param_groups[0], 'initial_lr'):
                if current_lr != optimizer.param_groups[0]['initial_lr']:
                    print(f"📉 LR actualizado: {current_lr:.2e}")

        # Early stopping (considera ambas métricas)
        combined_metric = val_rmse - val_accuracy  # Minimizar RMSE, maximizar accuracy
        if epoch == 0:
            best_combined_metric = combined_metric

        if combined_metric < best_combined_metric:
            best_combined_metric = combined_metric
            best_val_rmse = val_rmse
            patience_counter = 0

            save_checkpoint(model, optimizer, scheduler, epoch, best_val_rmse,
                          avg_train_loss, best_model_path)
            print(f"Mejor modelo guardado (RMSE: {best_val_rmse:.4f}, Acc: {val_accuracy:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping en época {epoch+1}")
                break

    # Cargar mejor modelo
    checkpoint = load_checkpoint(best_model_path, model, optimizer, scheduler, device)
    if checkpoint is None:
        print("No se pudo cargar el mejor modelo")

    # Guardar historial
    training_history = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_rmse': val_rmses,
        'val_mae': val_maes,
        'val_r2': val_r2s,
        'val_accuracy': val_accuracies,
        'val_f1': val_f1s
    })
    training_history.to_csv(os.path.join(save_path, "training_history.csv"), index=False)
    plot_training_history(training_history, save_path=save_path)

    return avg_train_loss, best_val_rmse

def evaluate_improved(model, data_loader, device, loss_fn_rfu, loss_fn_cpp, save_plots=False, save_path="./saved_model"):
    """Evaluación para ambas tareas"""
    model.eval()
    rfu_predictions, rfu_actuals = [], []
    cpp_predictions, cpp_actuals = [], []
    cpp_probs = []  # Para análisis de confianza

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluando", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            rfu_labels = batch['rfu_label'].to(device)
            cpp_labels = batch['cpp_label'].to(device)
            has_rfu = batch['has_rfu'].to(device)
            has_cpp = batch['has_cpp'].to(device)

            batch_rfu_predictions, batch_cpp_logits = model(input_ids, attention_mask)

            # Procesar RFU
            rfu_mask = has_rfu.bool()
            if rfu_mask.any():
                rfu_predictions.extend(batch_rfu_predictions[rfu_mask].cpu().numpy())
                rfu_actuals.extend(rfu_labels[rfu_mask].cpu().numpy())

            # Procesar CPP
            cpp_mask = has_cpp.bool()
            if cpp_mask.any():
                cpp_probs_batch = F.softmax(batch_cpp_logits[cpp_mask], dim=1)
                predicted_classes = torch.argmax(cpp_probs_batch, dim=1)

                cpp_predictions.extend(predicted_classes.cpu().numpy())
                cpp_actuals.extend(cpp_labels[cpp_mask].cpu().numpy())
                cpp_probs.extend(cpp_probs_batch.cpu().numpy())

    # Métricas RFU
    if len(rfu_predictions) > 0:
        rfu_predictions = np.array(rfu_predictions)
        rfu_actuals = np.array(rfu_actuals)
        rmse = np.sqrt(mean_squared_error(rfu_actuals, rfu_predictions))
        mae = mean_absolute_error(rfu_actuals, rfu_predictions)
        r2 = r2_score(rfu_actuals, rfu_predictions)
    else:
        rmse, mae, r2 = float('inf'), float('inf'), -float('inf')

    # Métricas CPP
    if len(cpp_predictions) > 0:
        cpp_predictions = np.array(cpp_predictions)
        cpp_actuals = np.array(cpp_actuals)
        accuracy = accuracy_score(cpp_actuals, cpp_predictions)
        f1 = f1_score(cpp_actuals, cpp_predictions, average='weighted', zero_division=0)

        # Análisis de confianza
        cpp_probs = np.array(cpp_probs)
        confidence = np.max(cpp_probs, axis=1)
        print(f"Confianza promedio CPP: {confidence.mean():.3f} ± {confidence.std():.3f}")
    else:
        accuracy, f1 = 0.0, 0.0

    if save_plots and len(rfu_predictions) > 0:
        create_evaluation_plots(rfu_actuals, rfu_predictions, save_path=save_path)

    return rmse, mae, r2, accuracy, f1

def create_evaluation_plots(y_true, y_pred, save_path="./saved_model"):
    """Crea gráficos de evaluación para la tarea de regresión"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis de Evaluación - Regresión RFU', fontsize=16, fontweight='bold')

        # Scatter plot: Predicho vs Real
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=50, color='blue')
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Línea perfecta')
        axes[0, 0].set_xlabel('Valores Reales')
        axes[0, 0].set_ylabel('Valores Predichos')
        axes[0, 0].set_title('Predicho vs Real')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Histograma de residuos
        residuals = y_pred - y_true
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Residuos (Predicho - Real)')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].set_title('Distribución de Residuos')
        axes[0, 1].grid(True, alpha=0.3)

        # Q-Q plot para normalidad de residuos
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot - Normalidad de Residuos')
        axes[1, 0].grid(True, alpha=0.3)

        # Residuos vs Predichos
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6, s=50, color='orange')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Valores Predichos')
        axes[1, 1].set_ylabel('Residuos')
        axes[1, 1].set_title('Residuos vs Predichos')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(save_path, "evaluation_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Gráficos de evaluación guardados en {plot_path}")
    except Exception as e:
        print(f"Error creando gráficos de evaluación: {e}")

def create_confusion_matrix(y_true, y_pred, class_names=None, save_path="./saved_model"):
    """Crea matriz de confusión para clasificación"""
    try:
        cm = confusion_matrix(y_true, y_pred)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Matriz de confusión
        if class_names is None:
            class_names = [f'Clase {i}' for i in range(len(cm))]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_title('Matriz de Confusión')
        axes[0].set_ylabel('Etiquetas Reales')
        axes[0].set_xlabel('Etiquetas Predichas')

        # Matriz normalizada
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=axes[1])
        axes[1].set_title('Matriz de Confusión Normalizada')
        axes[1].set_ylabel('Etiquetas Reales')
        axes[1].set_xlabel('Etiquetas Predichas')

        plt.tight_layout()
        plot_path = os.path.join(save_path, "confusion_matrix.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Reporte de clasificación
        print("\nReporte de Clasificación:")
        print("="*50)
        print(classification_report(y_true, y_pred, target_names=class_names))

    except Exception as e:
        print(f"Error creando matriz de confusión: {e}")

def detailed_analysis(model, test_loader, device, scaler, save_path="./saved_model"):
    """Análisis detallado de las predicciones"""
    model.eval()

    all_predictions = {'rfu': [], 'cpp': []}
    all_actuals = {'rfu': [], 'cpp': []}
    prediction_confidence = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Análisis detallado"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            rfu_labels = batch['rfu_label'].to(device)
            cpp_labels = batch['cpp_label'].to(device)
            has_rfu = batch['has_rfu'].to(device)
            has_cpp = batch['has_cpp'].to(device)

            rfu_pred, cpp_logits = model(input_ids, attention_mask)
            cpp_probs = F.softmax(cpp_logits, dim=1)
            cpp_pred = torch.argmax(cpp_probs, dim=1)

            # Recopilar datos para análisis
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                if has_rfu[i] == 1:
                    all_predictions['rfu'].append(rfu_pred[i].cpu().item())
                    all_actuals['rfu'].append(rfu_labels[i].cpu().item())

                if has_cpp[i] == 1:
                    all_predictions['cpp'].append(cpp_pred[i].cpu().item())
                    all_actuals['cpp'].append(cpp_labels[i].cpu().item())
                    prediction_confidence.append(torch.max(cpp_probs[i]).cpu().item())

    # Desnormalizar valores RFU si es necesario
    if len(all_predictions['rfu']) > 0 and scaler is not None:
        try:
            all_predictions['rfu'] = scaler.inverse_transform(
                np.array(all_predictions['rfu']).reshape(-1, 1)
            ).flatten()
            all_actuals['rfu'] = scaler.inverse_transform(
                np.array(all_actuals['rfu']).reshape(-1, 1)
            ).flatten()
        except:
            print("No se pudo desnormalizar los valores RFU")

    # Análisis estadístico
    print("\nANÁLISIS ESTADÍSTICO DETALLADO:")
    print("="*50)

    if len(all_predictions['rfu']) > 0:
        rfu_pred = np.array(all_predictions['rfu'])
        rfu_actual = np.array(all_actuals['rfu'])

        print(f"   RFU - Estadísticas de Predicción:")
        print(f"   Media predicha: {rfu_pred.mean():.4f} ± {rfu_pred.std():.4f}")
        print(f"   Media real: {rfu_actual.mean():.4f} ± {rfu_actual.std():.4f}")
        print(f"   Rango predicho: [{rfu_pred.min():.4f}, {rfu_pred.max():.4f}]")
        print(f"   Rango real: [{rfu_actual.min():.4f}, {rfu_actual.max():.4f}]")

        # Análisis de percentiles
        percentiles = [10, 25, 50, 75, 90]
        print(f"   Percentiles predichos: {np.percentile(rfu_pred, percentiles)}")
        print(f"   Percentiles reales: {np.percentile(rfu_actual, percentiles)}")

    if len(all_predictions['cpp']) > 0:
        cpp_pred = np.array(all_predictions['cpp'])
        cpp_actual = np.array(all_actuals['cpp'])
        confidence = np.array(prediction_confidence)

        print(f"\n CPP - Estadísticas de Clasificación:")
        print(f"   Distribución predicha: {np.bincount(cpp_pred)}")
        print(f"   Distribución real: {np.bincount(cpp_actual)}")
        print(f"   Confianza promedio: {confidence.mean():.4f} ± {confidence.std():.4f}")
        print(f"   Predicciones de alta confianza (>0.9): {(confidence > 0.9).sum()}/{len(confidence)}")

        # Crear matriz de confusión
        create_confusion_matrix(cpp_actual, cpp_pred, ['non-CPP', 'CPP'], save_path=save_path)

    print(" Análisis detallado completado")
