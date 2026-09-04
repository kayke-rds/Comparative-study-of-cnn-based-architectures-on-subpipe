import os
import gc
import cv2
import time
import torch
import psutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import albumentations as A
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_utils.submini_dataset import SubPipeMiniDataset, transforms_val, conv_bn_to_gn
from semantic_seg_models.bisenetv2.bisenetv2 import BiSeNetV2
from semantic_seg_models.fast_scnn_tramac.models.fast_scnn import FastSCNN
from semantic_seg_models.pidnet.pidnet import PIDNet

def plot_confusion_matrix(cm, cm_norm, model_name, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], xticklabels=['Fundo', 'Objeto'], yticklabels=['Fundo', 'Objeto'])
    axes[0].set_title(f'{model_name} - Matriz Absoluta')
    sns.heatmap(cm_norm, annot=True, fmt=".4f", cmap="Blues", ax=axes[1], xticklabels=['Fundo', 'Objeto'], yticklabels=['Fundo', 'Objeto'])
    axes[1].set_title(f'{model_name} - Matriz Normalizada')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"), dpi=300)
    plt.close()

def plot_metrics_histogram(results_df, output_dir):
    metrics_to_plot = ['IoU', 'Dice', 'Precision', 'Recall', 'Loss']
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("Comparação de Métricas de Teste e Desempenho Computacional", fontsize=16)
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        sns.barplot(x='Modelo', y=metric, data=results_df, ax=axes[i], palette="viridis")
        axes[i].set_title(metric)
        axes[i].set_ylim(0, 1.0) if metric != 'Loss' else axes[i].set_ylim(0, results_df['Loss'].max() * 1.2)
        for p in axes[i].patches:
            axes[i].annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=10)

    # Gráficos de Performance
    perf_metrics = ['FPS', 'Peak_RAM_MB', 'CPU_Usage_Percent']
    palettes = ["magma", "Reds", "Oranges"]

    for j, p_metric in enumerate(perf_metrics):
        idx = len(metrics_to_plot) + j
        sns.barplot(x='Modelo', y=p_metric, data=results_df, ax=axes[idx], palette=palettes[j])
        axes[idx].set_title(p_metric.replace("_", " "))
        for p in axes[idx].patches:
            axes[idx].annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "comparativo_metricas_com_performance.png"), dpi=300)
    plt.close()

def evaluate_model(model_name, model, checkpoint_path, test_loader, device, output_dir):
    print(f"\n--- Iniciando avaliação: {model_name} ---")

    conv_bn_to_gn(model, num_groups=32)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    if hasattr(model, 'aux_mode'): model.aux_mode = 'eval'
    if hasattr(model, 'augment'): model.augment = False

    writer = SummaryWriter(log_dir=f"runs/best-{model_name.lower()}-test")
    criterion_focal = smp.losses.FocalLoss(mode='binary', alpha=0.5, gamma=2.0)
    criterion_dice = smp.losses.DiceLoss(mode='binary')

    # FASE DE WARMUP
    print(f"[{model_name}] Realizando Warmup (5 batches)...")
    warmup_batches = 5
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= warmup_batches: break
            images = images.float().to(device)
            _ = model(images)

    # SETUP DE RECURSOS
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None) # Chamada de inicialização do psutil
    peak_ram = process.memory_info().rss / (1024 ** 2) # Converte para MB
    cpu_usages = []

    running_loss, total_iou, total_dice, total_precision, total_recall = 0.0, 0.0, 0.0, 0.0, 0.0
    global_TP, global_FP, global_FN, global_TN = 0, 0, 0, 0
    total_inference_time = 0.0
    total_images = 0

    print(f"[{model_name}] Iniciando Inferência e Coleta de Métricas...")
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            print(f"Batch {i+1} de {len(test_loader)}", end='\r')
            images, masks = images.float().to(device), masks.float().to(device)

            # Rastreamento de hardware pré-inferência
            current_ram = process.memory_info().rss / (1024 ** 2)
            peak_ram = max(peak_ram, current_ram)

            # Medição de tempo exata por hardware counter
            start_time = time.perf_counter()

            # Tratamento da saída variada das arquiteturas
            raw_outputs = model(images)
            if model_name in ["BiSeNetV2", "FastSCNN"]:
                outputs = raw_outputs[0].float()
            elif model_name == "PIDNet":
                outputs = F.interpolate(raw_outputs.float(), size=(640, 640), mode='bilinear', align_corners=True)
            else:
                outputs = raw_outputs.float()

            # Fim da medição de tempo
            inference_time = time.perf_counter() - start_time
            total_inference_time += inference_time
            total_images += images.size(0)

            # Rastreamento de CPU do batch
            cpu_usages.append(process.cpu_percent(interval=None))

            # Cálculos de Loss e Métricas
            loss = 0.6 * criterion_focal(outputs, masks) + 0.4 * criterion_dice(outputs, masks)
            running_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()

            tp = torch.sum(preds * masks)
            fp = torch.sum(preds * (1 - masks))
            fn = torch.sum((1 - preds) * masks)
            tn = torch.sum((1 - preds) * (1 - masks))

            global_TP += tp.item()
            global_FP += fp.item()
            global_FN += fn.item()
            global_TN += tn.item()

            precision = tp / (fp + tp + 1e-7)
            recall = tp / (fn + tp + 1e-7)
            intersection = (preds * masks).sum(dim=(2, 3))
            union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) - intersection
            iou = (intersection + 1e-7) / (union + 1e-7)
            dice = (2 * intersection + 1e-7) / (preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) + 1e-7)

            total_iou += iou.mean().item()
            total_dice += dice.mean().item()
            total_precision += precision.mean().item()
            total_recall += recall.mean().item()

    num_batches = len(test_loader)
    avg_cpu_usage = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0.0

    metrics = {
        "Modelo": model_name,
        "Loss": running_loss / num_batches,
        "IoU": total_iou / num_batches,
        "Dice": total_dice / num_batches,
        "Precision": total_precision / num_batches,
        "Recall": total_recall / num_batches,
        "FPS": total_images / total_inference_time,
        "Peak_RAM_MB": peak_ram,
        "CPU_Usage_Percent": avg_cpu_usage
    }

    writer.add_scalar("IoU", metrics["IoU"])
    writer.add_scalar("FPS", metrics["FPS"])
    writer.close()

    cm = np.array([[global_TN, global_FP], [global_FN, global_TP]], dtype=np.int64)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm, cm_norm, model_name, output_dir)

    print(f"\n[Resultados {model_name}] IoU: {metrics['IoU']:.4f} | FPS: {metrics['FPS']:.2f} | RAM Pico: {metrics['Peak_RAM_MB']:.1f} MB | CPU: {metrics['CPU_Usage_Percent']:.1f}%")

    model = None
    gc.collect()
    return metrics

def main():
    # CONFIGURAÇÃO PARA AVALIAÇÃO DE HARDWARE
    # Restringe o PyTorch a usar apenas 1 thread (importante para emular restrições e estabilizar métricas)
    torch.set_num_threads(1)
    print(f"PyTorch configurado para usar {torch.get_num_threads()} thread(s).")

    pasta_imagens = Path("../UnitedDataset/test/images")
    pasta_mascaras = Path("../UnitedDataset/test/masks")

    output_dir = "test_results_visuals"
    os.makedirs(output_dir, exist_ok=True)

    x_list, y_list = [], []
    for img_path in pasta_imagens.iterdir():
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}: continue
        mask_path = pasta_mascaras / f"{img_path.stem}_label.png"
        if mask_path.exists():
            x_list.append(str(img_path))
            y_list.append(str(mask_path))

    test_dataset = SubPipeMiniDataset(x_list, y_list, transforms=transforms_val(640))
    # Importante: drop_last=True mantém o batch shape constante para métricas de tempo estáveis
    test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    device = torch.device("cpu")

    models_to_test = [
        {"name": "BiSeNetV2", "model": BiSeNetV2(n_classes=1), "checkpoint": "models_checkpoints/bisenetv2_united_dataset_fold5_best_iou.pt"},
        {"name": "FastSCNN", "model": FastSCNN(num_classes=1, aux=True), "checkpoint": "models_checkpoints/fast_scnn_united_dataset_fold5_best_iou.pt"},
        {"name": "LinkNet", "model": smp.Linknet(encoder_name="tu-mobilenetv4_conv_small", encoder_weights="imagenet", in_channels=3, classes=1), "checkpoint": "models_checkpoints/linknet+mn4+linknet+mn4_united_dataset_fold1_best_iou.pt"},
        {"name": "PIDNet", "model": PIDNet(num_classes=1), "checkpoint": "models_checkpoints/pidnet_united_dataset_fold3_best_iou.pt"}
    ]

    all_results = []
    for config in models_to_test:
        if not os.path.exists(config["checkpoint"]): continue
        metrics = evaluate_model(config["name"], config["model"], config["checkpoint"], test_loader, device, output_dir)
        all_results.append(metrics)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(output_dir, "resultados_comparativos.csv"), index=False)
        plot_metrics_histogram(results_df, output_dir)
        print(f"\nFinalizado! Gráficos e CSV salvos em: {output_dir}/")

if __name__ == "__main__":
    main()
