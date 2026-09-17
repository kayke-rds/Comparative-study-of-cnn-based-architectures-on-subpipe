import argparse
import gc
import os
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_utils.submini_dataset import SubPipeMiniDataset, transforms_val
from semantic_seg_models.bisenetv2.bisenetv2 import BiSeNetV2
from semantic_seg_models.fast_scnn_tramac.models.fast_scnn import FastSCNN
from semantic_seg_models.pidnet.pidnet import PIDNet
from train_scripts.train_utils import calculate_batch_metrics, val_losses_composition


ARCHITECTURES = {
    "fast-scnn": {
        "display_name": "FastSCNN",
        "checkpoint_prefix": "fast-scnn",
    },
    "bisenetv2": {
        "display_name": "BiSeNetV2",
        "checkpoint_prefix": "bisenetv2",
    },
    "linknet": {
        "display_name": "LinkNet",
        "checkpoint_prefix": "linknet",
    },
    "pidnet": {
        "display_name": "PIDNet",
        "checkpoint_prefix": "pidnet",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Avalia no conjunto de teste os checkpoints representativos "
            "definidos em checkpoint_statistics.csv."
        )
    )

    parser.add_argument(
        "-pi",
        "--path_images",
        type=str,
        required=True,
        help="Pasta com as imagens de teste.",
    )
    parser.add_argument(
        "-pm",
        "--path_masks",
        type=str,
        required=True,
        help="Pasta com as mascaras de teste.",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        required=True,
        help="Pasta para salvar resultados.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        default="./models_checkpoints",
        help="Pasta dos checkpoints.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./models_checkpoints/checkpoint_statistics.csv",
        help="CSV usado para selecionar o fold representativo.",
    )
    parser.add_argument(
        "--selection_metric",
        choices=["iou", "dice"],
        default="iou",
        help=(
            "Metrica usada para selecionar o fold. "
            "Com 'iou', usa best_iou_closest_fold; com 'dice', "
            "usa best_dice_closest_fold."
        ),
    )
    parser.add_argument(
        "-s",
        "--img_size",
        type=int,
        default=320,
        help="Tamanho das imagens de entrada.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="Tamanho do batch para inferencia.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Dispositivo de inferencia.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="runs/test",
        help="Diretorio base dos logs do TensorBoard.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(ARCHITECTURES.keys()),
        default=list(ARCHITECTURES.keys()),
        help="Arquiteturas a avaliar.",
    )

    return parser


def normalize_architecture(name):
    aliases = {
        "fast-scnn": "fast-scnn",
        "fastscnn": "fast-scnn",
        "bisenet-v2": "bisenetv2",
        "bisenetv2": "bisenetv2",
        "linknet": "linknet",
        "pidnet": "pidnet",
    }
    return aliases.get(str(name).strip().lower())


def load_selection_csv(csv_path, selection_metric):
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"CSV de estatisticas nao encontrado: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    fold_column = (
        "best_iou_closest_fold"
        if selection_metric == "iou"
        else "best_dice_closest_fold"
    )
    value_column = (
        "best_iou_closest_value"
        if selection_metric == "iou"
        else "best_dice_closest_value"
    )

    required_columns = {
        "architecture",
        "best_iou_closest_fold",
        "best_dice_closest_fold",
        "best_iou_closest_value",
        "best_dice_closest_value",
    }

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"O CSV nao possui as colunas necessarias: {sorted(missing)}"
        )

    selections = {}

    for row in df.itertuples(index=False):
        architecture = normalize_architecture(row.architecture)

        if architecture is None:
            continue

        fold = int(getattr(row, fold_column))
        selection_value = float(getattr(row, value_column))

        if fold < 1:
            raise ValueError(
                f"Fold invalido para {architecture}: {fold}"
            )

        selections[architecture] = {
            "fold": fold,
            "selection_metric": selection_metric,
            "selection_value": selection_value,
            "best_iou_closest_fold": int(row.best_iou_closest_fold),
            "best_dice_closest_fold": int(row.best_dice_closest_fold),
            "best_iou_closest_value": float(row.best_iou_closest_value),
            "best_dice_closest_value": float(row.best_dice_closest_value),
        }

    missing_models = [
        model_name for model_name in ARCHITECTURES
        if model_name not in selections
    ]

    if missing_models:
        raise ValueError(
            "Nao foram encontradas no CSV as arquiteturas: "
            + ", ".join(missing_models)
        )

    return selections


def find_checkpoint(checkpoint_path, model_name, fold, checkpoint_type="best_iou"):
    checkpoint_dir = Path(checkpoint_path)

    exact_name = (
        f"{ARCHITECTURES[model_name]['checkpoint_prefix']}"
        f"_fold{fold}_{checkpoint_type}.pt"
    )
    exact_path = checkpoint_dir / exact_name

    if exact_path.is_file():
        return exact_path

    candidates = sorted(
        checkpoint_dir.glob(
            f"*{ARCHITECTURES[model_name]['checkpoint_prefix']}*"
            f"_fold{fold}_{checkpoint_type}.pt"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            f"Nao foi encontrado checkpoint para "
            f"{model_name}, fold {fold}, tipo {checkpoint_type} "
            f"em {checkpoint_dir}."
        )

    if len(candidates) > 1:
        print(
            f"Aviso: mais de um checkpoint encontrado para "
            f"{model_name} fold {fold}. Usando: {candidates[0]}"
        )

    return candidates[0]


def build_model(model_name):
    if model_name == "bisenetv2":
        return BiSeNetV2(n_classes=1)

    if model_name == "fast-scnn":
        return FastSCNN(num_classes=1, aux=True)

    if model_name == "linknet":
        return smp.Linknet(
            encoder_name="tu-mobilenetv4_conv_small",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )

    if model_name == "pidnet":
        return PIDNet(num_classes=1)

    raise ValueError(f"Arquitetura invalida: {model_name}")


def configure_eval_mode(model_name, model):
    if model_name == "bisenetv2":
        model.aux_mode = "eval"
    elif model_name == "pidnet":
        model.augment = False


def calculate_test_metrics(
    sum_loss,
    total_iou_pipe,
    total_dice_pipe,
    total_precision_pipe,
    total_recall_pipe,
    total_iou_bg,
    total_dice_bg,
    total_precision_bg,
    total_recall_bg,
    total_samples,
    total_tp,
    total_fp,
    total_tn,
    total_fn,
):
    eps = 1e-7

    epoch_loss = sum_loss / total_samples

    epoch_miou_pipe = total_iou_pipe / total_samples
    epoch_mdice_pipe = total_dice_pipe / total_samples
    epoch_mprecision_pipe = total_precision_pipe / total_samples
    epoch_mrecall_pipe = total_recall_pipe / total_samples

    epoch_miou_bg = total_iou_bg / total_samples
    epoch_mdice_bg = total_dice_bg / total_samples
    epoch_mprecision_bg = total_precision_bg / total_samples
    epoch_mrecall_bg = total_recall_bg / total_samples

    global_iou_pipe = (
        total_tp / (total_tp + total_fp + total_fn + eps)
    )
    global_dice_pipe = (
        2 * total_tp
        / (2 * total_tp + total_fp + total_fn + eps)
    )
    global_precision_pipe = (
        total_tp / (total_tp + total_fp + eps)
    )
    global_recall_pipe = (
        total_tp / (total_tp + total_fn + eps)
    )

    global_iou_bg = (
        total_tn / (total_tn + total_fn + total_fp + eps)
    )
    global_dice_bg = (
        2 * total_tn
        / (2 * total_tn + total_fn + total_fp + eps)
    )
    global_precision_bg = (
        total_tn / (total_tn + total_fn + eps)
    )
    global_recall_bg = (
        total_tn / (total_tn + total_fp + eps)
    )

    global_mean_iou = (global_iou_pipe + global_iou_bg) / 2
    global_mean_dice = (global_dice_pipe + global_dice_bg) / 2
    global_mean_precision = (
        global_precision_pipe + global_precision_bg
    ) / 2
    global_mean_recall = (
        global_recall_pipe + global_recall_bg
    ) / 2

    global_accuracy = (
        (total_tp + total_tn)
        / (
            total_tp
            + total_fp
            + total_tn
            + total_fn
            + eps
        )
    )

    return {
        "loss": epoch_loss,

        "miou_pipe": epoch_miou_pipe,
        "mdice_pipe": epoch_mdice_pipe,
        "mprecision_pipe": epoch_mprecision_pipe,
        "mrecall_pipe": epoch_mrecall_pipe,

        "miou_bg": epoch_miou_bg,
        "mdice_bg": epoch_mdice_bg,
        "mprecision_bg": epoch_mprecision_bg,
        "mrecall_bg": epoch_mrecall_bg,

        "iou_pipe": global_iou_pipe,
        "dice_pipe": global_dice_pipe,
        "precision_pipe": global_precision_pipe,
        "recall_pipe": global_recall_pipe,

        "iou_bg": global_iou_bg,
        "dice_bg": global_dice_bg,
        "precision_bg": global_precision_bg,
        "recall_bg": global_recall_bg,

        "mean_iou": global_mean_iou,
        "mean_dice": global_mean_dice,
        "mean_precision": global_mean_precision,
        "mean_recall": global_mean_recall,

        "accuracy": global_accuracy,

        "tp": total_tp,
        "fp": total_fp,
        "tn": total_tn,
        "fn": total_fn,
    }


def register_tensorboard_metrics(writer, metrics, step=0):
    split = "Test"

    writer.add_scalar(f"Loss/{split}", metrics["loss"], step)

    writer.add_scalar(
        f"MeanPerImage/Pipe/IoU/{split}",
        metrics["miou_pipe"],
        step,
    )
    writer.add_scalar(
        f"MeanPerImage/Pipe/Dice/{split}",
        metrics["mdice_pipe"],
        step,
    )
    writer.add_scalar(
        f"MeanPerImage/Pipe/Precision/{split}",
        metrics["mprecision_pipe"],
        step,
    )
    writer.add_scalar(
        f"MeanPerImage/Pipe/Recall/{split}",
        metrics["mrecall_pipe"],
        step,
    )

    writer.add_scalar(
        f"MeanPerImage/Background/IoU/{split}",
        metrics["miou_bg"],
        step,
    )
    writer.add_scalar(
        f"MeanPerImage/Background/Dice/{split}",
        metrics["mdice_bg"],
        step,
    )
    writer.add_scalar(
        f"MeanPerImage/Background/Precision/{split}",
        metrics["mprecision_bg"],
        step,
    )
    writer.add_scalar(
        f"MeanPerImage/Background/Recall/{split}",
        metrics["mrecall_bg"],
        step,
    )

    writer.add_scalar(
        f"Global/Pipe/IoU/{split}",
        metrics["iou_pipe"],
        step,
    )
    writer.add_scalar(
        f"Global/Pipe/Dice/{split}",
        metrics["dice_pipe"],
        step,
    )
    writer.add_scalar(
        f"Global/Pipe/Precision/{split}",
        metrics["precision_pipe"],
        step,
    )
    writer.add_scalar(
        f"Global/Pipe/Recall/{split}",
        metrics["recall_pipe"],
        step,
    )

    writer.add_scalar(
        f"Global/Background/IoU/{split}",
        metrics["iou_bg"],
        step,
    )
    writer.add_scalar(
        f"Global/Background/Dice/{split}",
        metrics["dice_bg"],
        step,
    )
    writer.add_scalar(
        f"Global/Background/Precision/{split}",
        metrics["precision_bg"],
        step,
    )
    writer.add_scalar(
        f"Global/Background/Recall/{split}",
        metrics["recall_bg"],
        step,
    )

    writer.add_scalar(
        f"Global/ClassMean/IoU/{split}",
        metrics["mean_iou"],
        step,
    )
    writer.add_scalar(
        f"Global/ClassMean/Dice/{split}",
        metrics["mean_dice"],
        step,
    )
    writer.add_scalar(
        f"Global/ClassMean/Precision/{split}",
        metrics["mean_precision"],
        step,
    )
    writer.add_scalar(
        f"Global/ClassMean/Recall/{split}",
        metrics["mean_recall"],
        step,
    )

    writer.add_scalar(
        f"Global/Accuracy/{split}",
        metrics["accuracy"],
        step,
    )

    writer.add_scalar(
        f"ConfusionMatrix/TP/{split}",
        metrics["tp"],
        step,
    )
    writer.add_scalar(
        f"ConfusionMatrix/FP/{split}",
        metrics["fp"],
        step,
    )
    writer.add_scalar(
        f"ConfusionMatrix/TN/{split}",
        metrics["tn"],
        step,
    )
    writer.add_scalar(
        f"ConfusionMatrix/FN/{split}",
        metrics["fn"],
        step,
    )


def save_confusion_matrix(metrics, model_name, output_dir):
    cm = np.array(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ],
        dtype=np.float64,
    )

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums != 0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        ax=axes[0],
        xticklabels=["Fundo", "Objeto"],
        yticklabels=["Fundo", "Objeto"],
    )
    axes[0].set_title(f"{model_name} - Teste - Matriz Absoluta")
    axes[0].set_xlabel("Predito")
    axes[0].set_ylabel("Real")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".4f",
        cmap="Blues",
        ax=axes[1],
        xticklabels=["Fundo", "Objeto"],
        yticklabels=["Fundo", "Objeto"],
    )
    axes[1].set_title(f"{model_name} - Teste - Matriz Normalizada")
    axes[1].set_xlabel("Predito")
    axes[1].set_ylabel("Real")

    plt.tight_layout()

    output_path = Path(output_dir) / f"{model_name}_test_confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def evaluate_test(
    model_name,
    model,
    checkpoint_path,
    test_loader,
    device,
    img_size,
    writer,
    output_dir,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )

    model.to(device)
    configure_eval_mode(model_name, model)
    model.eval()

    criterion_focal = smp.losses.FocalLoss(
        mode="binary",
        alpha=0.5,
        gamma=2.0,
    )
    criterion_dice = smp.losses.DiceLoss(mode="binary")

    sum_loss = 0.0

    total_iou_pipe = 0.0
    total_dice_pipe = 0.0
    total_precision_pipe = 0.0
    total_recall_pipe = 0.0

    total_iou_bg = 0.0
    total_dice_bg = 0.0
    total_precision_bg = 0.0
    total_recall_bg = 0.0

    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    total_samples = 0
    total_inference_time = 0.0

    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)
    cpu_usages = []
    peak_ram_mb = process.memory_info().rss / (1024 ** 2)

    print(f"[{model_name}][test] iniciando inferencia...")

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader, start=1):
            images = images.float().to(device)
            masks = masks.float().to(device)

            current_ram_mb = process.memory_info().rss / (1024 ** 2)
            peak_ram_mb = max(peak_ram_mb, current_ram_mb)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            raw_outputs = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()

            total_inference_time += time.perf_counter() - start_time

            cpu_usages.append(process.cpu_percent(interval=None))

            current_ram_mb = process.memory_info().rss / (1024 ** 2)
            peak_ram_mb = max(peak_ram_mb, current_ram_mb)

            loss = val_losses_composition(
                model_name,
                raw_outputs,
                masks,
                img_size,
                criterion_focal,
                criterion_dice,
            )

            sum_loss += loss.item() * images.shape[0]

            if model_name == "linknet":
                outputs = raw_outputs.float()

            elif model_name == "pidnet":
                outputs = F.interpolate(
                    raw_outputs.float(),
                    size=(img_size, img_size),
                    mode="bilinear",
                    align_corners=True,
                ).float()

            else:
                outputs = raw_outputs[0].float()

            preds = (
                torch.sigmoid(outputs) > 0.5
            ).float()

            batch_metrics = calculate_batch_metrics(
                preds,
                masks,
            )

            total_iou_pipe += batch_metrics["iou_pipe_sum"]
            total_dice_pipe += batch_metrics["dice_pipe_sum"]
            total_precision_pipe += batch_metrics["precision_pipe_sum"]
            total_recall_pipe += batch_metrics["recall_pipe_sum"]

            total_iou_bg += batch_metrics["iou_bg_sum"]
            total_dice_bg += batch_metrics["dice_bg_sum"]
            total_precision_bg += batch_metrics["precision_bg_sum"]
            total_recall_bg += batch_metrics["recall_bg_sum"]

            total_tp += batch_metrics["tp"]
            total_fp += batch_metrics["fp"]
            total_tn += batch_metrics["tn"]
            total_fn += batch_metrics["fn"]

            total_samples += batch_metrics["samples"]

            print(
                f"[{model_name}][test] batch "
                f"{batch_idx}/{len(test_loader)}",
                end="\r",
            )

    metrics = calculate_test_metrics(
        sum_loss=sum_loss,
        total_iou_pipe=total_iou_pipe,
        total_dice_pipe=total_dice_pipe,
        total_precision_pipe=total_precision_pipe,
        total_recall_pipe=total_recall_pipe,
        total_iou_bg=total_iou_bg,
        total_dice_bg=total_dice_bg,
        total_precision_bg=total_precision_bg,
        total_recall_bg=total_recall_bg,
        total_samples=total_samples,
        total_tp=total_tp,
        total_fp=total_fp,
        total_tn=total_tn,
        total_fn=total_fn,
    )

    register_tensorboard_metrics(
        writer,
        metrics,
        step=0,
    )

    inference_time = total_inference_time
    fps = (
        total_samples / inference_time
        if inference_time > 0
        else float("nan")
    )

    avg_cpu_usage = (
        sum(cpu_usages) / len(cpu_usages)
        if cpu_usages
        else 0.0
    )

    metrics["fps"] = fps
    metrics["peak_ram_mb"] = peak_ram_mb
    metrics["cpu_usage_percent"] = avg_cpu_usage

    writer.add_scalar("Performance/FPS/Test", fps, 0)
    writer.add_scalar(
        "Performance/Peak_RAM_MB/Test",
        peak_ram_mb,
        0,
    )
    writer.add_scalar(
        "Performance/CPU_Usage_Percent/Test",
        avg_cpu_usage,
        0,
    )

    save_confusion_matrix(
        metrics,
        ARCHITECTURES[model_name]["display_name"],
        output_dir,
    )

    print(
        f"\n[{model_name}] "
        f"Test Pipe Global IoU: {metrics['iou_pipe']:.4f} | "
        f"Test Pipe Global Dice: {metrics['dice_pipe']:.4f} | "
        f"Test BG Global IoU: {metrics['iou_bg']:.4f} | "
        f"Test BG Global Dice: {metrics['dice_bg']:.4f} | "
        f"FPS: {metrics['fps']:.2f} | "
        f"RAM Pico: {metrics['peak_ram_mb']:.1f} MB | "
        f"CPU: {metrics['cpu_usage_percent']:.1f}%"
    )

    return metrics, checkpoint


def save_summary_csv(results, output_dir):
    rows = []

    for model_name, result in results.items():
        row = {
            "Modelo": ARCHITECTURES[model_name]["display_name"],
            "Fold": result["fold"],
            "Checkpoint": result["checkpoint"],
            "Checkpoint_Epoch": result["checkpoint_epoch"],
            "Checkpoint_IoU": result["checkpoint_iou"],
            "Checkpoint_Dice": result["checkpoint_dice"],
            "Test_Loss": result["metrics"]["loss"],

            "Test_MeanPerImage_Pipe_IoU":
                result["metrics"]["miou_pipe"],
            "Test_MeanPerImage_Pipe_Dice":
                result["metrics"]["mdice_pipe"],
            "Test_MeanPerImage_Pipe_Precision":
                result["metrics"]["mprecision_pipe"],
            "Test_MeanPerImage_Pipe_Recall":
                result["metrics"]["mrecall_pipe"],

            "Test_MeanPerImage_Background_IoU":
                result["metrics"]["miou_bg"],
            "Test_MeanPerImage_Background_Dice":
                result["metrics"]["mdice_bg"],
            "Test_MeanPerImage_Background_Precision":
                result["metrics"]["mprecision_bg"],
            "Test_MeanPerImage_Background_Recall":
                result["metrics"]["mrecall_bg"],

            "Test_Global_Pipe_IoU":
                result["metrics"]["iou_pipe"],
            "Test_Global_Pipe_Dice":
                result["metrics"]["dice_pipe"],
            "Test_Global_Pipe_Precision":
                result["metrics"]["precision_pipe"],
            "Test_Global_Pipe_Recall":
                result["metrics"]["recall_pipe"],

            "Test_Global_Background_IoU":
                result["metrics"]["iou_bg"],
            "Test_Global_Background_Dice":
                result["metrics"]["dice_bg"],
            "Test_Global_Background_Precision":
                result["metrics"]["precision_bg"],
            "Test_Global_Background_Recall":
                result["metrics"]["recall_bg"],

            "Test_Global_ClassMean_IoU":
                result["metrics"]["mean_iou"],
            "Test_Global_ClassMean_Dice":
                result["metrics"]["mean_dice"],
            "Test_Global_ClassMean_Precision":
                result["metrics"]["mean_precision"],
            "Test_Global_ClassMean_Recall":
                result["metrics"]["mean_recall"],

            "Test_Global_Accuracy":
                result["metrics"]["accuracy"],

            "Test_TP": result["metrics"]["tp"],
            "Test_FP": result["metrics"]["fp"],
            "Test_TN": result["metrics"]["tn"],
            "Test_FN": result["metrics"]["fn"],

            "FPS": result["metrics"]["fps"],
            "Peak_RAM_MB": result["metrics"]["peak_ram_mb"],
            "CPU_Usage_Percent": result["metrics"]["cpu_usage_percent"],
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    output_path = Path(output_dir) / "test_results.csv"
    df.to_csv(output_path, index=False)

    return df


def plot_comparison(results_df, output_dir):
    metric_groups = [
        (
            "Teste - Mean Per Image",
            [
                "Test_MeanPerImage_Pipe_IoU",
                "Test_MeanPerImage_Pipe_Dice",
                "Test_MeanPerImage_Pipe_Precision",
                "Test_MeanPerImage_Pipe_Recall",
                "Test_MeanPerImage_Background_IoU",
                "Test_MeanPerImage_Background_Dice",
                "Test_MeanPerImage_Background_Precision",
                "Test_MeanPerImage_Background_Recall",
            ],
        ),
        (
            "Teste - Global",
            [
                "Test_Global_Pipe_IoU",
                "Test_Global_Pipe_Dice",
                "Test_Global_Pipe_Precision",
                "Test_Global_Pipe_Recall",
                "Test_Global_Background_IoU",
                "Test_Global_Background_Dice",
                "Test_Global_Background_Precision",
                "Test_Global_Background_Recall",
                "Test_Global_ClassMean_IoU",
                "Test_Global_ClassMean_Dice",
                "Test_Global_ClassMean_Precision",
                "Test_Global_ClassMean_Recall",
                "Test_Global_Accuracy",
            ],
        ),
    ]

    for title, columns in metric_groups:
        compact_names = [
            column.replace("Test_MeanPerImage_", "")
            .replace("Test_Global_", "")
        for column in columns]

        values = results_df[columns].T
        values.index = compact_names

        ax = values.plot(
            kind="bar",
            figsize=(18, 9),
        )

        ax.set_title(title)
        ax.set_ylabel("Valor")
        ax.set_xlabel("Metrica")
        ax.set_ylim(0, 1.0)
        ax.legend(
            title="Modelo",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        plt.tight_layout()

        filename = (
            "test_mean_per_image_comparison.png"
            if "Mean Per Image" in title
            else "test_global_comparison.png"
        )

        plt.savefig(
            Path(output_dir) / filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    loss_df = results_df.set_index("Modelo")[["Test_Loss"]]

    ax = loss_df.plot(
        kind="bar",
        figsize=(10, 6),
        legend=False,
    )
    ax.set_title("Teste - Loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Modelo")
    plt.tight_layout()

    plt.savefig(
        Path(output_dir) / "test_loss_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    performance_df = results_df.set_index("Modelo")[
        ["FPS", "Peak_RAM_MB", "CPU_Usage_Percent"]
    ]

    axes = performance_df.plot(
        kind="bar",
        subplots=True,
        layout=(1, 3),
        figsize=(18, 6),
        legend=False,
    )

    for ax, title in zip(
        axes.flatten(),
        ["FPS", "Peak RAM (MB)", "CPU Usage (%)"],
    ):
        ax.set_title(title)
        ax.set_xlabel("Modelo")

    plt.suptitle("Teste - Performance de Inferencia")
    plt.tight_layout(rect=(0, 0, 1, 0.94))

    plt.savefig(
        Path(output_dir) / "test_performance_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main(
    path_images,
    path_masks,
    out_dir,
    checkpoint_path,
    csv_path,
    selection_metric,
    img_size,
    batch_size,
    device,
    log_dir,
    models,
):
    torch.set_num_threads(1)
    print(
        f"PyTorch usando {torch.get_num_threads()} thread(s)."
    )

    if device == "auto":
        selected_device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        selected_device = torch.device(device)

    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Foi solicitado CUDA, mas CUDA nao esta disponivel."
        )

    print(f"Dispositivo de inferencia: {selected_device}")
    print(f"Selecao de fold baseada em: best_{selection_metric}")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selections = load_selection_csv(
        csv_path,
        selection_metric,
    )

    images_dir = Path(path_images)
    masks_dir = Path(path_masks)

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Pasta de imagens nao encontrada: {images_dir}"
        )

    if not masks_dir.is_dir():
        raise FileNotFoundError(
            f"Pasta de mascaras nao encontrada: {masks_dir}"
        )

    image_paths = []
    mask_paths = []

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
        }:
            continue

        mask_path = masks_dir / f"{image_path.stem}_label.png"

        if mask_path.exists():
            image_paths.append(str(image_path))
            mask_paths.append(str(mask_path))

    if not image_paths:
        raise RuntimeError(
            "Nenhum par imagem/mascara valido foi encontrado."
        )

    print(f"Total de imagens de teste: {len(image_paths)}")

    test_dataset = SubPipeMiniDataset(
        image_paths,
        mask_paths,
        transforms=transforms_val(img_size),
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(selected_device.type == "cuda"),
        drop_last=False,
    )

    results = {}
    selection_rows = []

    for model_name in models:
        selection = selections[model_name]
        fold = selection["fold"]

        checkpoint_path_model = find_checkpoint(
            checkpoint_path,
            model_name,
            fold,
            checkpoint_type="best_iou",
        )

        print(f"\n===== {ARCHITECTURES[model_name]['display_name']} =====")
        print(f"Fold selecionado: {fold}")
        print(f"Checkpoint: {checkpoint_path_model}")
        print(
            f"Valor usado na selecao ({selection_metric}): "
            f"{selection['selection_value']:.6f}"
        )

        checkpoint_preview = torch.load(
            checkpoint_path_model,
            map_location="cpu",
            weights_only=False,
        )

        print(
            f"Epoch salva: {checkpoint_preview['epoch']} | "
            f"IoU checkpoint: {checkpoint_preview['iou']:.6f} | "
            f"Dice checkpoint: {checkpoint_preview['dice']:.6f}"
        )

        model = build_model(model_name)

        model_log_dir = (
            Path(log_dir)
            / ARCHITECTURES[model_name]["display_name"].lower()
        )
        model_log_dir.mkdir(parents=True, exist_ok=True)

        writer = SummaryWriter(log_dir=str(model_log_dir))

        metrics, checkpoint = evaluate_test(
            model_name=model_name,
            model=model,
            checkpoint_path=checkpoint_path_model,
            test_loader=test_loader,
            device=selected_device,
            img_size=img_size,
            writer=writer,
            output_dir=output_dir,
        )

        writer.close()

        results[model_name] = {
            "fold": fold,
            "checkpoint": str(checkpoint_path_model),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_iou": float(checkpoint["iou"]),
            "checkpoint_dice": float(checkpoint["dice"]),
            "metrics": metrics,
        }

        selection_rows.append(
            {
                "Modelo": ARCHITECTURES[model_name]["display_name"],
                "Fold": fold,
                "SelectionMetric": selection_metric,
                "SelectionValue": selection["selection_value"],
                "Checkpoint": str(checkpoint_path_model),
                "CheckpointEpoch": int(checkpoint["epoch"]),
                "CheckpointIoU": float(checkpoint["iou"]),
                "CheckpointDice": float(checkpoint["dice"]),
            }
        )

        del model
        gc.collect()

        if selected_device.type == "cuda":
            torch.cuda.empty_cache()

    results_df = save_summary_csv(
        results,
        output_dir,
    )

    pd.DataFrame(selection_rows).to_csv(
        output_dir / "selected_checkpoints.csv",
        index=False,
    )

    plot_comparison(
        results_df,
        output_dir,
    )

    print(
        f"\nFinalizado. Resultados salvos em: {output_dir}"
    )


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    main(**vars(args))
