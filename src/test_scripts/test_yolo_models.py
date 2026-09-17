import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import gc
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import yaml
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

from train_scripts.train_utils import calculate_batch_metrics


TEST_YAML = "/home/Cake/Documentos/UFRB/TCC/MarinaPipeDivided/test-data.yaml"

MODELS_TO_TEST = [
    {
        "name": "YOLOv8n-seg",
        "checkpoint": "runs/segment/runs/yolo8_kfold_united_dataset/fold5/weights/best.pt",
    },
    {
        "name": "YOLO11n-seg",
        "checkpoint": "runs/segment/runs/yolo11_kfold_united_dataset/fold1/weights/best.pt",
    },
    {
        "name": "YOLO26n-seg",
        "checkpoint": "runs/segment/runs/yolo26_kfold_united_dataset/fold5/weights/best.pt",
    },
]

IMGSZ = 320
CONF_THRESHOLD = 0.25
IOU_NMS = 0.70
DEVICE = "cpu"

OUTPUT_DIR = "test_results_yolo_marinapipe"
LOG_DIR = "runs/test_yolo/marinapipe"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_test_images_from_yaml(yaml_path):
    with open(Path(yaml_path), "r") as f:
        data = yaml.safe_load(f)

    if "test" not in data:
        raise ValueError("O test-data.yaml nao possui a chave 'test'.")

    test_path = Path(data["test"])

    if not test_path.is_absolute():
        root = data.get("path", "")
        if root:
            test_path = Path(root) / test_path

    test_path = test_path.resolve()

    if not test_path.exists():
        raise FileNotFoundError(f"Diretorio de teste nao encontrado: {test_path}")

    masks_path = Path(
        str(test_path).replace(os.sep + "images", os.sep + "masks")
    )

    image_paths = sorted(
        p for p in test_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(f"Nenhuma imagem encontrada em {test_path}")

    print(f"Diretorio de teste: {test_path}")
    print(f"Diretorio de mascaras: {masks_path}")
    print(f"Imagens encontradas: {len(image_paths)}")

    return image_paths, masks_path


def instances_to_semantic_mask(result, height, width):
    """Une as instancias previstas pelo YOLO em uma mascara binaria [1, H, W]."""
    empty = torch.zeros((1, height, width), dtype=torch.float32)

    if result.masks is None:
        return empty

    masks = result.masks.data

    if masks is None or masks.numel() == 0:
        return empty

    masks = masks.float()

    if masks.shape[-2:] != (height, width):
        masks = torch.nn.functional.interpolate(
            masks.unsqueeze(1),
            size=(height, width),
            mode="nearest",
        ).squeeze(1)

    return torch.any(masks > 0.5, dim=0).float().unsqueeze(0)


def calculate_test_metrics(
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

    epoch_miou_pipe = total_iou_pipe / total_samples
    epoch_mdice_pipe = total_dice_pipe / total_samples
    epoch_mprecision_pipe = total_precision_pipe / total_samples
    epoch_mrecall_pipe = total_recall_pipe / total_samples

    epoch_miou_bg = total_iou_bg / total_samples
    epoch_mdice_bg = total_dice_bg / total_samples
    epoch_mprecision_bg = total_precision_bg / total_samples
    epoch_mrecall_bg = total_recall_bg / total_samples

    global_iou_pipe = total_tp / (total_tp + total_fp + total_fn + eps)
    global_dice_pipe = 2 * total_tp / (2 * total_tp + total_fp + total_fn + eps)
    global_precision_pipe = total_tp / (total_tp + total_fp + eps)
    global_recall_pipe = total_tp / (total_tp + total_fn + eps)

    global_iou_bg = total_tn / (total_tn + total_fn + total_fp + eps)
    global_dice_bg = 2 * total_tn / (2 * total_tn + total_fn + total_fp + eps)
    global_precision_bg = total_tn / (total_tn + total_fn + eps)
    global_recall_bg = total_tn / (total_tn + total_fp + eps)

    global_mean_iou = (global_iou_pipe + global_iou_bg) / 2
    global_mean_dice = (global_dice_pipe + global_dice_bg) / 2
    global_mean_precision = (global_precision_pipe + global_precision_bg) / 2
    global_mean_recall = (global_recall_pipe + global_recall_bg) / 2

    global_accuracy = (total_tp + total_tn) / (
        total_tp + total_fp + total_tn + total_fn + eps
    )

    return {
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

    writer.add_scalar(f"MeanPerImage/Pipe/IoU/{split}", metrics["miou_pipe"], step)
    writer.add_scalar(f"MeanPerImage/Pipe/Dice/{split}", metrics["mdice_pipe"], step)
    writer.add_scalar(f"MeanPerImage/Pipe/Precision/{split}", metrics["mprecision_pipe"], step)
    writer.add_scalar(f"MeanPerImage/Pipe/Recall/{split}", metrics["mrecall_pipe"], step)

    writer.add_scalar(f"MeanPerImage/Background/IoU/{split}", metrics["miou_bg"], step)
    writer.add_scalar(f"MeanPerImage/Background/Dice/{split}", metrics["mdice_bg"], step)
    writer.add_scalar(f"MeanPerImage/Background/Precision/{split}", metrics["mprecision_bg"], step)
    writer.add_scalar(f"MeanPerImage/Background/Recall/{split}", metrics["mrecall_bg"], step)

    writer.add_scalar(f"Global/Pipe/IoU/{split}", metrics["iou_pipe"], step)
    writer.add_scalar(f"Global/Pipe/Dice/{split}", metrics["dice_pipe"], step)
    writer.add_scalar(f"Global/Pipe/Precision/{split}", metrics["precision_pipe"], step)
    writer.add_scalar(f"Global/Pipe/Recall/{split}", metrics["recall_pipe"], step)

    writer.add_scalar(f"Global/Background/IoU/{split}", metrics["iou_bg"], step)
    writer.add_scalar(f"Global/Background/Dice/{split}", metrics["dice_bg"], step)
    writer.add_scalar(f"Global/Background/Precision/{split}", metrics["precision_bg"], step)
    writer.add_scalar(f"Global/Background/Recall/{split}", metrics["recall_bg"], step)

    writer.add_scalar(f"Global/ClassMean/IoU/{split}", metrics["mean_iou"], step)
    writer.add_scalar(f"Global/ClassMean/Dice/{split}", metrics["mean_dice"], step)
    writer.add_scalar(f"Global/ClassMean/Precision/{split}", metrics["mean_precision"], step)
    writer.add_scalar(f"Global/ClassMean/Recall/{split}", metrics["mean_recall"], step)

    writer.add_scalar(f"Global/Accuracy/{split}", metrics["accuracy"], step)

    writer.add_scalar(f"ConfusionMatrix/TP/{split}", metrics["tp"], step)
    writer.add_scalar(f"ConfusionMatrix/FP/{split}", metrics["fp"], step)
    writer.add_scalar(f"ConfusionMatrix/TN/{split}", metrics["tn"], step)
    writer.add_scalar(f"ConfusionMatrix/FN/{split}", metrics["fn"], step)


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
    plt.savefig(
        Path(output_dir) / f"{model_name}_test_confusion_matrix.png",
        dpi=300,
    )
    plt.close()


def evaluate_test(
    model_name,
    checkpoint_path,
    image_paths,
    masks_path,
    device,
    writer,
    output_dir,
):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {checkpoint_path}")

    model = YOLO(str(checkpoint_path))

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

    for index, image_path in enumerate(image_paths, start=1):
        mask_path = masks_path / f"{image_path.stem}_label.png"

        if not mask_path.exists():
            print(f"\nAviso: mascara nao encontrada: {mask_path}")
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            print(f"\nAviso: erro ao carregar imagem: {image_path}")
            continue

        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if gt_mask is None:
            print(f"\nAviso: erro ao carregar mascara: {mask_path}")
            continue

        height, width = gt_mask.shape
        masks = torch.from_numpy(
            (gt_mask > 0).astype(np.float32)
        ).view(1, 1, height, width)

        current_ram_mb = process.memory_info().rss / (1024 ** 2)
        peak_ram_mb = max(peak_ram_mb, current_ram_mb)

        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        results = list(model.predict(
            source=image,
            imgsz=IMGSZ,
            conf=CONF_THRESHOLD,
            iou=IOU_NMS,
            device=device,
            retina_masks=True,
            verbose=False,
        ))

        if device == "cuda":
            torch.cuda.synchronize()

        total_inference_time += time.perf_counter() - start_time

        cpu_usages.append(process.cpu_percent(interval=None))

        current_ram_mb = process.memory_info().rss / (1024 ** 2)
        peak_ram_mb = max(peak_ram_mb, current_ram_mb)

        preds = instances_to_semantic_mask(
            results[0], height, width
        ).unsqueeze(0)

        batch_metrics = calculate_batch_metrics(preds, masks)

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
            f"[{model_name}][test] imagem {index}/{len(image_paths)}",
            end="\r",
        )

    if total_samples == 0:
        raise RuntimeError("Nenhuma imagem pode ser avaliada.")

    metrics = calculate_test_metrics(
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

    register_tensorboard_metrics(writer, metrics, step=0)

    fps = (
        total_samples / total_inference_time
        if total_inference_time > 0
        else float("nan")
    )
    avg_cpu_usage = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0.0

    metrics["fps"] = fps
    metrics["peak_ram_mb"] = peak_ram_mb
    metrics["cpu_usage_percent"] = avg_cpu_usage

    writer.add_scalar("Performance/FPS/Test", fps, 0)
    writer.add_scalar("Performance/Peak_RAM_MB/Test", peak_ram_mb, 0)
    writer.add_scalar("Performance/CPU_Usage_Percent/Test", avg_cpu_usage, 0)

    save_confusion_matrix(metrics, model_name, output_dir)

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

    del model
    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()

    return metrics


def save_summary_csv(results, output_dir):
    rows = []

    for model_name, result in results.items():
        rows.append({
            "Modelo": model_name,
            "Checkpoint": result["checkpoint"],

            "Test_MeanPerImage_Pipe_IoU": result["metrics"]["miou_pipe"],
            "Test_MeanPerImage_Pipe_Dice": result["metrics"]["mdice_pipe"],
            "Test_MeanPerImage_Pipe_Precision": result["metrics"]["mprecision_pipe"],
            "Test_MeanPerImage_Pipe_Recall": result["metrics"]["mrecall_pipe"],

            "Test_MeanPerImage_Background_IoU": result["metrics"]["miou_bg"],
            "Test_MeanPerImage_Background_Dice": result["metrics"]["mdice_bg"],
            "Test_MeanPerImage_Background_Precision": result["metrics"]["mprecision_bg"],
            "Test_MeanPerImage_Background_Recall": result["metrics"]["mrecall_bg"],

            "Test_Global_Pipe_IoU": result["metrics"]["iou_pipe"],
            "Test_Global_Pipe_Dice": result["metrics"]["dice_pipe"],
            "Test_Global_Pipe_Precision": result["metrics"]["precision_pipe"],
            "Test_Global_Pipe_Recall": result["metrics"]["recall_pipe"],

            "Test_Global_Background_IoU": result["metrics"]["iou_bg"],
            "Test_Global_Background_Dice": result["metrics"]["dice_bg"],
            "Test_Global_Background_Precision": result["metrics"]["precision_bg"],
            "Test_Global_Background_Recall": result["metrics"]["recall_bg"],

            "Test_Global_ClassMean_IoU": result["metrics"]["mean_iou"],
            "Test_Global_ClassMean_Dice": result["metrics"]["mean_dice"],
            "Test_Global_ClassMean_Precision": result["metrics"]["mean_precision"],
            "Test_Global_ClassMean_Recall": result["metrics"]["mean_recall"],

            "Test_Global_Accuracy": result["metrics"]["accuracy"],

            "Test_TP": result["metrics"]["tp"],
            "Test_FP": result["metrics"]["fp"],
            "Test_TN": result["metrics"]["tn"],
            "Test_FN": result["metrics"]["fn"],

            "FPS": result["metrics"]["fps"],
            "Peak_RAM_MB": result["metrics"]["peak_ram_mb"],
            "CPU_Usage_Percent": result["metrics"]["cpu_usage_percent"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / "test_results.csv", index=False)

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

    indexed_df = results_df.set_index("Modelo")

    for title, columns in metric_groups:
        compact_names = [
            column.replace("Test_MeanPerImage_", "").replace("Test_Global_", "")
            for column in columns
        ]

        values = indexed_df[columns].T
        values.index = compact_names

        ax = values.plot(kind="bar", figsize=(18, 9))
        ax.set_title(title)
        ax.set_ylabel("Valor")
        ax.set_xlabel("Metrica")
        ax.set_ylim(0, 1.0)
        ax.legend(title="Modelo", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

        filename = (
            "test_mean_per_image_comparison.png"
            if "Mean Per Image" in title
            else "test_global_comparison.png"
        )

        plt.savefig(Path(output_dir) / filename, dpi=300, bbox_inches="tight")
        plt.close()

    performance_df = indexed_df[["FPS", "Peak_RAM_MB", "CPU_Usage_Percent"]]

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


def main():
    torch.set_num_threads(1)
    print(f"PyTorch usando {torch.get_num_threads()} thread(s).")

    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Foi solicitado CUDA, mas CUDA nao esta disponivel.")

    print(f"Dispositivo de inferencia: {DEVICE}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths, masks_path = load_test_images_from_yaml(TEST_YAML)

    results = {}

    for config in MODELS_TO_TEST:
        model_name = config["name"]

        print(f"\n===== {model_name} =====")
        print(f"Checkpoint: {config['checkpoint']}")

        model_log_dir = Path(LOG_DIR) / model_name.lower()
        model_log_dir.mkdir(parents=True, exist_ok=True)

        writer = SummaryWriter(log_dir=str(model_log_dir))

        metrics = evaluate_test(
            model_name=model_name,
            checkpoint_path=config["checkpoint"],
            image_paths=image_paths,
            masks_path=masks_path,
            device=DEVICE,
            writer=writer,
            output_dir=output_dir,
        )

        writer.close()

        results[model_name] = {
            "checkpoint": config["checkpoint"],
            "metrics": metrics,
        }

    results_df = save_summary_csv(results, output_dir)
    plot_comparison(results_df, output_dir)

    print(f"\nFinalizado. Resultados salvos em: {output_dir}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
