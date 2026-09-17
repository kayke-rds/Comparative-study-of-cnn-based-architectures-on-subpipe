import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import segmentation_models_pytorch as smp
from pathlib import Path
from torch.utils.data import DataLoader
from data_utils.submini_dataset import SubPipeMiniDataset, transforms_train, transforms_val
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold

from semantic_seg_models.bisenetv2.bisenetv2 import BiSeNetV2
from semantic_seg_models.fast_scnn_tramac.models.fast_scnn import FastSCNN
from semantic_seg_models.pidnet.pidnet import PIDNet

MODELS = ["bisenetv2", "fast-scnn", "linknet", "pidnet"]


def parse_args():

    parser = argparse.ArgumentParser(description="Script com flags e valores.")

    parser.add_argument(
        "-s", "--img_size", type=int, default=320, help="Altura da imagem (padrão: 320x320)"
    )
    parser.add_argument(
        "-n", "--epochs", type=int, default=10, help="Número de épocas (padrão: 10)"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=16, help="Tamanho do batch (padrão: 16)"
    )
    parser.add_argument(
        "-l", "--lr", type=float, default=1e-4, help="Taxa de aprendizado (padrão: 1e-4)"
    )
    parser.add_argument(
        "-c", "--checkpoint_path", type=str, default="./models_checkpoints", help="Caminho para salvar checkpoints (padrão: ./models_checkpoints)"
    )
    parser.add_argument(
        "-m", "--model_name", type=str, help="Nome do modelo. Valores possíveis: bisenetv2, fast-scnn, linknet e pidnet"
    )
    parser.add_argument(
        "-pi", "--path_images", type=str, help="Caminho para as imagens"
    )
    parser.add_argument(
        "-pm", "--path_masks", type=str, help="Caminho para as masks"
    )
    parser.add_argument(
        "-ld", "--log_dir", type=str, help="Nome para o diretório de logs do TensorBoard"
    )

    return parser


def extrair_bordas(mascara_original):
    """
    Gera ground-truth de borda para a D branch da PIDNet.

    Entrada:
        mascara_original: [B, H, W] ou [B, 1, H, W]
        valores 0/1

    Saída:
        [B, 1, H//8, W//8]
    """

    if mascara_original.dim() == 3:
        x = mascara_original.unsqueeze(1).float()
    elif mascara_original.dim() == 4:
        x = mascara_original.float()
    else:
        raise ValueError(
            f"Máscara deve possuir 3 ou 4 dimensões, recebeu {mascara_original.shape}"
        )

    kernel = torch.tensor(
        [[[-1., -1., -1.],
          [-1.,  8., -1.],
          [-1., -1., -1.]]],
        device=x.device,
        dtype=x.dtype
    ).unsqueeze(0)

    gradiente = F.conv2d(
        x,
        kernel,
        padding=1
    )

    borda = (gradiente != 0).float()

    borda = F.max_pool2d(
        borda,
        kernel_size=8,
        stride=8
    )

    return borda


class BoundaryLoss(nn.Module):
    def __init__(self, coeff_bce=20.0):
        super().__init__()
        self.coeff_bce = coeff_bce

    def forward(self, bd_pre, bd_gt):
        """
        bd_pre:
            [B, 1, H, W] logits da D branch

        bd_gt:
            [B, 1, H, W] ou [B, H, W]
            ground-truth binário de borda
        """

        if bd_gt.dim() == 3:
            bd_gt = bd_gt.unsqueeze(1)

        bd_gt = bd_gt.float()

        log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
        target = bd_gt.permute(0, 2, 3, 1).contiguous().view(1, -1)

        pos_index = (target == 1)
        neg_index = (target == 0)

        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        total = pos_num + neg_num

        if pos_num == 0 or neg_num == 0:
            return self.coeff_bce * F.binary_cross_entropy_with_logits(
                log_p,
                target,
                reduction='mean'
            )

        weight = torch.zeros_like(log_p)

        # Balanceamento usado no PIDNet oficial
        weight[pos_index] = neg_num.float() / total.float()
        weight[neg_index] = pos_num.float() / total.float()

        loss = F.binary_cross_entropy_with_logits(
            log_p,
            target,
            weight=weight,
            reduction='mean'
        )

        return self.coeff_bce * loss


def train_losses_composition(model_name, outputs, masks, img_size, criterion_focal, criterion_dice, criterion_boundary):

    match model_name:
        case "bisenetv2":
            loss_focal = criterion_focal(outputs[0].float(), masks) + 0.3*(criterion_focal(outputs[1].float(), masks) + criterion_focal(outputs[2].float(), masks) + criterion_focal(outputs[3].float(), masks) + criterion_focal(outputs[4].float(), masks))
            loss_dice = criterion_dice(outputs[0].float(), masks) + 0.3*(criterion_dice(outputs[1].float(), masks) + criterion_dice(outputs[2].float(), masks) + criterion_dice(outputs[3].float(), masks) + criterion_dice(outputs[4].float(), masks))
            loss = (0.6*loss_focal + 0.4*loss_dice)
        case "fast-scnn":
            loss_focal = 0.7*criterion_focal(outputs[0].float(), masks) + 0.3*criterion_focal(outputs[1].float(), masks)
            loss_dice = 0.7*criterion_dice(outputs[0].float(), masks) + 0.3*criterion_dice(outputs[1].float(), masks)
            loss = (0.6*loss_focal + 0.4*loss_dice)
        case "linknet":
            loss_focal = criterion_focal(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = (0.6*loss_focal + 0.4*loss_dice)
        case "pidnet":
            out0 = F.interpolate(
                outputs[0].float(),
                size=(img_size, img_size),
                mode='bilinear',
                align_corners=True
            )

            out1 = F.interpolate(
                outputs[1].float(),
                size=(img_size, img_size),
                mode='bilinear',
                align_corners=True
            )

            out2 = outputs[2].float()

            target_boundary = extrair_bordas(masks)

            loss_main = (
                0.6 * criterion_focal(out1, masks)
                + 0.4 * criterion_dice(out1, masks)
            )

            loss_aux = (
                0.6 * criterion_focal(out0, masks)
                + 0.4 * criterion_dice(out0, masks)
            )

            loss_boundary = criterion_boundary(
                out2,
                target_boundary
            )

            loss = (
                1.0 * loss_main
                + 0.4 * loss_aux
                + loss_boundary
            )
        case _:
            raise ValueError(
                f"Erro: Nome de modelo {model_name} inválido. "
                f"Os nomes possíveis são: {' '.join(MODELS)}"
            )

    return loss


def val_losses_composition(model_name, outputs, masks, img_size, criterion_focal, criterion_dice):
    criterion_focal = smp.losses.FocalLoss(mode='binary', alpha=0.5, gamma=2.0)
    criterion_dice = smp.losses.DiceLoss(mode='binary')

    match model_name:
        case "bisenetv2":
            outputs = outputs[0].float()

            loss_focal = criterion_focal(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = 0.6*loss_focal + 0.4*loss_dice

        case "fast-scnn":
            outputs = outputs[0].float()

            loss_focal = criterion_focal(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = (0.6*loss_focal + 0.4*loss_dice)

        case "linknet":
            outputs = outputs.float()
            loss_focal = criterion_focal(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = (0.6*loss_focal + 0.4*loss_dice)

        case "pidnet":
            outputs = outputs.float()
            outputs = F.interpolate(outputs, size=(img_size, img_size), mode='bilinear', align_corners=True)

            loss_focal = criterion_focal(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = (0.6*loss_focal + 0.4*loss_dice)

        case _:
            raise ValueError(
                f"Erro: Nome de modelo {model_name} inválido. "
                f"Os nomes possíveis são: {' '.join(MODELS)}"
            )

    return loss


def calculate_batch_metrics(preds, masks):

    eps = 1e-7

    tp_pipe = torch.sum(preds * masks, dim=(1, 2, 3))
    fp_pipe = torch.sum(preds * (1 - masks), dim=(1, 2, 3))
    tn_pipe = torch.sum((1 - preds) * (1 - masks), dim=(1, 2, 3))
    fn_pipe = torch.sum((1 - preds) * masks, dim=(1, 2, 3))

    union_pipe = tp_pipe + fp_pipe + fn_pipe

    iou_pipe = torch.where(
        union_pipe == 0,
        torch.ones_like(union_pipe),
        tp_pipe / (union_pipe + eps)
    )

    dice_den_pipe = 2 * tp_pipe + fp_pipe + fn_pipe

    dice_pipe = torch.where(
        dice_den_pipe == 0,
        torch.ones_like(dice_den_pipe),
        2 * tp_pipe / (dice_den_pipe + eps)
    )

    precision_pipe = torch.where(
        union_pipe == 0,
        torch.ones_like(union_pipe),
        tp_pipe / (tp_pipe + fp_pipe + eps)
    )

    recall_pipe = torch.where(
        union_pipe == 0,
        torch.ones_like(union_pipe),
        tp_pipe / (tp_pipe + fn_pipe + eps)
    )


    tp_bg = tn_pipe
    fp_bg = fn_pipe
    fn_bg = fp_pipe
    tn_bg = tp_pipe

    union_bg = tp_bg + fp_bg + fn_bg

    iou_bg = torch.where(
        union_bg == 0,
        torch.ones_like(union_bg),
        tp_bg / (union_bg + eps)
    )

    dice_den_bg = 2 * tp_bg + fp_bg + fn_bg

    dice_bg = torch.where(
        dice_den_bg == 0,
        torch.ones_like(dice_den_bg),
        2 * tp_bg / (dice_den_bg + eps)
    )

    precision_bg = torch.where(
        union_bg == 0,
        torch.ones_like(union_bg),
        tp_bg / (tp_bg + fp_bg + eps)
    )

    recall_bg = torch.where(
        union_bg == 0,
        torch.ones_like(union_bg),
        tp_bg / (tp_bg + fn_bg + eps)
    )

    return {
        "iou_pipe_sum": iou_pipe.sum().item(),
        "dice_pipe_sum": dice_pipe.sum().item(),
        "precision_pipe_sum": precision_pipe.sum().item(),
        "recall_pipe_sum": recall_pipe.sum().item(),

        "iou_bg_sum": iou_bg.sum().item(),
        "dice_bg_sum": dice_bg.sum().item(),
        "precision_bg_sum": precision_bg.sum().item(),
        "recall_bg_sum": recall_bg.sum().item(),

        "samples": preds.shape[0],

        "tp": tp_pipe.sum().item(),
        "fp": fp_pipe.sum().item(),
        "tn": tn_pipe.sum().item(),
        "fn": fn_pipe.sum().item(),
    }


def calculate_and_register_epoch_metrics(
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
    writer,
    i,
    step
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
        total_tp
        / (total_tp + total_fp + total_fn + eps)
    )

    global_dice_pipe = (
        2 * total_tp
        / (2 * total_tp + total_fp + total_fn + eps)
    )

    global_precision_pipe = (
        total_tp
        / (total_tp + total_fp + eps)
    )

    global_recall_pipe = (
        total_tp
        / (total_tp + total_fn + eps)
    )

    global_iou_bg = (
        total_tn
        / (total_tn + total_fn + total_fp + eps)
    )

    global_dice_bg = (
        2 * total_tn
        / (2 * total_tn + total_fn + total_fp + eps)
    )

    global_precision_bg = (
        total_tn
        / (total_tn + total_fn + eps)
    )

    global_recall_bg = (
        total_tn
        / (total_tn + total_fp + eps)
    )

    global_mean_iou = (
        global_iou_pipe + global_iou_bg
    ) / 2

    global_mean_dice = (
        global_dice_pipe + global_dice_bg
    ) / 2

    global_mean_precision = (
        global_precision_pipe + global_precision_bg
    ) / 2

    global_mean_recall = (
        global_recall_pipe + global_recall_bg
    ) / 2

    global_accuracy = (
        total_tp + total_tn
    ) / (
        total_tp + total_fp + total_tn + total_fn + eps
    )

    if step == "train":
        split = "Train"
    elif step == "val":
        split = "Val"
    else:
        raise ValueError(
            "Erro: argumento step precisa ser 'train' ou 'val'."
        )

    writer.add_scalar(
        f"Loss/{split}",
        epoch_loss,
        i
    )

    writer.add_scalar(
        f"MeanPerImage/Pipe/IoU/{split}",
        epoch_miou_pipe,
        i
    )

    writer.add_scalar(
        f"MeanPerImage/Pipe/Dice/{split}",
        epoch_mdice_pipe,
        i
    )

    writer.add_scalar(
        f"MeanPerImage/Pipe/Precision/{split}",
        epoch_mprecision_pipe,
        i
    )

    writer.add_scalar(
        f"MeanPerImage/Pipe/Recall/{split}",
        epoch_mrecall_pipe,
        i
    )

    writer.add_scalar(
        f"MeanPerImage/Background/IoU/{split}",
        epoch_miou_bg,
        i
    )

    writer.add_scalar(
        f"MeanPerImage/Background/Dice/{split}",
        epoch_mdice_bg,
        i
    )

    writer.add_scalar(
        f"MeanPerImage/Background/Precision/{split}",
        epoch_mprecision_bg,
        i
    )

    writer.add_scalar(
        f"MeanPerImage/Background/Recall/{split}",
        epoch_mrecall_bg,
        i
    )

    writer.add_scalar(
        f"Global/Pipe/IoU/{split}",
        global_iou_pipe,
        i
    )

    writer.add_scalar(
        f"Global/Pipe/Dice/{split}",
        global_dice_pipe,
        i
    )

    writer.add_scalar(
        f"Global/Pipe/Precision/{split}",
        global_precision_pipe,
        i
    )

    writer.add_scalar(
        f"Global/Pipe/Recall/{split}",
        global_recall_pipe,
        i
    )

    writer.add_scalar(
        f"Global/Background/IoU/{split}",
        global_iou_bg,
        i
    )

    writer.add_scalar(
        f"Global/Background/Dice/{split}",
        global_dice_bg,
        i
    )

    writer.add_scalar(
        f"Global/Background/Precision/{split}",
        global_precision_bg,
        i
    )

    writer.add_scalar(
        f"Global/Background/Recall/{split}",
        global_recall_bg,
        i
    )

    writer.add_scalar(
        f"Global/ClassMean/IoU/{split}",
        global_mean_iou,
        i
    )

    writer.add_scalar(
        f"Global/ClassMean/Dice/{split}",
        global_mean_dice,
        i
    )

    writer.add_scalar(
        f"Global/ClassMean/Precision/{split}",
        global_mean_precision,
        i
    )

    writer.add_scalar(
        f"Global/ClassMean/Recall/{split}",
        global_mean_recall,
        i
    )

    writer.add_scalar(
        f"Global/Accuracy/{split}",
        global_accuracy,
        i
    )

    writer.add_scalar(
        f"ConfusionMatrix/TP/{split}",
        total_tp,
        i
    )

    writer.add_scalar(
        f"ConfusionMatrix/FP/{split}",
        total_fp,
        i
    )

    writer.add_scalar(
        f"ConfusionMatrix/TN/{split}",
        total_tn,
        i
    )

    writer.add_scalar(
        f"ConfusionMatrix/FN/{split}",
        total_fn,
        i
    )

    print(
        f"\n\n[{split}] "
        f"Loss: {epoch_loss:.4f} | "
        f"Pipe mIoU: {epoch_miou_pipe:.4f} | "
        f"Pipe Global IoU: {global_iou_pipe:.4f} | "
        f"BG mIoU: {epoch_miou_bg:.4f} | "
        f"BG Global IoU: {global_iou_bg:.4f} | "
        f"ClassMean Global IoU: {global_mean_iou:.4f}"
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
    }

def train_epoch(
    i,
    model,
    model_name,
    train_loader,
    device,
    img_size,
    optimizer,
    writer,
    criterion_focal,
    criterion_dice,
    criterion_boundary
):
    print(f"Época: {i}\n")
    print("Etapa de Treino:\n")

    model.train()

    if model_name == "bisenetv2":
        model.aux_mode = 'train'

    elif model_name == "pidnet":
        model.augment = True

    sum_train_loss = 0.0

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

    c = 0

    for images, masks in train_loader:

        print(
            f"Batch {c} de {len(train_loader)}",
            end='\r'
        )

        c += 1

        images = images.float().to(device)
        masks = masks.float().to(device)

        outputs = model(images)

        loss = train_losses_composition(
            model_name,
            outputs,
            masks,
            img_size,
            criterion_focal,
            criterion_dice,
            criterion_boundary
        )

        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        sum_train_loss += (
            loss.item() * images.shape[0]
        )

        if model_name == "pidnet":

            outputs = F.interpolate(
                outputs[1].float(),
                size=(img_size, img_size),
                mode='bilinear',
                align_corners=True
            ).float()

            preds = (
                torch.sigmoid(outputs) > 0.5
            ).float()

        elif model_name == "linknet":

            preds = (
                torch.sigmoid(outputs) > 0.5
            ).float()

        else:

            preds = (
                torch.sigmoid(outputs[0]) > 0.5
            ).float()

        batch_metrics = calculate_batch_metrics(
            preds,
            masks
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

    return calculate_and_register_epoch_metrics(
        sum_train_loss,

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

        writer,
        i,
        "train"
    )


def validate_epoch(
    i,
    model,
    model_name,
    val_loader,
    device,
    img_size,
    writer,
    criterion_focal,
    criterion_dice
):
    print("Etapa de validação:\n")

    model.eval()

    if model_name == "bisenetv2":
        model.aux_mode = 'eval'

    elif model_name == "pidnet":
        model.augment = False

    sum_val_loss = 0.0

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

    c = 0

    with torch.no_grad():

        for images, masks in val_loader:

            print(
                f"Batch {c} de {len(val_loader)}",
                end='\r'
            )

            c += 1

            images = images.float().to(device)
            masks = masks.float().to(device)

            outputs = model(images)

            loss = val_losses_composition(
                model_name,
                outputs,
                masks,
                img_size,
                criterion_focal,
                criterion_dice
            )

            sum_val_loss += (
                loss.item() * images.shape[0]
            )

            if model_name == "linknet":

                preds = (
                    torch.sigmoid(outputs) > 0.5
                ).float()

            elif model_name == "pidnet":

                outputs = F.interpolate(
                    outputs.float(),
                    size=(img_size, img_size),
                    mode='bilinear',
                    align_corners=True
                ).float()

                preds = (
                    torch.sigmoid(outputs) > 0.5
                ).float()

            else:

                preds = (
                    torch.sigmoid(outputs[0]) > 0.5
                ).float()

            batch_metrics = calculate_batch_metrics(
                preds,
                masks
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

    return calculate_and_register_epoch_metrics(
        sum_val_loss,

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

        writer,
        i,
        "val"
    )


def register_checkpoint(i, epochs, epoch_val_iou, epoch_val_dice, best_iou, best_dice, model, optimizer, model_name, checkpoint_path, fold):
    if epoch_val_iou > best_iou:
        print("Salvando checkpoint:\n")
        best_iou = epoch_val_iou

        torch.save({
            'epoch': i + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iou': best_iou,
            'dice': epoch_val_dice,
        }, f"{checkpoint_path}/{model_name}_fold{fold+1}_best_iou.pt")

        print(f"Novo recorde de IoU. Modelo salvo em: {os.path.join(checkpoint_path, model_name)}_fold{fold+1}_best_iou.pt")

    if epoch_val_dice > best_dice:
        print("Salvando checkpoint:\n")
        best_dice = epoch_val_dice

        torch.save({
            'epoch': i + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iou': epoch_val_iou,
            'dice': best_dice,
        }, f"{checkpoint_path}/{model_name}_fold{fold+1}_best_dice.pt")

        print(f"Novo recorde de Dice. Modelo salvo em: {os.path.join(checkpoint_path, model_name)}_fold{fold+1}_best_dice.pt")

    if i+1 == epochs:
        print("Salvando último checkpoint:\n")

        torch.save({
            'epoch': i + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iou': epoch_val_iou,
            'dice': epoch_val_dice,
        }, f"{checkpoint_path}/{model_name}_fold{fold+1}_last.pt")

        print(f"Último checkpoint. Modelo salvo em: {os.path.join(checkpoint_path, model_name)}_fold{fold+1}_last.pt")

    return best_iou, best_dice

def generate_kfolds(path_images, path_masks, k_folds):
    if Path(path_images).is_dir() and Path(path_masks).is_dir():
        pasta_imagens = Path(path_images)
        pasta_mascaras = Path(path_masks)
    else:
        raise ValueError("Erro, pasta de imagens ou máscaras não existe.")

    x_list = []
    y_list = []
    areas_list = []

    extensoes_validas = {'.jpg', '.jpeg', '.png'}

    print("Lendo diretórios e calculando áreas das máscaras...")

    for img_path in pasta_imagens.iterdir():
        if img_path.suffix.lower() not in extensoes_validas:
            continue

        nome_base = img_path.stem
        mask_path = pasta_mascaras / f"{nome_base}_label.png"

        if not mask_path.exists():
            print(f"Aviso: Máscara não encontrada para {img_path.name}. Ignorando.")
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Aviso: Erro ao ler a máscara {mask_path.name}. Ignorando.")
            continue

        area = np.sum(mask > 0)

        x_list.append(str(img_path))
        y_list.append(str(mask_path))
        areas_list.append(area)

    x = np.array(x_list)
    y_paths = np.array(y_list)
    areas = np.array(areas_list)

    cortes = np.percentile(areas, [25, 50, 75])

    labels = np.digitize(areas, cortes)

    print(f"\nTotal de imagens prontas: {len(x)}")
    print("Distribuição das classes geradas: ")
    for i in range(len(cortes) + 1):
        print(f"Classe {i}: {np.sum(labels == i)} imagens")

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    return skf, x, labels, y_paths


def train_session(
    img_size=320,
    epochs=10,
    batch_size=16,
    lr=1e-4,
    checkpoint_path="./models_checkpoints",
    model_name='model',
    path_images="",
    path_masks="",
    log_dir=""
):

    criterion_focal = smp.losses.FocalLoss(mode='binary', alpha=0.5, gamma=2.0)
    criterion_dice = smp.losses.DiceLoss(mode='binary')
    criterion_boundary = BoundaryLoss()

    k_folds = 5

    skf, x, labels, y_paths = generate_kfolds(path_images, path_masks, k_folds)

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, labels)):
        print(f"--- Iniciando Fold {fold + 1}/{k_folds} ---")

        writer = SummaryWriter(log_dir=f"runs/{log_dir}/{model_name}-{fold+1}")

        os.makedirs(checkpoint_path, exist_ok=True)

        match model_name:
            case "bisenetv2":
                model = BiSeNetV2(n_classes=1)
            case "fast-scnn":
                model = FastSCNN(num_classes=1, aux=True)
            case "linknet":
                model = smp.Linknet(
                    encoder_name="tu-mobilenetv4_conv_small",
                    in_channels=3,
                    classes=1,
                )
            case "pidnet":
                model = PIDNet(num_classes=1)
            case _:
                raise ValueError(
                    f"Erro: Nome de modelo {model_name} inválido. "
                    f"Os nomes possíveis são: {' '.join(MODELS)}"
                )
        x_train, y_train = x[train_idx], y_paths[train_idx]
        x_val, y_val = x[val_idx], y_paths[val_idx]

        train_dataset = SubPipeMiniDataset(
            x_train,
            y_train,
            transforms=transforms_train(img_size))

        val_dataset = SubPipeMiniDataset(
            x_val,
            y_val,
            transforms=transforms_val(img_size))

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Treinando no dispositivo {device}")

        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs-15, eta_min=1e-6
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=15
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[15]
        )

        best_iou = 0.0
        best_dice = 0.0

        for i in range(epochs):
            train_epoch(i, model, model_name, train_loader, device, img_size, optimizer, writer, criterion_focal, criterion_dice, criterion_boundary)

            epoch_val_metrics = validate_epoch(
                i,
                model,
                model_name,
                val_loader,
                device,
                img_size,
                writer,
                criterion_focal,
                criterion_dice
            )

            epoch_val_iou = epoch_val_metrics["iou_pipe"]
            epoch_val_dice = epoch_val_metrics["dice_pipe"]

            best_iou, best_dice = register_checkpoint(i, epochs, epoch_val_iou, epoch_val_dice, best_iou, best_dice, model, optimizer, model_name, checkpoint_path, fold)

            scheduler.step()

        writer.close()


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    train_session(**vars(args))
