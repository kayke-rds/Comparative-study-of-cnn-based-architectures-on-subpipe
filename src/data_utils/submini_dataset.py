import cv2
import torch
import albumentations as A
import numpy as np
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from semantic_seg_models.fast_scnn_tramac.models.fast_scnn import FastSCNN
from torch.utils.tensorboard import SummaryWriter
import os
import segmentation_models_pytorch as smp
from torch.optim import lr_scheduler


IMAGE_SIZE = 640

def conv_bn_to_gn(model, num_groups=32):
    """
    Varre o modelo recursivamente e substitui BatchNorm2d por GroupNorm.
    Ajusta os grupos se o número de canais não for divisível por 32.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            channels = child.num_features

            # Se os canais forem divisíveis pelo número de grupos ideal, usa ele.
            # Caso contrário, tenta reduzir os grupos ou usa GroupNorm por canal (InstanceNorm equivalente)
            if channels % num_groups == 0:
                groups = num_groups
            elif channels % 16 == 0:
                groups = 16
            elif channels % 8 == 0:
                groups = 8
            else:
                groups = 1 # Se for um número ímpar ou quebrado, age como LayerNorm/InstanceNorm

            # Substitui a camada
            setattr(model, name, nn.GroupNorm(num_groups=groups, num_channels=channels))
        else:
            # Aplica recursivamente nos blocos internos
            conv_bn_to_gn(child, num_groups)



def transforms_train(img_size=IMAGE_SIZE):
   return A.Compose([

    A.Resize(height=img_size, width=img_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-45, 45), p=0.3),

    A.OneOf(list([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    ]), p=0.7),

    A.OneOf(list([
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    ]), p=0.5),

    A.OneOf(list([
        A.MotionBlur(p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.GaussNoise(std_range=(0.015, 0.03), p=0.2),
    ]), p=0.4),

    A.Normalize(mean=(0.3551499061927887, 0.5187109166720711, 0.4940478866148916), std=(0.11498362917744831, 0.0765838378600218, 0.14597918539958313)),
    ToTensorV2(),
])

def transforms_val(img_size=IMAGE_SIZE):
   return A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.Normalize(mean=(0.3551499061927887, 0.5187109166720711, 0.4940478866148916), std=(0.11498362917744831, 0.0765838378600218, 0.14597918539958313)),
    ToTensorV2(),
])

class SubPipeMiniDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = str(self.image_paths[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = str(self.mask_paths[idx])

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = (mask_img > 0).astype(np.float32)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float()

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        return image, mask

if __name__ == "__main__":
    pasta_imagens = Path("./dataset/subpipe/images_enhanced")
    pasta_mascaras = Path("./dataset/subpipe/masks")

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

    # np.digitize pega a área de cada imagem e diz em qual "balde" (0, 1, 2, ou 3) ela caiu
    labels = np.digitize(areas, cortes)

    print(f"\nTotal de imagens prontas: {len(x)}")
    print(f"Distribuição das classes geradas: ")
    for i in range(len(cortes) + 1):
        print(f"Classe {i}: {np.sum(labels == i)} imagens")

    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, labels)):
        print(f"--- Iniciando Fold {fold + 1}/{k_folds} ---")
        img_size=640
        epochs=10
        batch_size=4
        lr=1e-4
        checkpoint_path="./"
        model_name='scnn_test'

        writer = SummaryWriter(log_dir=f"runs/fast-scnn-teste-fold{fold}")

        os.makedirs(checkpoint_path, exist_ok=True)

        model = FastSCNN(num_classes=1, aux=True)

        x_train, y_train = x[train_idx], y_paths[train_idx]
        x_val, y_val = x[val_idx], y_paths[val_idx]

        train_dataset = SubPipeMiniDataset(
            x_train,
            y_train,
            transforms=transforms_train(IMAGE_SIZE))

        val_dataset = SubPipeMiniDataset(
            x_val,
            y_val,
            transforms=transforms_val(IMAGE_SIZE))

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Treinando no dispositivo {device}")

        model = model.to(device)

        criterion_focal = smp.losses.FocalLoss(mode='binary', alpha=0.5, gamma=2.0)
        criterion_dice = smp.losses.DiceLoss(mode='binary')

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
        )

        best_iou = 0.0
        best_dice = 0.0

        batch_size_target = 32
        acumulation_steps = int(batch_size_target/batch_size)

        for i in range(epochs):
            print(f"Época: {i}\n")
            print("Etapa de Treino:\n")
            model.train()
            sum_train_loss = 0.0
            total_iou = 0.0
            total_dice = 0.0
            total_precision = 0.0
            total_recall = 0.0
            c = 0

            for images, masks in train_loader:
                print(f"Batch {c} de {len(train_loader)}", end='\r')
                c += 1
                images = images.float().to(device)
                masks = masks.float().to(device)

                outputs = model(images)

                loss_focal = 0.7*criterion_focal(outputs[0].float(), masks) + 0.3*criterion_focal(outputs[1].float(), masks)
                loss_dice = 0.7*criterion_dice(outputs[0].float(), masks) + 0.3*criterion_dice(outputs[1].float(), masks)
                loss = (0.6*loss_focal + 0.4*loss_dice)
                loss_scaled = loss/acumulation_steps

                loss_scaled.backward()

                if (c) % acumulation_steps == 0 or c == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                sum_train_loss += loss.item()

                preds = (torch.sigmoid(outputs[0]) > 0.5).float()

                true_positive = torch.sum(preds * masks)
                false_positive = torch.sum(preds * (1 - masks))
                false_negative = torch.sum((1 - preds) * masks)

                precision = true_positive / (false_positive + true_positive + 1e-7)
                recall = true_positive / (false_negative + true_positive + 1e-7)

                intersection = (preds * masks).sum(dim=(2, 3))
                union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) - intersection
                iou = (intersection + 1e-7) / (union + 1e-7)
                dice = (2 * intersection + 1e-7) / (preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) + 1e-7)

                total_iou += iou.mean().item()
                total_dice += dice.mean().item()
                total_precision += precision.mean().item()
                total_recall += recall.mean().item()

            epoch_train_loss = sum_train_loss / len(train_loader)
            epoch_train_iou = total_iou / len(train_loader)
            epoch_train_dice = total_dice / len(train_loader)
            epoch_train_precision = total_precision / len(train_loader)
            epoch_train_recall = total_recall / len(train_loader)

            writer.add_scalar("Total loss/Train", epoch_train_loss, i)
            writer.add_scalar("IoU/Train", epoch_train_iou, i)
            writer.add_scalar("Dice/Train", epoch_train_dice, i)
            writer.add_scalar("Precision/Train", epoch_train_precision, i)
            writer.add_scalar("Recall/Train", epoch_train_recall, i)

            print(f"\n\nPerda da Época {i}: Loss Train: {epoch_train_loss:.4f} | IoU Train: {epoch_train_iou:.4f} | Dice Train: {epoch_train_dice:.4f}\n")

            print("Etapa de validação:\n")
            model.eval()
            running_val_loss = 0.0
            total_iou = 0.0
            total_dice = 0.0
            total_precision = 0.0
            total_recall = 0.0

            c = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    print(f"Batch {c} de {len(val_loader)}", end='\r')
                    c += 1
                    images = images.float().to(device)
                    masks = masks.float().to(device)

                    outputs = model(images)[0].float()

                    loss_focal = criterion_focal(outputs, masks)
                    loss_dice = criterion_dice(outputs, masks)
                    loss = (0.6*loss_focal + 0.4*loss_dice)
                    running_val_loss += loss.item()

                    preds = (torch.sigmoid(outputs) > 0.5).float()

                    true_positive = torch.sum(preds * masks)
                    false_positive = torch.sum(preds * (1 - masks))
                    false_negative = torch.sum((1 - preds) * masks)

                    precision = true_positive / (false_positive + true_positive + 1e-7)
                    recall = true_positive / (false_negative + true_positive + 1e-7)

                    intersection = (preds * masks).sum(dim=(2, 3))
                    union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) - intersection
                    iou = (intersection + 1e-7) / (union + 1e-7)
                    dice = (2 * intersection + 1e-7) / (preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) + 1e-7)

                    total_iou += iou.mean().item()
                    total_dice += dice.mean().item()
                    total_precision += precision.mean().item()
                    total_recall += recall.mean().item()

            epoch_val_loss = running_val_loss / len(val_loader)
            epoch_val_iou = total_iou / len(val_loader)
            epoch_val_dice = total_dice / len(val_loader)
            epoch_val_precision = total_precision / len(val_loader)
            epoch_val_recall = total_recall / len(val_loader)

            scheduler.step(epoch_val_loss)

            writer.add_scalar("Total loss/Val", epoch_val_loss, i)
            writer.add_scalar("IoU/Val", epoch_val_iou, i)
            writer.add_scalar("Dice/Val", epoch_val_dice, i)
            writer.add_scalar("Precision/Val", epoch_val_precision, i)
            writer.add_scalar("Recall/Val", epoch_val_recall, i)

            print(f"\n\n[Resultados] Loss Treino: {epoch_train_loss:.4f} | Loss Val: {epoch_val_loss:.4f} | IoU Val: {epoch_val_iou:.4f} | Dice Val: {epoch_val_dice:.4f}")

            if epoch_val_iou > best_iou:
                print("Salvar melhor checkpoint:\n")
                best_iou = epoch_val_iou

                torch.save({
                    'epoch': i + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iou': best_iou,
                    'dice': epoch_val_dice,
                }, f"{checkpoint_path}/{model_name}_best_iou.pt")

                print(f"Novo recorde de IoU. Modelo salvo em: {os.path.join(checkpoint_path, model_name)}_best_iou.pt")

            if epoch_val_dice > best_dice:
                print("Salvar melhor checkpoint:\n")
                best_dice = epoch_val_dice

                torch.save({
                    'epoch': i + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iou': epoch_val_iou,
                    'dice': best_dice,
                }, f"{checkpoint_path}/{model_name}_best_dice.pt")

                print(f"Novo recorde de Dice. Modelo salvo em: {os.path.join(checkpoint_path, model_name)}_best_dice.pt")

        writer.close()
