import argparse
from json import encoder
import os
import torch
import albumentations as A
import segmentation_models_pytorch as smp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataset_utils.submini_dataset import SubPipeMiniDataset, divideDataset, transforms_train, transforms_val
from torch.utils.tensorboard import SummaryWriter

def parse_args():

    parser = argparse.ArgumentParser(description="Script com flags e valores.")

    parser.add_argument(
        "-e", "--encoder", type=str, default='mn4', help="Nome do encoder (padrão: mn4)"
    )
    parser.add_argument(
        "-s", "--img_size", type=int, default=640, help="Tamanho da imagem (padrão: 640x640)"
    )
    parser.add_argument(
        "-n", "--epochs", type=int, default=10, help="Número de épocas (padrão: 10)"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=4, help="Tamanho do batch (padrão: 4)"
    )
    parser.add_argument(
        "-l", "--lr", type=float, default=1e-4, help="Taxa de aprendizado (padrão: 1e-4)"
    )
    parser.add_argument(
        "-c", "--checkpoint_path", type=str, default="./models_checkpoints", help="Caminho para salvar checkpoints (padrão: ./models_checkpoints)"
    )

    parser.add_argument(
        "-m", "--model_name", type=str, help="Nome do modelo sem extensão"
    )

    return parser

def main(
    encoder='tu-mobilenetv4_conv_small',
    img_size=640,
    epochs=10,
    batch_size=4,
    lr=1e-4,
    checkpoint_path="./",
    model_name=f'linknet+{encoder}',
):

    writer = SummaryWriter(log_dir=f"runs/linknet+{encoder}-subpipe")

    os.makedirs(checkpoint_path, exist_ok=True)

    match encoder:
        case 'mn4':
            model = smp.Linknet(
                encoder_name="tu-mobilenetv4_conv_small",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            )
        case 'mo0':
            model = smp.Linknet(
                encoder_name="mobileone_s0",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            )
        case _:
            raise ValueError(f"Opção de encoder inválida (flag -e), escolha mn4 para mobilenetv4 ou mo0 para mobileone-s0.")

    train_imgs, val_imgs, train_labels, val_labels = divideDataset()

    train_dataset = SubPipeMiniDataset(train_imgs,
        train_labels,
        "./dataset/Cam0_images_enhanced",
        "./dataset/original_masks",
        transforms=transforms_train(img_size=(img_size, img_size)))

    val_dataset = SubPipeMiniDataset(val_imgs,
        val_labels,
        "./dataset/Cam0_images_enhanced",
        "./dataset/original_masks",
        transforms=transforms_val(img_size=(img_size, img_size)))

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

    batch_size_target = 32
    acumulation_steps = int(batch_size_target/batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=(acumulation_steps)**0.5*lr, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
    )

    best_iou = 0.0
    best_dice = 0.0

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

            outputs = model(images).float()

            loss_focal = criterion_focal(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = (0.6*loss_focal + 0.4*loss_dice)
            loss_scaled = loss/acumulation_steps

            loss_scaled.backward()

            if (c) % acumulation_steps == 0 or c == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            sum_train_loss += loss.item()

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

        print(f"\n\n[Resultados] Loss Train: {epoch_train_loss:.4f} | IoU Train: {epoch_train_iou:.4f} | Dice Train: {epoch_train_dice:.4f} | Precison Train: {epoch_train_precision:.4f} | Recall Train: {epoch_train_recall:.4f}")


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

                outputs = model(images).float()

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

        print(f"\n\n[Resultados] Loss Val: {epoch_val_loss:.4f} | IoU Val: {epoch_val_iou:.4f} | Dice Val: {epoch_val_dice:.4f} | Precison Val: {epoch_val_precision:.4f} | Recall Val: {epoch_val_recall:.4f}")

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

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    main(**vars(args))
