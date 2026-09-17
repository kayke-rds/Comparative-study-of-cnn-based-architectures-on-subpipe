import cv2
import torch
import albumentations as A
import numpy as np
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


# Não está atualmente em uso, mas é útil para treinar em resoluções maiores
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



def transforms_train(img_size):
   return A.Compose([

    A.Resize(height=img_size, width=img_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-45, 45), p=0.3),

    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    ], p=0.7),

    A.OneOf([
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    ], p=0.5),

    A.OneOf([
        A.MotionBlur(p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.GaussNoise(std_range=(0.015, 0.03), p=0.2),
    ], p=0.4),

    A.Normalize(mean=(0.3551499061927887, 0.5187109166720711, 0.4940478866148916), std=(0.11498362917744831, 0.0765838378600218, 0.14597918539958313)),
    ToTensorV2(),
])

def transforms_val(img_size):
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
        if image is None:
            raise FileNotFoundError("Falha na leitura da máscara")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = str(self.mask_paths[idx])

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask_img is None:
            raise FileNotFoundError("Falha na leitura da máscara")

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
