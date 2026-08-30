import cv2
import shutil
import random
from pathlib import Path

# Configurações de diretórios
dataset_marina_path = Path("../MarinaPipe")
output_dataset_path = Path("../MarinaPipeFiltered")

output_images = output_dataset_path / "images"
output_masks = output_dataset_path / "masks"

# Criar pastas de saída
output_images.mkdir(parents=True, exist_ok=True)
output_masks.mkdir(parents=True, exist_ok=True)

# Parâmetros de amostragem
# Define quantas imagens vazias manter para cada imagem com tubulação (Ex: 1.0 = 1:1)
proporcao_vazias = 0.5

# Percorrer os 7 vídeos
for video_dir in dataset_marina_path.glob("video_*"):
    video_name = video_dir.name
    img_dir = video_dir / "resized_selected_images"
    mask_dir = video_dir / "coarse_annotation"

    if not img_dir.exists() or not mask_dir.exists():
        continue

    imagens_com_tubo = []
    imagens_sem_tubo = []

    # Iterar sobre as máscaras em coarse_annotation
    for mask_path in mask_dir.glob("*.*"):
        # Encontrar a imagem correspondente usando o nome do arquivo (sem extensão)
        img_path = next(img_dir.glob(f"{mask_path.stem[:-6]}.*"), None)

        if not img_path:
            continue

        # Ler a máscara em escala de cinza para verificar se há anotação
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Se houver qualquer pixel > 0, há tubulação na máscara
        if cv2.countNonZero(mask) > 0:
            imagens_com_tubo.append((img_path, mask_path))
        else:
            imagens_sem_tubo.append((img_path, mask_path))

    # Determinar a quantidade de imagens sem tubo a reter
    qtd_amostras_vazias = int(len(imagens_com_tubo) * proporcao_vazias)
    amostras_sem_tubo = random.sample(
        imagens_sem_tubo,
        min(qtd_amostras_vazias, len(imagens_sem_tubo))
    )

    arquivos_selecionados = imagens_com_tubo + amostras_sem_tubo

    # Copiar os arquivos para o novo diretório com o prefixo do vídeo
    for img_p, mask_p in arquivos_selecionados:
        new_img_name = f"{video_name}_{img_p.name}"
        new_mask_name = f"{video_name}_{mask_p.name}"

        shutil.copy(img_p, output_images / new_img_name)
        shutil.copy(mask_p, output_masks / new_mask_name)

print(f"Extração concluída em: {output_dataset_path}")
