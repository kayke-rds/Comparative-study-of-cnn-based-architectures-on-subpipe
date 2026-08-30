import os
import cv2
import numpy as np
import shutil
from pathlib import Path

def compensate_red_channel(image: np.ndarray) -> np.ndarray:
    """
    Compensa a atenuação do canal vermelho utilizando o canal verde como referência.
    """
    b, g, r = cv2.split(image.astype(np.float32))
    mean_r = np.mean(r)
    mean_g = np.mean(g)

    if mean_r < mean_g:
        r_comp = r + (mean_g - mean_r) * (1.0 - r / 255.0) * (g / 255.0)
    else:
        r_comp = r

    r_comp = np.clip(r_comp, 0, 255).astype(np.uint8)
    return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r_comp])

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size = (8, 8)) -> np.ndarray:
    """
    Aplica CLAHE no canal de luminância (L) do espaço de cores LAB.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_clahe = clahe.apply(l)

    merged_lab = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

def enhance_pipeline_combined(image: np.ndarray) -> np.ndarray:
    """
    Pipeline em cascata:
    1º Compensação cromática do canal vermelho
    2º Realce de contraste adaptativo (CLAHE)
    """
    color_corrected = compensate_red_channel(image)
    fully_enhanced = apply_clahe(color_corrected, clip_limit=3.0, tile_size=(8, 8))
    return fully_enhanced

def process_dataset_combined(input_dir: str, output_dir: str):
    """
    Percorre a pasta de imagens, processa as imagens com a
    abordagem combinada e copia para o output dir.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    if not input_path.exists():
        raise Exception("Erro: O caminho informado não existe.")

    img_files = list(input_path.glob('*.*'))
    print(f"[Combinado] Processando {len(img_files)} imagens")

    for img_file in img_files:
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        img_processed = enhance_pipeline_combined(img)
        cv2.imwrite(str(output_path / img_file.name), img_processed)


if __name__ == "__main__":
    # Configure os caminhos do seu dataset
    DATASET_ORIGINAL = "../MarinaPipeFiltered/images"
    DATASET_SAIDA = "../MarinaPipeFilteredEnhanced/images"

    process_dataset_combined(DATASET_ORIGINAL, DATASET_SAIDA)
    print("Dataset combinando Vermelho + CLAHE gerado com sucesso!")
