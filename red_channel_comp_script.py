import os
import cv2
import numpy as np
import shutil
from pathlib import Path

def compensate_red_channel(image: np.ndarray) -> np.ndarray:
    """
    Compensa a atenuação do canal vermelho em imagens subaquáticas
    utilizando a informação de intensidade do canal verde.
    """
    # Separa os canais no espaço BGR (padrão OpenCV)
    b, g, r = cv2.split(image.astype(np.float32))
    
    mean_r = np.mean(r)
    mean_g = np.mean(g)
    
    # Aplica a compensação apenas se o vermelho estiver atenuado em relação ao verde
    if mean_r < mean_g:
        # Fator de ganho proporcional à diferença das médias e à intensidade do verde
        r_comp = r + (mean_g - mean_r) * (1.0 - r / 255.0) * (g / 255.0)
    else:
        r_comp = r
        
    r_comp = np.clip(r_comp, 0, 255).astype(np.uint8)
    
    return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r_comp])

def process_dataset_red_only(input_dir: str, output_dir: str):
    """
    Percorre a estrutura do dataset YOLO, processa as imagens com
    compensação do canal vermelho e copia as labels intactas.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        img_dir = input_path / 'images' / split
        lbl_dir = input_path / 'labels' / split
        
        if not img_dir.exists():
            continue
            
        out_img_dir = output_path / 'images' / split
        out_lbl_dir = output_path / 'labels' / split
        
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # Processa e salva as imagens
        img_files = list(img_dir.glob('*.*'))
        print(f"[Canal Vermelho] Processando {len(img_files)} imagens em: {split}")
        
        for img_file in img_files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
                
            img_processed = compensate_red_channel(img)
            cv2.imwrite(str(out_img_dir / img_file.name), img_processed)
            
        # Copia as labels originais
        if lbl_dir.exists():
            for lbl_file in lbl_dir.glob('*.txt'):
                shutil.copy(str(lbl_file), str(out_lbl_dir / lbl_file.name))

if __name__ == "__main__":
    # Configure os caminhos do seu dataset
    DATASET_ORIGINAL = "/home/Cake/Documentos/UFRB/TCC/yolo_testes_subpipe/dataset_yolo_final"
    DATASET_SAIDA = "/home/Cake/Documentos/UFRB/TCC/yolo_testes_subpipe/dataset_yolo_red_comp"
    
    process_dataset_red_only(DATASET_ORIGINAL, DATASET_SAIDA)
    print("Dataset com correção do canal vermelho gerado com sucesso!")