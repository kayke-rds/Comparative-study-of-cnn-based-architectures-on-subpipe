import glob
import os
import time
import cv2
import numpy as np

SEGMENTATION_DIR = "./dataset/subpipe/masks"
MASK_YOLO_DIR = './dataset/subpipe/yolo_masks'
CLASS_PIPE_LABEL = 0

os.makedirs(MASK_YOLO_DIR, exist_ok=True)

masks_paths = sorted(glob.glob(os.path.join(SEGMENTATION_DIR, "*.png")))
print(f"Total de máscaras encontradas: {len(masks_paths)}")

start_time = time.time()

for idx, mask_path in enumerate(masks_paths, start=1):
    base_name = os.path.basename(mask_path)
    print(f"[{idx}/{len(masks_paths)}] Convertendo: {base_name}...", end="\r")

    # MUDANÇA 1: Carregar como imagem COLORIDA (BGR)
    mask_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask_bgr is None:
        continue

    height, width, _ = mask_bgr.shape

    # MUDANÇA 2: Converter para o espaço de cores HSV (muito melhor para isolar o Vermelho)
    hsv = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2HSV)

    # No HSV, o vermelho fica nas extremidades do espectro.
    # Definimos os limites inferiores e superiores para capturar tons de vermelho.
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Cria as máscaras binárias isolando a cor vermelha
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    thresh = cv2.bitwise_or(mask1, mask2) # Combina as duas faixas de vermelho

    # Se não houver nenhum pixel vermelho na imagem, gera o .txt de background vazio para o YOLO
    if cv2.countNonZero(thresh) == 0:
        txt_name = base_name[:-4] + ".txt"
        caminho_txt = os.path.join(MASK_YOLO_DIR, txt_name)
        open(caminho_txt, 'w').close()
        continue

    # Encontra os contornos na máscara que contém APENAS o que era vermelho
    contours, _ = cv2.findContours(thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    txt_lines = []
    for contour in contours:
        # Filtro de ruído leve (linhas pequenas)
        if cv2.contourArea(contour) < 15:
            continue

        coords = contour.reshape(-1, 2).astype(np.float32)
        coords[:, 0] /= width
        coords[:, 1] /= height

        string_pontos = " ".join(f"{val:.6f}" for val in coords.ravel())
        txt_lines.append(f"{CLASS_PIPE_LABEL} {string_pontos}")

    txt_name = base_name[:-4] + ".txt"
    caminho_txt = os.path.join(MASK_YOLO_DIR, txt_name)

    if txt_lines:
        with open(caminho_txt, "w") as f:
            f.write("\n".join(txt_lines))
    else:
        open(caminho_txt, 'w').close()

print(f"\nProcesso concluído com sucesso em {time.time() - start_time:.2f} segundos!")
