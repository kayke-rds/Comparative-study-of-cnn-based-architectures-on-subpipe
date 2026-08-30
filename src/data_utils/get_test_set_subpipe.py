import cv2
import random
import shutil
from pathlib import Path

# 1. Configurações de caminhos de entrada e saída
base_dir = Path("./dataset/subpipe")
masks_dir = base_dir / "masks"
images_dir = base_dir / "images_enhanced"
yolo_dir = base_dir / "yolo_masks"

output_dir = Path("../SubPipeDivided")

# Criar estrutura train/test
for split in ["train", "test"]:
    for folder in ["images", "masks", "labels"]:
        (output_dir / split / folder).mkdir(parents=True, exist_ok=True)

# 2. Varredura do diretório único
tubo = []
vazio = []

for mask_path in masks_dir.glob("*.*"):
    # Buscar arquivos correspondentes (ignorando a extensão da imagem)
    img_path = next(images_dir.glob(f"{mask_path.stem[:-6]}.*"), None)
    yolo_path = yolo_dir / f"{mask_path.stem}.txt"

    if not img_path or not yolo_path.exists():
        continue

    # Lê a máscara. O vermelho (tubo) vira ~76 em cinza e o preto (fundo) continua 0
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue

    # Separar entre imagens com tubulação e background puro
    if cv2.countNonZero(mask) > 0:
        tubo.append((img_path, mask_path, yolo_path))
    else:
        vazio.append((img_path, mask_path, yolo_path))

# 3. Cálculos de proporção para o conjunto total
total_amostras = len(tubo) + len(vazio)
qtd_teste_total = int(total_amostras * 0.15)

# 5% do teste deve ser de máscaras vazias
qtd_teste_vazio = int(qtd_teste_total * 0.05)

# Forçar ao menos 1 amostra vazia no teste se o arredondamento zerar
if qtd_teste_vazio == 0 and len(vazio) > 0 and qtd_teste_total > 0:
    qtd_teste_vazio = 1

qtd_teste_tubo = qtd_teste_total - qtd_teste_vazio

# Travas de segurança caso a cota exceda o número de amostras reais disponíveis
qtd_teste_vazio = min(qtd_teste_vazio, len(vazio))
qtd_teste_tubo = min(qtd_teste_tubo, len(tubo))

# 4. Amostragem aleatória
teste_vazio = random.sample(vazio, qtd_teste_vazio)
teste_tubo = random.sample(tubo, qtd_teste_tubo)

teste_set = teste_vazio + teste_tubo
# O treino recebe absolutamente tudo que sobrou
treino_set = [item for item in (tubo + vazio) if item not in teste_set]

# 5. Efetuar cópia física para a estrutura final
for conj, nome_split in [(teste_set, "test"), (treino_set, "train")]:
    for img_p, mask_p, yolo_p in conj:
        shutil.copy(img_p, output_dir / nome_split / "images" / img_p.name)
        shutil.copy(mask_p, output_dir / nome_split / "masks" / mask_p.name)
        shutil.copy(yolo_p, output_dir / nome_split / "labels" / yolo_p.name)

print(f"Divisão concluída! Treino: {len(treino_set)} imagens | Teste: {len(teste_set)} imagens.")
