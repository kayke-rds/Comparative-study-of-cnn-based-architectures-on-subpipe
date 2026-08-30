import os
import random
import shutil
from pathlib import Path
import yaml

# === CONFIGURAÇÕES ===
DATASET_ORIGEM = Path("./dataset")  # Pasta atual do seu dataset
DATASET_DESTINO = Path("./yolo_dataset")  # Pasta do novo dataset estruturado
VAL_RATIO = 0.15  # 15% para validação
SEED = 42  # Semente fixa para reprodutibilidade

# Extensões de imagem suportadas
EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def extrair_id(caminho_arquivo: Path) -> str:
    """Extrai o ID base do arquivo, ignorando a extensão e sufixos conhecidos.

    Exemplo: '1693573934.247.jpg' -> '1693573934.247' '1693573934.247_label.txt'
    -> '1693573934.247'
    """
    nome = caminho_arquivo.name

    # Remove o sufixo do label se existir
    if nome.endswith("_label.txt"):
        return nome.replace("_label.txt", "")

    # Caso seja imagem, remove a extensão final
    return caminho_arquivo.stem


def preparar_dataset():
    pasta_images = DATASET_ORIGEM / "original_images"
    pasta_labels = DATASET_ORIGEM / "yolo_masks"

    if not pasta_images.exists() or not pasta_labels.exists():
        raise FileNotFoundError(
            f"Verifique se as pastas '{pasta_images}' e '{pasta_labels}' existem."
        )

    # 1. Mapear todas as imagens disponíveis por ID
    mapa_imagens = {}
    for img_path in pasta_images.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in EXTENSOES_IMAGEM:
            img_id = extrair_id(img_path)
            mapa_imagens[img_id] = img_path

    # 2. Mapear todos os labels disponíveis por ID
    mapa_labels = {}
    for lbl_path in pasta_labels.iterdir():
        if lbl_path.is_file() and lbl_path.name.endswith("_label.txt"):
            lbl_id = extrair_id(lbl_path)
            mapa_labels[lbl_id] = lbl_path

    # 3. Cruzar imagens e labels válidos
    ids_comuns = sorted(list(set(mapa_imagens.keys()) & set(mapa_labels.keys())))
    print(f"Encontrados {len(ids_comuns)} pares válidos de Imagem + Label.")

    if len(ids_comuns) == 0:
        raise ValueError(
            "Nenhum par correspondente entre imagens e labels foi encontrado."
        )

    # 4. Separar em Treino e Validação (85% / 15%)
    random.seed(SEED)
    random.shuffle(ids_comuns)

    num_val = int(len(ids_comuns) * VAL_RATIO)
    val_ids = set(ids_comuns[:num_val])
    train_ids = set(ids_comuns[num_val:])

    print(
        f"Divisão: {len(train_ids)} imagens para Treino | {len(val_ids)} imagens para Validação."
    )

    # 5. Criar estrutura de pastas do YOLO
    subpastas = ["images/train", "images/val", "labels/train", "labels/val"]
    for sub in subpastas:
        (DATASET_DESTINO / sub).mkdir(parents=True, exist_ok=True)

    # 6. Copiar arquivos para os destinos corretos
    def copiar_arquivos(ids, split):
        for item_id in ids:
            img_origem = mapa_imagens[item_id]
            lbl_origem = mapa_labels[item_id]

            # Copia imagem mantendo sua extensão original
            img_destino = DATASET_DESTINO / "images" / split / img_origem.name
            shutil.copy2(img_origem, img_destino)

            # Copia label renomeando para <ID>.txt (padrão YOLO sem o '_label')
            lbl_destino = (
                DATASET_DESTINO / "labels" / split / f"{item_id}.txt"
            )
            shutil.copy2(lbl_origem, lbl_destino)

    copiar_arquivos(train_ids, "train")
    copiar_arquivos(val_ids, "val")

    # 7. Gerar o arquivo data.yaml automaticamente
    data_yaml = {
        "path": str(DATASET_DESTINO.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "objeto"},  # Ajuste o nome da classe conforme necessário
    }

    yaml_path = DATASET_DESTINO / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print("\nDataset organizado com sucesso!")
    print(f"Estrutura gerada em: {DATASET_DESTINO.resolve()}")
    print(f"Arquivo de configuração criado em: {yaml_path}")


if __name__ == "__main__":
    preparar_dataset()