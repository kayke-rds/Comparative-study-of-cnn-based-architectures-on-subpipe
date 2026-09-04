import cv2
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

def main():
    pasta_imagens = Path("../UnitedDataset/train/images")
    pasta_mascaras = Path("../UnitedDataset/train/masks")

    x_list = []
    y_list = []
    areas_list = []

    extensoes_validas = {'.jpg', '.jpeg', '.png'}

    print("Lendo diretórios e calculando áreas das máscaras...")

    for img_path in pasta_imagens.iterdir():
        print(f"Processando imagem {img_path.stem}", end="\r")
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
    areas = np.array(areas_list)

    cortes = np.percentile(areas, [25, 50, 75])

    labels = np.digitize(areas, cortes)

    print(f"\nTotal de imagens prontas: {len(x)}")
    print(f"Distribuição das classes geradas: ")
    for i in range(len(cortes) + 1):
        print(f"Classe {i}: {np.sum(labels == i)} imagens")

    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    out_dir = Path("../UnitedDataset/train/")
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, labels)):
        fold_num = fold + 1

        train_txt = out_dir / f"fold{fold_num}_train.txt"
        val_txt = out_dir / f"fold{fold_num}_val.txt"
        yaml_path = out_dir / f"fold{fold_num}_data.yaml"

        train_txt.write_text("\n".join(x[train_idx]) + "\n")
        val_txt.write_text("\n".join(x[val_idx]) + "\n")

        yaml_content = (
            f"train: {train_txt.resolve()}\n"
            f"val: {val_txt.resolve()}\n\n"
            f"nc: 1\n"
            f"names:\n  0: pipe\n"
        )
        yaml_path.write_text(yaml_content)

        print(f"Fold {fold_num}: {len(train_idx)} treino / {len(val_idx)} val -> {yaml_path}")

if __name__ == "__main__":
    main()
