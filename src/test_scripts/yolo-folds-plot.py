import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


def parse_args():

    parser = argparse.ArgumentParser(description="Script com flags e valores.")

    parser.add_argument(
        "-p", "--path", type=str, help="Caminho para os experimento"
    )
    parser.add_argument(
        "-m", "--model_name", type=str, help="Nome do modelo utilizado no experimento"
    )

    return parser

def analisar_resultados_kfold(caminhos_csv, nome):
    metricas_plot = [
        'train/box_loss', 'val/box_loss',
        'train/seg_loss', 'val/seg_loss',
        'metrics/mAP50(M)', 'metrics/mAP50-95(M)'
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()

    melhor_map50 = -1
    melhor_map50_95 = -1
    fold_melhor_map50 = -1
    fold_melhor_map50_95 = -1

    for i, caminho in enumerate(caminhos_csv):
        df = pd.read_csv(caminho)
        df.columns = df.columns.str.strip()

        fold_atual = i + 1

        for j, metrica in enumerate(metricas_plot):
            if metrica in df.columns:
                axes[j].plot(df['epoch'], df[metrica], label=f'Fold {fold_atual}')
                axes[j].set_title(metrica)
                axes[j].set_xlabel('Epoch')
                axes[j].legend()

        if 'metrics/mAP50(M)' in df.columns:
            max_map50 = df['metrics/mAP50(M)'].max()
            if max_map50 > melhor_map50:
                melhor_map50 = max_map50
                fold_melhor_map50 = fold_atual

        if 'metrics/mAP50-95(M)' in df.columns:
            max_map50_95 = df['metrics/mAP50-95(M)'].max()
            if max_map50_95 > melhor_map50_95:
                melhor_map50_95 = max_map50_95
                fold_melhor_map50_95 = fold_atual

    plt.tight_layout()
    plt.savefig(f'curvas_{nome}_kfold.png', dpi=300)
    plt.show()

    print("=== RESULTADOS DAS MÁSCARAS (M) ===")
    print(f"Melhor Fold para mAP50(M): Fold {fold_melhor_map50} com valor de {melhor_map50:.4f}")
    print(f"Melhor Fold para mAP50-95(M): Fold {fold_melhor_map50_95} com valor de {melhor_map50_95:.4f}")

    with open(f"melhores_folds_{nome}.txt", mode="w") as txt:
        txt.writelines(f"Melhor Fold para mAP50(M): Fold {fold_melhor_map50} com valor de {melhor_map50:.4f}\nMelhor Fold para mAP50-95(M): Fold {fold_melhor_map50_95} com valor de {melhor_map50_95:.4f}")


def main(path: str, model_name: str):
    arquivos_csv = [
        f'{path}/fold1/results.csv',
        f'{path}/fold2/results.csv',
        f'{path}/fold3/results.csv',
        f'{path}/fold4/results.csv',
        f'{path}/fold5/results.csv'
    ]
    analisar_resultados_kfold(arquivos_csv, model_name)

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    main(**vars(args))
