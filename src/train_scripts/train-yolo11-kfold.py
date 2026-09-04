import argparse
from ultralytics import YOLO

def parse_args():

    parser = argparse.ArgumentParser(description="Script com flags e valores.")

    parser.add_argument(
        "-s", "--img_size", type=int, default=640, help="Tamanho da imagem (padrão: 640x640)"
    )
    parser.add_argument(
        "-n", "--epochs", type=int, default=10, help="Número de épocas (padrão: 10)"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=16, help="Tamanho do batch (padrão: 16)"
    )

    parser.add_argument(
        "-m", "--model_name", type=str, help="Nome do modelo sem extensão"
    )

    return parser

def main(img_size=640,
    epochs=10,
    batch_size=16,
    model_name='yolo11_unified_dataset',):

    k_folds = 5

    for fold_num in range(1, k_folds + 1):
        print(f"--- Iniciando Fold {fold_num}/{k_folds} ---")

        model = YOLO("./src/yolo_base_models/yolo11n-seg.pt")

        results = model.train(
            data=f"../UnitedDataset/train/fold{fold_num}_data.yaml",
            epochs=epochs,
            imgsz=img_size,
            seed=42,
            batch=batch_size,
            device=0,
            workers=4,
            cache=False,
            project=f"runs/{model_name}",
            name=f"fold{fold_num}"
        )


if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    main(**vars(args))
