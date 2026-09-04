import pandas as pd
from ultralytics import YOLO

def avaliar_modelo(model_path, data_path, project_path, model_name):
    model = YOLO(model_path)

    metrics = model.val(
        data=data_path,
        split="test",
        project=project_path,
        name="test_metrics"
    )

    return {
        "Modelo": model_name,
        "Box_Precision": metrics.box.mp,
        "Box_Recall": metrics.box.mr,
        "Box_mAP50": metrics.box.map50,
        "Box_mAP50-95": metrics.box.map,
        "Mask_Precision": metrics.seg.mp,
        "Mask_Recall": metrics.seg.mr,
        "Mask_mAP50": metrics.seg.map50,
        "Mask_mAP50-95": metrics.seg.map
    }

def main():
    resultados = []
    yaml_path = "../UnitedDataset/test-data.yaml"

    resultados.append(avaliar_modelo(
        "./runs/segment/runs/yolo8_kfold_united_dataset/fold5/weights/best.pt",
        yaml_path,
        "runs/yolo8_kfold_united_dataset",
        "YOLO8"
    ))

    resultados.append(avaliar_modelo(
        "./runs/segment/runs/yolo11_kfold_united_dataset/fold1/weights/best.pt",
        yaml_path,
        "runs/yolo11_kfold_united_dataset",
        "YOLO11"
    ))

    resultados.append(avaliar_modelo(
        "./runs/segment/runs/yolo26_kfold_united_dataset/fold5/weights/best.pt",
        yaml_path,
        "runs/yolo26_kfold_united_dataset",
        "YOLO26"
    ))

    df_metrics = pd.DataFrame(resultados)
    df_metrics.to_csv("metricas_test_set.csv", index=False)

if __name__ == '__main__':
    main()
