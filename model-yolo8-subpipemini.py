from ultralytics import YOLO

def main():
    # 1. Carrega o modelo pré-treinado de segmentação
    model = YOLO("yolov8n-seg.pt") 

    # 2. Inicia o treinamento
    results = model.train(
        data="./yolo_dataset_enhanced/data.yaml",
        epochs=200,
        imgsz=640,
        seed=42,
        batch=16,
        device=0,           # Alterar para 'cpu' se não tiver GPU disponível
        workers=4,
        cache=False,
        name="model-yolo8-subpipemini"
    )

if __name__ == '__main__':
    main()