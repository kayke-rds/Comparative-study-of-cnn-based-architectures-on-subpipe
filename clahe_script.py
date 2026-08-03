import os
import shutil
import cv2
from pathlib import Path

def melhorar_imagem_subaquatica(img_path, output_path):
    """
    Aplica CLAHE no canal L (Luminância) do espaço de cores LAB
    com tratamento de erros de leitura e canais.
    """
    try:
        img = cv2.imread(str(img_path))
        
        # Verifica se a imagem foi lida corretamente e possui 3 canais (BGR)
        if img is None or img.ndim != 3 or img.shape[2] != 3:
            print(f"[AVISO] Imagem inválida ou sem 3 canais ignorada: {img_path.name}")
            return False

        # Converte para espaço LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Aplica CLAHE no canal de Luminância
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Recombina e retorna para BGR
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        cv2.imwrite(str(output_path), enhanced)
        return True
    
    except Exception as e:
        print(f"[ERRO] Falha ao processar {img_path.name}: {e}")
        return False

def gerar_dataset_enhanced(origem_dir="./dataset_yolo_final", destino_dir="./dataset_yolo_enhanced"):
    origem = Path(origem_dir)
    destino = Path(destino_dir)
    
    extensoes_validas = {".jpg", ".jpeg", ".png", ".bmp"}
    
    print(f"--- Iniciando processamento: {origem} -> {destino} ---")
    
    for split in ["train", "val"]:
        img_src = origem / "images" / split
        img_dst = destino / "images" / split
        label_src = origem / "labels" / split
        label_dst = destino / "labels" / split
        
        # Cria diretórios de destino
        img_dst.mkdir(parents=True, exist_ok=True)
        label_dst.mkdir(parents=True, exist_ok=True)
        
        # 1. Copiar arquivos de label (.txt) intactos para manter as anotações
        if label_src.exists():
            labels = list(label_src.glob("*.txt"))
            for lbl in labels:
                shutil.copy(lbl, label_dst / lbl.name)
            print(f"[{split.upper()}] Copiados {len(labels)} arquivos de label (.txt).")
        else:
            print(f"[{split.upper()}] AVISO: Pasta de labels não encontrada em {label_src}!")

        # 2. Processar imagens com CLAHE
        imagens = [f for f in img_src.glob("*.*") if f.suffix.lower() in extensoes_validas]
        print(f"[{split.upper()}] Processando {len(imagens)} imagens com CLAHE...")
        
        sucessos = 0
        for idx, img_file in enumerate(imagens, start=1):
            out_file = img_dst / img_file.name
            ok = melhorar_imagem_subaquatica(img_file, out_file)
            if ok:
                sucessos += 1
            
            # Log de progresso a cada 50 imagens
            if idx % 50 == 0 or idx == len(imagens):
                print(f"  -> Processado {idx}/{len(imagens)} imagens...")
                
        print(f"[{split.upper()}] Concluído: {sucessos}/{len(imagens)} imagens salvas com sucesso.\n")

    print("=== Dataset aprimorado pronto para uso! ===")
    print("Lembre-se de apontar o caminho 'path' no seu data.yaml para './dataset_yolo_clahe'")

if __name__ == "__main__":
    gerar_dataset_enhanced(
        origem_dir="./dataset_yolo_final",
        destino_dir="./dataset_yolo_clahe"
    )