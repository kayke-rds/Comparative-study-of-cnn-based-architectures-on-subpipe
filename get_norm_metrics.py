import cv2
import numpy as np
from pathlib import Path

def calcular_estatisticas_dataset(pasta_imagens, caminho_saida_txt):
    extensoes_aceitas = {'.jpg', '.jpeg', '.png'}

    caminhos = [
        p for p in Path(pasta_imagens).iterdir()
        if p.is_file() and p.suffix.lower() in extensoes_aceitas
    ]

    if not caminhos:
        print("Nenhuma imagem com extensão .jpg, .jpeg ou .png encontrada na pasta especificada.")
        return

    print(f"Calculando métricas para {len(caminhos)} imagens...")

    pixel_num = 0
    soma_canais = np.zeros(3)
    soma_canais_quadrado = np.zeros(3)

    for path in caminhos:
        img = cv2.imread(str(path))
        if img is None:
            print(f"Erro ao ler a imagem: {path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalizar para a escala [0, 1] (esperado pelas redes neurais)
        img = img.astype(np.float32) / 255.0

        pixel_num += (img.shape[0] * img.shape[1])
        soma_canais += np.sum(img, axis=(0, 1))
        soma_canais_quadrado += np.sum(np.square(img), axis=(0, 1))

    media = soma_canais / pixel_num

    # Variância = E[X^2] - (E[X])^2
    variancia = (soma_canais_quadrado / pixel_num) - np.square(media)
    desvio_padrao = np.sqrt(variancia)

    texto_resultado = (
        f"Estatísticas do Dataset Subpipe ({len(caminhos)} imagens)\n"
        f"--------------------------------------------------\n"
        f"Média (RGB): {media.tolist()}\n"
        f"Desvio Padrão (RGB): {desvio_padrao.tolist()}\n\n"
        f"Copie e cole no seu script do Albumentations:\n"
        f"A.Normalize(mean={media.tolist()}, std={desvio_padrao.tolist()})"
    )

    with open(caminho_saida_txt, 'w', encoding='utf-8') as f:
        f.write(texto_resultado)

    print("Cálculo concluído!")
    print(texto_resultado)

# ==========================================

pasta_dataset = "./dataset/subpipe/images_enhanced"
arquivo_saida = "enhanced_images_norm_metrics.txt"

calcular_estatisticas_dataset(pasta_dataset, arquivo_saida)
