import cv2
import numpy as np


def ler_contornos_txt(caminho_txt, largura, altura, normalizado=True):
    """Lê o arquivo .txt linha por linha e retorna uma lista de contornos,

    permitindo múltiplos objetos por imagem.
    """
    contornos = []

    with open(caminho_txt, "r") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue

            partes = linha.split()

            # Pula o primeiro elemento (classe) e pega apenas as coordenadas
            coords_str = partes[1:]

            # Valida se há pares completos de coordenadas (x, y)
            if len(coords_str) < 4 or len(coords_str) % 2 != 0:
                continue

            # Converte para float32 e molda em pares (N, 2)
            coords = np.array(coords_str, dtype=np.float32).reshape(-1, 2)

            # Desnormaliza se necessário
            if normalizado:
                coords[:, 0] *= largura
                coords[:, 1] *= altura

            # Formata no padrão exigido pelo OpenCV: np.int32 com shape (N, 1, 2)
            contorno = coords.astype(np.int32).reshape(-1, 1, 2)
            contornos.append(contorno)

    return contornos


def gerar_sobreposicao(caminho_mascara, caminho_txt, normalizado=True):
    mascara = cv2.imread(caminho_mascara, cv2.IMREAD_GRAYSCALE)
    if mascara is None:
        raise FileNotFoundError(
            f"Não foi possível carregar a imagem: {caminho_mascara}"
        )

    altura, largura = mascara.shape

    # Agora obtemos uma LISTA de contornos (um para cada linha do .txt)
    contornos = ler_contornos_txt(caminho_txt, largura, altura, normalizado)

    if not contornos:
        raise ValueError("O arquivo .txt não contém contornos válidos.")

    sobreposicao = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
    overlay_transparente = sobreposicao.copy()

    # --- DESENHO DE TODOS OS CONTORNOS ---

    # 1. Preenchimento (Verde): desenha todos os objetos da lista
    cv2.drawContours(
        overlay_transparente, contornos, -1, (0, 255, 0), cv2.FILLED
    )

    # Transparência
    cv2.addWeighted(
        overlay_transparente, 0.35, sobreposicao, 0.65, 0, sobreposicao
    )

    # 2. Bordas (Vermelho): espessura = 2 para todos os contornos
    cv2.drawContours(sobreposicao, contornos, -1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite("sobreposicao_resultado.png", sobreposicao)
    print(
        f"Sucesso! {len(contornos)} objeto(s) processado(s) e salvo(s) em 'sobreposicao_resultado.png'."
    )

    return sobreposicao

# === EXECUÇÃO ===
if __name__ == "__main__":
    gerar_sobreposicao(
        caminho_mascara="/home/Cake/Documentos/UFRB/TCC/yolo_testes_subpipe/dataset/original_masks/1693574361.444_label.png", 
        caminho_txt="/home/Cake/Documentos/UFRB/TCC/yolo_testes_subpipe/dataset/yolo_masks/1693574361.444_label.txt", 
        normalizado=True)