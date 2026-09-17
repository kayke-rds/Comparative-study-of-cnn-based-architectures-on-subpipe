import argparse
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Script de teste de contorno de label yolo."
        )
    )

    parser.add_argument(
        "-pl",
        "--path_label",
        type=str,
        required=True,
        help="Pasta com txt yolo.",
    )

    parser.add_argument(
        "-pm",
        "--path_mask",
        type=str,
        required=True,
        help="Pasta com as máscaras em imagem.",
    )

    return parser


def ler_contornos_txt(path_label, largura, altura, normalizado=True):
    """Lê o arquivo .txt linha por linha e retorna uma lista de contornos,

    permitindo múltiplos objetos por imagem.
    """
    contornos = []

    with open(path_label, "r") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue

            partes = linha.split()

            coords_str = partes[1:]

            if len(coords_str) < 4 or len(coords_str) % 2 != 0:
                continue

            coords = np.array(coords_str, dtype=np.float32).reshape(-1, 2)

            if normalizado:
                coords[:, 0] *= largura
                coords[:, 1] *= altura

            contorno = coords.astype(np.int32).reshape(-1, 1, 2)
            contornos.append(contorno)

    return contornos


def gerar_sobreposicao(path_mask, path_label, normalizado=True):
    mascara = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
    if mascara is None:
        raise FileNotFoundError(
            f"Não foi possível carregar a imagem: {path_mask}"
        )

    altura, largura = mascara.shape

    contornos = ler_contornos_txt(path_label, largura, altura, normalizado)

    if not contornos:
        raise ValueError("O arquivo .txt não contém contornos válidos.")

    sobreposicao = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
    overlay_transparente = sobreposicao.copy()


    cv2.drawContours(
        overlay_transparente, contornos, -1, (0, 255, 0), cv2.FILLED
    )

    cv2.addWeighted(
        overlay_transparente, 0.35, sobreposicao, 0.65, 0, sobreposicao
    )

    cv2.drawContours(sobreposicao, contornos, -1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite("sobreposicao_resultado.png", sobreposicao)
    print(
        f"Sucesso! {len(contornos)} objeto(s) processado(s) e salvo(s) em 'sobreposicao_resultado.png'."
    )

    return sobreposicao

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    gerar_sobreposicao(**vars(args))
