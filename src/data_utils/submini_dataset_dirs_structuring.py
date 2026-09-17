import argparse
import glob
import os
import shutil
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Script para criação de um dataset de imagens e masks a partir da pasta Segmentation do SubPipeMini"
        )
    )

    parser.add_argument(
        "-s",
        "--segmentation_dir",
        type=str,
        required=True,
        help="Caminho para a pasta Segmentation original do SubPipeMini.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Pasta de destino do dataset.",
    )

    return parser


def main(segmentation_dir, output_dir):
    if not os.path.exists(segmentation_dir):
        raise ValueError("Erro: Caminho para pasta Segmentation não existe.")

    os.makedirs(output_dir, exist_ok=True)

    images_dir = f"{output_dir}/images"
    masks_dir = f"{output_dir}/masks"

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    content_path = sorted(glob.glob(os.path.join(segmentation_dir, "*")))
    print(f"Total de arquivos encontradas: {len(content_path)}")

    if len(content_path) != 1294:
        raise ValueError("Erro, a quantidade de arquivos encontrada não foi igual ao total do dataset subpipemini.")

    start_time = time.time()

    for idx, item_path in enumerate(content_path, start=1):
        base_name = os.path.basename(item_path)
        print(f"[{idx}/{len(content_path)}] Organizando: {base_name}...", end="\r")

        if base_name[-10:] == "_label.png":
            _ = shutil.copy(item_path, masks_dir)
        else:
            _ = shutil.copy(item_path, images_dir)


    print(f"\nProcesso concluído com sucesso em {time.time() - start_time:.2f} segundos!")


if __name__  == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    main(**vars(args))
