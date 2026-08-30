import glob
import os
import shutil
import time

ORIGINAL_SEGMENTATION_DIR = "/home/Cake/Documentos/UFRB/TCC/subpipe-seg-project/dataset/Segmentation"
DATASET_DIR = '/home/Cake/Documentos/UFRB/TCC/subpipe-seg-project/dataset/sub-pipe'

os.makedirs(DATASET_DIR, exist_ok=True)

images_dir = f"{DATASET_DIR}/images"
masks_dir = f"{DATASET_DIR}/masks"

os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

content_path = sorted(glob.glob(os.path.join(ORIGINAL_SEGMENTATION_DIR, "*")))
print(f"Total de arquivos encontradas: {len(content_path)}")

if len(content_path) != 1294:
    raise Exception("Erro, a quantidade de arquivos encontrada não foi igual ao total do dataset subpipemini.")

start_time = time.time()

for idx, item_path in enumerate(content_path, start=1):
    base_name = os.path.basename(item_path)
    print(f"[{idx}/{len(content_path)}] Organizando: {base_name}...", end="\r")

    if base_name[-10:] == "_label.png":
        _ = shutil.copy(item_path, masks_dir)
    else:
        _ = shutil.copy(item_path, images_dir)


print(f"\nProcesso concluído com sucesso em {time.time() - start_time:.2f} segundos!")
