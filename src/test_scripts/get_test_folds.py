import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch


ARCHITECTURE_NAMES = {
    "fast-scnn": "fast-scnn",
    "bisenetv2": "bisenet-v2",
    "bisenet-v2": "bisenet-v2",
    "linknet": "linknet",
    "pidnet": "pidnet",
}

FILENAME_PATTERN = re.compile(
    r"^(?P<architecture>.+)_fold(?P<fold>\d+)_(?P<checkpoint>best_iou|best_dice)\.pt$"
)


def normalize_architecture(name: str) -> str | None:
    key = name.lower()
    return key if key in ARCHITECTURE_NAMES else None


def load_metric(checkpoint_path: Path, metric: str) -> float:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if metric not in checkpoint:
        raise KeyError(
            f"O checkpoint '{checkpoint_path.name}' nao possui a chave '{metric}'."
        )

    value = checkpoint[metric]

    if torch.is_tensor(value):
        value = value.item()

    return float(value)


def collect_checkpoints(models_dir: Path):
    data = {}

    for path in models_dir.glob("*_fold*_best_*.pt"):
        match = FILENAME_PATTERN.match(path.name)
        if not match:
            continue

        architecture_raw = match.group("architecture")
        architecture = normalize_architecture(architecture_raw)

        if architecture is None:
            continue

        fold = int(match.group("fold"))
        checkpoint_type = match.group("checkpoint")

        data.setdefault(architecture, {}).setdefault(checkpoint_type, {})[fold] = path

    return data


def calculate_statistics(values_by_fold: dict[int, float]):
    folds = sorted(values_by_fold)
    values = np.array([values_by_fold[fold] for fold in folds], dtype=float)

    mean = float(np.mean(values))

    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    closest_fold = min(folds, key=lambda fold: abs(values_by_fold[fold] - mean))
    closest_value = values_by_fold[closest_fold]

    return mean, std, closest_fold, closest_value


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calcula media, desvio padrao e fold mais proximo da media "
            "para os checkpoints best_iou e best_dice de cada arquitetura."
        )
    )
    parser.add_argument(
        "models_dir",
        nargs="?",
        default="models_checkpoints",
        help="Pasta contendo os checkpoints. Padrao: models_checkpoints",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "CSV de saida. Padrao: "
            "<models_dir>/checkpoint_statistics.csv"
        ),
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Quantidade esperada de folds por arquitetura. Padrao: 5",
    )

    args = parser.parse_args()

    models_dir = Path(args.models_dir)

    if not models_dir.is_dir():
        raise FileNotFoundError(
            f"A pasta '{models_dir}' nao foi encontrada."
        )

    output_path = (
        Path(args.output)
        if args.output
        else models_dir / "checkpoint_statistics.csv"
    )

    checkpoint_data = collect_checkpoints(models_dir)

    expected_architectures = list(ARCHITECTURE_NAMES.values())
    # Remove duplicatas preservando a ordem.
    expected_architectures = list(dict.fromkeys(expected_architectures))

    rows = []

    for architecture_key, display_name in ARCHITECTURE_NAMES.items():
        if architecture_key not in checkpoint_data:
            continue

        architecture_checkpoints = checkpoint_data[architecture_key]

        iou_paths = architecture_checkpoints.get("best_iou", {})
        dice_paths = architecture_checkpoints.get("best_dice", {})

        missing_iou = [
            fold for fold in range(1, args.folds + 1)
            if fold not in iou_paths
        ]
        missing_dice = [
            fold for fold in range(1, args.folds + 1)
            if fold not in dice_paths
        ]

        if missing_iou:
            raise FileNotFoundError(
                f"{display_name}: faltam checkpoints best_iou dos folds "
                f"{missing_iou}."
            )

        if missing_dice:
            raise FileNotFoundError(
                f"{display_name}: faltam checkpoints best_dice dos folds "
                f"{missing_dice}."
            )

        best_iou_values = {
            fold: load_metric(path, "iou")
            for fold, path in sorted(iou_paths.items())
            if fold <= args.folds
        }

        best_dice_values = {
            fold: load_metric(path, "dice")
            for fold, path in sorted(dice_paths.items())
            if fold <= args.folds
        }

        if len(best_iou_values) != args.folds:
            raise ValueError(
                f"{display_name}: foram encontrados {len(best_iou_values)} "
                f"best_iou dentro dos folds esperados, mas eram esperados "
                f"{args.folds}."
            )

        if len(best_dice_values) != args.folds:
            raise ValueError(
                f"{display_name}: foram encontrados {len(best_dice_values)} "
                f"best_dice dentro dos folds esperados, mas eram esperados "
                f"{args.folds}."
            )

        (
            iou_mean,
            iou_std,
            iou_closest_fold,
            iou_closest_value,
        ) = calculate_statistics(best_iou_values)

        (
            dice_mean,
            dice_std,
            dice_closest_fold,
            dice_closest_value,
        ) = calculate_statistics(best_dice_values)

        rows.append(
            {
                "architecture": display_name,
                "best_iou_mean": iou_mean,
                "best_iou_std": iou_std,
                "best_iou_closest_fold": iou_closest_fold,
                "best_iou_closest_value": iou_closest_value,
                "best_dice_mean": dice_mean,
                "best_dice_std": dice_std,
                "best_dice_closest_fold": dice_closest_fold,
                "best_dice_closest_value": dice_closest_value,
            }
        )

    if not rows:
        raise RuntimeError(
            "Nenhum checkpoint compativel foi encontrado na pasta."
        )

    order = {
        "fast-scnn": 0,
        "bisenet-v2": 1,
        "linknet": 2,
        "pidnet": 3,
    }
    rows.sort(key=lambda row: order.get(row["architecture"], 999))

    fieldnames = [
        "architecture",
        "best_iou_mean",
        "best_iou_std",
        "best_iou_closest_fold",
        "best_iou_closest_value",
        "best_dice_mean",
        "best_dice_std",
        "best_dice_closest_fold",
        "best_dice_closest_value",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            formatted_row = row.copy()

            for column in [
                "best_iou_mean",
                "best_iou_std",
                "best_iou_closest_value",
                "best_dice_mean",
                "best_dice_std",
                "best_dice_closest_value",
            ]:
                formatted_row[column] = f"{formatted_row[column]:.6f}"

            writer.writerow(formatted_row)

    print(f"CSV salvo em: {output_path}")

    print("\nResumo:")
    for row in rows:
        print(
            f"{row['architecture']}: "
            f"best IoU = {row['best_iou_mean']:.6f} +/- "
            f"{row['best_iou_std']:.6f} "
            f"(fold {row['best_iou_closest_fold']} mais proximo), "
            f"best Dice = {row['best_dice_mean']:.6f} +/- "
            f"{row['best_dice_std']:.6f} "
            f"(fold {row['best_dice_closest_fold']} mais proximo)"
        )


if __name__ == "__main__":
    main()
