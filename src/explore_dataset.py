"""
explore_dataset.py — Inspección del dataset thesantatitan/deepseek-svg-dataset.

Uso:
    python src/explore_dataset.py
    python src/explore_dataset.py --n 5            # mostrar 5 ejemplos
    python src/explore_dataset.py --split train    # cambiar de split

Imprime:
    - Splits disponibles y tamaño de cada uno
    - Features (columnas) y su tipo
    - N ejemplos (truncados para que se lea en terminal)
    - Estadísticas básicas (longitud promedio de prompt y de SVG)
"""

from __future__ import annotations

import argparse
from statistics import mean, median

from datasets import load_dataset

DATASET_ID = "thesantatitan/deepseek-svg-dataset"


def _truncate(s: str, limit: int = 200) -> str:
    s = str(s)
    if len(s) <= limit:
        return s
    return s[:limit] + f"  …(+{len(s) - limit} chars)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="Cuántos ejemplos mostrar")
    parser.add_argument("--split", type=str, default=None, help="Split específico (train/test/...)")
    parser.add_argument("--sample_stats", type=int, default=200, help="Cuántas filas para estadísticas")
    args = parser.parse_args()

    print(f"Cargando dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID)

    print("\n=== Splits disponibles ===")
    for name, split in ds.items():
        print(f"  {name:<10} -> {len(split):>8} filas")

    split_name = args.split or next(iter(ds.keys()))
    split = ds[split_name]
    print(f"\nInspeccionando split: '{split_name}' ({len(split)} filas)")

    print("\n=== Features (columnas) ===")
    for col, feat in split.features.items():
        print(f"  {col:<20} :: {feat}")

    print(f"\n=== {args.n} ejemplo(s) (truncados a 200 chars por campo) ===")
    for i in range(min(args.n, len(split))):
        row = split[i]
        print(f"\n--- Ejemplo {i} ---")
        for col in split.column_names:
            value = row[col]
            print(f"  [{col}] {_truncate(value, 200)}")

    # Estadísticas sobre los primeros N ejemplos
    print(f"\n=== Estadísticas sobre los primeros {args.sample_stats} ejemplos ===")
    sample = split.select(range(min(args.sample_stats, len(split))))
    for col in split.column_names:
        try:
            lengths = [len(str(row[col])) for row in sample]
        except Exception:
            continue
        print(
            f"  [{col}] len chars -> "
            f"min={min(lengths)}  p50={int(median(lengths))}  "
            f"mean={int(mean(lengths))}  max={max(lengths)}"
        )


if __name__ == "__main__":
    main()
