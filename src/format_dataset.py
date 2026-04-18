"""
format_dataset.py — Aplica el chat template de Qwen al dataset y mide longitud en tokens.

Uso:
    python src/format_dataset.py
    python src/format_dataset.py --n_show 2 --sample_stats 1000

Imprime:
    - Metadatos del tokenizer (vocab, pad/eos)
    - Un ejemplo ya formateado (tal cual lo verá el modelo, con tokens de control)
    - Estadísticas de longitud en tokens + sugerencia de max_seq_length
"""

from __future__ import annotations

import argparse
from statistics import mean, median

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET_ID = "thesantatitan/deepseek-svg-dataset"
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def _show_trimmed(text: str, head: int = 800, tail: int = 400) -> None:
    if len(text) <= head + tail:
        print(text)
        return
    print(text[:head])
    print(f"\n  ...[{len(text) - (head + tail)} chars omitidos]...\n")
    print(text[-tail:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_show", type=int, default=1, help="Cuántos ejemplos formateados mostrar")
    parser.add_argument("--sample_stats", type=int, default=500, help="Cuántas filas para medir tokens")
    args = parser.parse_args()

    print(f"Cargando tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  vocab_size = {tokenizer.vocab_size}")
    print(f"  pad_token  = {tokenizer.pad_token}")
    print(f"  eos_token  = {tokenizer.eos_token}")
    print(f"  chat_template cargado = {'sí' if tokenizer.chat_template else 'no'}")

    print(f"\nCargando dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID)
    train = ds["train"]

    print(f"\n=== {args.n_show} ejemplo(s) con apply_chat_template ===")
    for i in range(args.n_show):
        messages = train[i]["messages"]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        print(f"\n--- Ejemplo {i} (string que ve el modelo) ---")
        _show_trimmed(rendered)

    print(f"\n=== Longitud en tokens (primeras {args.sample_stats} filas) ===")
    n = min(args.sample_stats, len(train))
    sample = train.select(range(n))
    lengths = []
    for row in sample:
        # Nota: en transformers 5.x, apply_chat_template(tokenize=True) devuelve
        # un BatchEncoding (dict), no una lista. Hacemos el render a string primero
        # y luego tokenizamos explícitamente con .encode() — más portable y claro.
        rendered = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False,
        )
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        lengths.append(len(ids))
    lengths.sort()
    p50 = lengths[int(n * 0.50)]
    p95 = lengths[int(n * 0.95)]
    p99 = lengths[int(n * 0.99) if n >= 100 else -1]
    print(f"  min={min(lengths)}  p50={p50}  mean={int(mean(lengths))}  p95={p95}  p99={p99}  max={max(lengths)}")

    suggested = ((p95 // 128) + 1) * 128
    print(f"\n  Sugerencia max_seq_length = {suggested}")
    print(f"  → cubre p95 de ejemplos sin truncar, múltiplo de 128 (amigable con la GPU).")


if __name__ == "__main__":
    main()
