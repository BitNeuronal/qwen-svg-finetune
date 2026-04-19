"""
train.py — Fine-tune Qwen2.5-Coder-1.5B-Instruct para generar SVG, con TRL + LoRA.

Uso:
    # Smoke run local (100 ejemplos, 50 steps, ~5-10 min en M2 Max)
    python src/train.py --smoke

    # Entrenamiento completo (1 epoch sobre 4750 ejemplos)
    python src/train.py

    # Override de quant (ej. forzar LoRA sin cuantizar aunque haya CUDA)
    python src/train.py --quant none

Device:
    - CUDA:  QLoRA 4-bit con bitsandbytes (modo Colab / GPU NVIDIA)
    - MPS:   LoRA fp16 sin cuantizar (Apple Silicon)
    - CPU:   LoRA fp32 (solo para test; muy lento)

El script elige automáticamente, pero podés forzar con --quant.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATASET_ID = "thesantatitan/deepseek-svg-dataset"

# LoRA: los 7 módulos lineales típicos de Qwen (attention + MLP).
# Menos módulos = menos parámetros entrenables = menos calidad.
# "all-linear" también funciona; acá elegimos explícito para que sea reproducible.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def detect_device() -> dict:
    """Detecta GPU disponible y recomienda cuantización."""
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        bf16_ok = cap[0] >= 8  # Ampere o superior (T4 es Turing, sm_75 -> fp16)
        return {
            "name": "cuda",
            "dtype": torch.bfloat16 if bf16_ok else torch.float16,
            "bf16": bf16_ok,
            "fp16": not bf16_ok,
            "recommend_4bit": True,
        }
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # En MPS fp16 es inestable (no hay loss scaling dinámico). Usamos bf16 nativo
        # para evitar overflow → NaN. Requiere torch >= 2.4 en Apple Silicon.
        return {
            "name": "mps",
            "dtype": torch.bfloat16,
            "bf16": False,   # el modelo ya está cargado en bf16, no activamos autocast
            "fp16": False,
            "recommend_4bit": False,
        }
    return {
        "name": "cpu",
        "dtype": torch.float32,
        "bf16": False,
        "fp16": False,
        "recommend_4bit": False,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Carga modelo+dataset+config, imprime info y sale (sin entrenar). Paso 3.")
    p.add_argument("--smoke", action="store_true",
                   help="Run corto (100 ejemplos, 50 steps) para validar que todo corre. Paso 4.")
    p.add_argument("--output_dir", default="./outputs/qwen-svg-lora")
    # Training hyperparams
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=1,
                   help="per_device_train_batch_size. En T4 sube a 2; en M2 Max dejá en 1.")
    p.add_argument("--grad_accum", type=int, default=8,
                   help="Effective batch = batch_size * grad_accum.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_epochs", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=20,
                   help="Steps de warmup (en smoke mode se ajusta automático a 2).")
    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # Quant override
    p.add_argument("--quant", choices=["auto", "4bit", "none"], default="auto",
                   help="'auto' decide según device; '4bit' fuerza QLoRA; 'none' LoRA sin cuantizar.")
    return p.parse_args()


def build_model(model_id: str, use_4bit: bool, device: dict):
    """Carga el modelo con o sin cuantización."""
    kwargs = {"dtype": device["dtype"]}
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            sys.exit("bitsandbytes no instalado. En Mac: corré con --quant none.")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=device["dtype"],
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device["name"] == "mps" and not use_4bit:
        model = model.to("mps")
    return model


def main() -> None:
    args = parse_args()

    # -------------------- Device & modo --------------------
    device = detect_device()
    use_4bit = (args.quant == "4bit") or (args.quant == "auto" and device["recommend_4bit"])

    print("=== Device ===")
    print(f"  {device['name']}  (dtype={device['dtype']})")
    print(f"  Modo: {'QLoRA 4-bit' if use_4bit else 'LoRA sin cuantizar'}")
    if device["name"] == "cpu":
        print("  ⚠  CPU-only: será lento. Considerá Colab T4 para el entrenamiento real.")
    if use_4bit and device["name"] != "cuda":
        sys.exit("ERROR: 4-bit quantization requiere CUDA (bitsandbytes).")

    # -------------------- Tokenizer --------------------
    print(f"\n=== Tokenizer: {MODEL_ID} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  pad={tokenizer.pad_token}  eos={tokenizer.eos_token}")

    # -------------------- Modelo --------------------
    print(f"\n=== Cargando modelo (primera vez baja ~3 GB) ===")
    model = build_model(MODEL_ID, use_4bit, device)
    model.config.use_cache = False  # requerido para gradient checkpointing

    # -------------------- LoRA config --------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )
    print("\n=== LoRA ===")
    print(f"  r={args.lora_r}  alpha={args.lora_alpha}  dropout={args.lora_dropout}")
    print(f"  target_modules={LORA_TARGET_MODULES}")

    # -------------------- Dataset --------------------
    print(f"\n=== Dataset: {DATASET_ID} ===")
    ds = load_dataset(DATASET_ID)
    train_ds = ds["train"]
    eval_ds = ds["test"]
    if args.smoke:
        print("  SMOKE mode: subsample -> 100 train / 20 eval, max_steps=50")
        train_ds = train_ds.select(range(100))
        eval_ds = eval_ds.select(range(20))
    # Nos quedamos solo con la columna `messages` — TRL la consume directo con el chat_template del tokenizer.
    keep_cols = ["messages"]
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    eval_ds = eval_ds.remove_columns([c for c in eval_ds.column_names if c not in keep_cols])
    print(f"  train={len(train_ds)}  eval={len(eval_ds)}")

    # -------------------- SFTConfig --------------------
    warmup_steps = 2 if args.smoke else args.warmup_steps
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=50 if args.smoke else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        max_length=args.max_seq_length,
        packing=False,
        assistant_only_loss=True,  # loss solo sobre tokens de assistant
        logging_steps=5 if args.smoke else 20,
        save_strategy="no" if args.smoke else "epoch",
        eval_strategy="no",
        bf16=device["bf16"],
        fp16=device["fp16"] and not use_4bit,  # cuando 4bit, compute_dtype controla
        # gradient_checkpointing: útil en CUDA para ahorrar VRAM (T4 16GB).
        # En MPS causa NaN cascade con Qwen2.5 + LoRA (loss salta a ~85 en step ~10).
        # Lo desactivamos en Mac; en CUDA/Colab lo mantenemos activo.
        gradient_checkpointing=(device["name"] != "mps"),
        report_to="none",
        seed=42,
    )

    # -------------------- Trainer --------------------
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # Cuántos params entrenamos de verdad
    trainer.model.print_trainable_parameters()

    # -------------------- Dry-run: salir sin entrenar --------------------
    if args.dry_run:
        print("\n=== DRY-RUN OK ===")
        print("  Config cargada, modelo + dataset OK, trainer construido sin errores.")
        print("  Próximo paso: correr --smoke para validar que la loss baja.")
        return

    # -------------------- Train! --------------------
    print("\n=== Training ===")
    trainer.train()

    # -------------------- Save --------------------
    if args.smoke:
        print("\n=== SMOKE run completado ===")
        print("  Revisá que la loss haya bajado y que no haya habido errores.")
        print("  No se guardó el adaptador (smoke mode).")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"\n=== Adaptador guardado en {args.output_dir} ===")


if __name__ == "__main__":
    main()
