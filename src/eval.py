"""
eval.py — Evalúa el adaptador LoRA en 5 prompts held-out del test set.

Compara adapter vs. modelo base con 3 métricas por muestra:
  1. Well-formedness XML     — % de outputs que parsean como XML válido.
  2. Renderizabilidad        — % de outputs que cairosvg renderiza a PNG.
  3. Uso de tags SVG válidos — % de tags dentro del whitelist del spec.

Guarda por cada sample (outputs/samples/NN/):
  - prompt.txt          — el prompt del usuario
  - adapter.svg + .png  — lo que generó el adapter
  - base.svg + .png     — lo que genera el modelo base (a menos que --skip-baseline)
  - ground.svg + .png   — el SVG de referencia del dataset
  - sidebyside.png      — comparación visual horizontal
  - adapter_raw.txt / base_raw.txt — output crudo del modelo (para debug)

Y un reporte agregado en outputs/eval_report.md con tabla comparativa.

Uso:
    # Después del training, con el adapter guardado en ./qwen-svg-lora
    python src/eval.py

    # Adapter en otro path, más samples
    python src/eval.py --adapter_path ./outputs/qwen-svg-lora --n_samples 10

    # Solo evalúa el adapter (más rápido, sin comparación)
    python src/eval.py --skip-baseline

Requiere:
    - GPU CUDA con bitsandbytes (carga el base en 4-bit) O
    - MPS/CPU con fp16/fp32 (sin cuantización; más RAM/más lento).
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATASET_ID = "thesantatitan/deepseek-svg-dataset"

# Subset de tags SVG 1.1/2.0 que aparecen razonablemente en outputs generados.
# No pretende ser exhaustivo — el objetivo es detectar alucinaciones obvias
# tipo <shape>, <draw>, <foo> cuando el modelo inventa tags.
VALID_SVG_TAGS = {
    "svg", "g", "defs", "symbol", "use", "switch",
    "path", "circle", "ellipse", "line", "polyline", "polygon", "rect",
    "text", "tspan", "textPath",
    "image", "foreignObject", "a",
    "linearGradient", "radialGradient", "stop", "pattern", "clipPath", "mask",
    "filter", "feGaussianBlur", "feColorMatrix", "feBlend", "feComposite",
    "feOffset", "feMerge", "feMergeNode", "feFlood", "feImage", "feTurbulence",
    "feDisplacementMap", "feMorphology", "feConvolveMatrix", "feDiffuseLighting",
    "feSpecularLighting", "feDistantLight", "fePointLight", "feSpotLight",
    "feTile", "feComponentTransfer", "feFuncR", "feFuncG", "feFuncB", "feFuncA",
    "feDropShadow",
    "marker", "view", "title", "desc", "metadata", "style",
    "animate", "animateTransform", "animateMotion", "set", "mpath",
}

SVG_REGEX = re.compile(r"<svg\b[^>]*>.*?</svg>", re.DOTALL | re.IGNORECASE)
TAG_REGEX = re.compile(r"<([A-Za-z][A-Za-z0-9\-_]*)")


# -----------------------------------------------------------------
# Device / modelo
# -----------------------------------------------------------------

def detect_device() -> dict:
    if torch.cuda.is_available():
        return {"name": "cuda", "dtype": torch.float16, "recommend_4bit": True}
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return {"name": "mps", "dtype": torch.bfloat16, "recommend_4bit": False}
    return {"name": "cpu", "dtype": torch.float32, "recommend_4bit": False}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_path", default="./qwen-svg-lora",
                   help="Path al adapter LoRA entrenado.")
    p.add_argument("--base_model", default=MODEL_ID)
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--output_dir", default="./outputs/samples")
    p.add_argument("--report_path", default="./outputs/eval_report.md")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-baseline", dest="skip_baseline", action="store_true",
                   help="Solo evalúa el adapter, no genera con base.")
    p.add_argument("--quant", choices=["auto", "4bit", "none"], default="auto")
    return p.parse_args()


def build_base_model(base_model: str, use_4bit: bool, device: dict):
    kwargs = {"dtype": device["dtype"]}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=device["dtype"],
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if device["name"] == "mps" and not use_4bit:
        model = model.to("mps")
    return model


# -----------------------------------------------------------------
# Inferencia
# -----------------------------------------------------------------

def generate(model, tokenizer, messages, max_new_tokens, temperature, top_p) -> str:
    """Aplica chat template + genera. Devuelve solo los tokens nuevos decodificados."""
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


# -----------------------------------------------------------------
# Métricas sobre SVG generado
# -----------------------------------------------------------------

def extract_svg(text: str) -> str | None:
    m = SVG_REGEX.search(text)
    return m.group(0) if m else None


def is_well_formed(svg: str) -> bool:
    try:
        ET.fromstring(svg)
        return True
    except ET.ParseError:
        return False


def can_render(svg: str) -> bool:
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                         output_width=256, output_height=256)
        return True
    except Exception:
        return False


def tag_validity(svg: str) -> tuple[int, int]:
    tags = TAG_REGEX.findall(svg)
    tags = [t.split(":")[-1] for t in tags]  # strip ns:tag -> tag
    valid = sum(1 for t in tags if t in VALID_SVG_TAGS)
    return valid, len(tags)


def render_png(svg: str, path: Path, size: int = 512) -> bool:
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                         write_to=str(path),
                         output_width=size, output_height=size)
        return True
    except Exception as e:
        print(f"    render falló: {type(e).__name__}: {str(e)[:80]}")
        return False


def sidebyside(paths: list[Path | None], labels: list[str], out: Path) -> None:
    """Compone los PNGs horizontalmente con un header de texto por panel."""
    from PIL import Image, ImageDraw, ImageFont
    W, H = 512, 512
    HEADER = 40
    panels = []
    for p, label in zip(paths, labels):
        canvas = Image.new("RGB", (W, H + HEADER), "white")
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.load_default()
            draw.text((20, 10), label, fill="black", font=font)
        except Exception:
            draw.text((20, 10), label, fill="black")
        if p and p.exists():
            img = Image.open(p).convert("RGB").resize((W, H))
            canvas.paste(img, (0, HEADER))
        else:
            draw.rectangle([20, HEADER + 20, W - 20, H + HEADER - 20],
                           outline="red", width=3)
            draw.text((W // 2 - 60, HEADER + H // 2),
                      "(render failed)", fill="red")
        panels.append(canvas)
    PAD = 16
    total_w = W * len(panels) + PAD * (len(panels) - 1)
    combined = Image.new("RGB", (total_w, H + HEADER), "white")
    for i, panel in enumerate(panels):
        combined.paste(panel, (i * (W + PAD), 0))
    combined.save(out)


# -----------------------------------------------------------------
# Pipeline por sample
# -----------------------------------------------------------------

def evaluate_sample(svg: str | None) -> dict:
    """Aplica las 3 métricas sobre un SVG extraído (puede ser None)."""
    metrics = {"has_svg": bool(svg), "well_formed": False, "renders": False,
               "valid_tags": 0, "total_tags": 0}
    if svg:
        metrics["well_formed"] = is_well_formed(svg)
        metrics["renders"] = can_render(svg)
        v, t = tag_validity(svg)
        metrics["valid_tags"] = v
        metrics["total_tags"] = t
    return metrics


def run_sample(model, tokenizer, messages, sample_dir: Path, args, skip_baseline: bool):
    """Genera adapter + (base) + ground truth para un sample y guarda todo a disco."""
    sample_dir.mkdir(parents=True, exist_ok=True)

    input_messages = messages[:-1]
    user_prompt = next(m["content"] for m in messages if m["role"] == "user")
    ground_truth_text = messages[-1]["content"]
    (sample_dir / "prompt.txt").write_text(user_prompt, encoding="utf-8")

    # Ground truth
    gt_svg = extract_svg(ground_truth_text)
    gt_png_path = None
    if gt_svg:
        (sample_dir / "ground.svg").write_text(gt_svg, encoding="utf-8")
        p = sample_dir / "ground.png"
        if render_png(gt_svg, p):
            gt_png_path = p

    # ADAPTER
    print("  Generando con adapter...")
    adapter_text = generate(model, tokenizer, input_messages,
                            args.max_new_tokens, args.temperature, args.top_p)
    (sample_dir / "adapter_raw.txt").write_text(adapter_text, encoding="utf-8")
    adapter_svg = extract_svg(adapter_text)
    adapter_metrics = evaluate_sample(adapter_svg)
    adapter_png_path = None
    if adapter_svg:
        (sample_dir / "adapter.svg").write_text(adapter_svg, encoding="utf-8")
        p = sample_dir / "adapter.png"
        if render_png(adapter_svg, p):
            adapter_png_path = p

    # BASE
    base_metrics = None
    base_png_path = None
    if not skip_baseline:
        print("  Generando con base (adapter disabled)...")
        with model.disable_adapter():
            base_text = generate(model, tokenizer, input_messages,
                                 args.max_new_tokens, args.temperature, args.top_p)
        (sample_dir / "base_raw.txt").write_text(base_text, encoding="utf-8")
        base_svg = extract_svg(base_text)
        base_metrics = evaluate_sample(base_svg)
        if base_svg:
            (sample_dir / "base.svg").write_text(base_svg, encoding="utf-8")
            p = sample_dir / "base.png"
            if render_png(base_svg, p):
                base_png_path = p

    # Side-by-side
    if skip_baseline:
        sidebyside([adapter_png_path, gt_png_path],
                   ["Adapter (fine-tuned)", "Ground truth"],
                   sample_dir / "sidebyside.png")
    else:
        sidebyside([base_png_path, adapter_png_path, gt_png_path],
                   ["Base model", "Adapter (fine-tuned)", "Ground truth"],
                   sample_dir / "sidebyside.png")

    # Log
    def fmt(m):
        return (f"well_formed={m['well_formed']} renders={m['renders']} "
                f"tags={m['valid_tags']}/{m['total_tags']}")
    print(f"  prompt: {user_prompt[:80]}{'...' if len(user_prompt) > 80 else ''}")
    print(f"  adapter: {fmt(adapter_metrics)}")
    if base_metrics:
        print(f"  base:    {fmt(base_metrics)}")

    return {"prompt": user_prompt, "adapter": adapter_metrics, "base": base_metrics}


# -----------------------------------------------------------------
# Reporte agregado
# -----------------------------------------------------------------

def write_report(results: list[dict], args: argparse.Namespace, path: Path) -> None:
    def pct(xs, key):
        if not xs: return "–"
        return f"{100 * sum(1 for x in xs if x[key]) / len(xs):.0f}%"
    def tag_pct(xs):
        if not xs: return "–"
        v = sum(x["valid_tags"] for x in xs)
        t = sum(x["total_tags"] for x in xs)
        return f"{100 * v / t:.0f}% ({v}/{t})" if t > 0 else "–"

    adapters = [r["adapter"] for r in results]
    bases = [r["base"] for r in results if r["base"]]

    lines = [
        "# Evaluación del adapter LoRA", "",
        f"- **Adapter**: `{args.adapter_path}`",
        f"- **Base model**: `{args.base_model}`",
        f"- **N samples**: {len(results)} (held-out del test set, seed={args.seed})",
        f"- **Sampling**: temperature={args.temperature}, top_p={args.top_p}",
        "",
        "## Métricas agregadas", "",
        "| Métrica | Base | Adapter |",
        "|---|---|---|",
        f"| Contiene `<svg>` | {pct(bases, 'has_svg')} | {pct(adapters, 'has_svg')} |",
        f"| Well-formed XML | {pct(bases, 'well_formed')} | {pct(adapters, 'well_formed')} |",
        f"| Renderiza (cairosvg) | {pct(bases, 'renders')} | {pct(adapters, 'renders')} |",
        f"| Tags SVG válidos | {tag_pct(bases)} | {tag_pct(adapters)} |",
        "",
        "## Samples individuales", "",
    ]
    for i, r in enumerate(results):
        lines.append(f"### Sample {i:02d}")
        lines.append("")
        lines.append(f"**Prompt:** {r['prompt']}")
        lines.append("")
        a = r["adapter"]
        lines.append(f"- **Adapter** — well_formed={a['well_formed']}, "
                     f"renders={a['renders']}, tags={a['valid_tags']}/{a['total_tags']}")
        if r["base"]:
            b = r["base"]
            lines.append(f"- **Base** — well_formed={b['well_formed']}, "
                         f"renders={b['renders']}, tags={b['valid_tags']}/{b['total_tags']}")
        lines.append("")
        lines.append(f"![sidebyside](samples/{i:02d}/sidebyside.png)")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------
# main
# -----------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = detect_device()
    use_4bit = (args.quant == "4bit") or (args.quant == "auto" and device["recommend_4bit"])
    if use_4bit and device["name"] != "cuda":
        sys.exit("ERROR: 4-bit requiere CUDA.")
    print(f"=== Device: {device['name']}  use_4bit={use_4bit} ===")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)

    # Tokenizer + base model
    print(f"\n=== Tokenizer: {args.base_model} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"\n=== Cargando modelo base ===")
    model = build_base_model(args.base_model, use_4bit, device)
    model.config.use_cache = True
    model.eval()

    # Adapter
    print(f"\n=== Cargando adapter: {args.adapter_path} ===")
    if not Path(args.adapter_path).exists():
        sys.exit(f"ERROR: adapter_path {args.adapter_path} no existe.")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    # Sample indices del test set
    print(f"\n=== Dataset: {DATASET_ID} (split=test) ===")
    ds = load_dataset(DATASET_ID, split="test")
    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(ds), generator=rng).tolist()
    sample_idxs = perm[:args.n_samples]
    print(f"  Sampled idxs (seed={args.seed}): {sample_idxs}")

    # Loop
    results = []
    for i, idx in enumerate(sample_idxs):
        print(f"\n--- Sample {i+1}/{args.n_samples} (test idx {idx}) ---")
        sample_dir = out_dir / f"{i:02d}"
        r = run_sample(model, tokenizer, ds[idx]["messages"], sample_dir,
                       args, args.skip_baseline)
        r["idx"] = idx
        results.append(r)

    # Reporte final
    write_report(results, args, Path(args.report_path))
    print(f"\n=== Reporte escrito en {args.report_path} ===")
    print(f"=== Samples en {args.output_dir}/ ===")


if __name__ == "__main__":
    main()
