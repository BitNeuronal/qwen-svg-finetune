"""
check_env.py — Verifica que el entorno de qwen-svg-finetune está listo.

Uso:
    python src/check_env.py

Imprime:
    - Versión de Python
    - Versiones de torch, transformers, trl, peft, datasets, accelerate
    - Device disponible (CUDA / MPS / CPU)
    - Si huggingface_hub tiene sesión iniciada
    - Si bitsandbytes está disponible (opcional, solo CUDA)
    - Si cairosvg renderiza correctamente
"""

from __future__ import annotations

import platform
import sys


def _header(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    exit_code = 0

    _header("Sistema")
    print(f"Python       : {sys.version.split()[0]}  ({platform.machine()} / {platform.system()})")

    _header("Librerías core")
    try:
        import torch
        print(f"torch        : {torch.__version__}")
    except ImportError as e:
        print(f"torch        : FALTA  ({e})")
        exit_code = 1

    for pkg in ("transformers", "trl", "peft", "datasets", "accelerate", "huggingface_hub"):
        try:
            mod = __import__(pkg)
            print(f"{pkg:<13}: {getattr(mod, '__version__', '?')}")
        except ImportError as e:
            print(f"{pkg:<13}: FALTA  ({e})")
            exit_code = 1

    _header("Device")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA         : OK  ({torch.cuda.get_device_name(0)})")
            print("→ Modo recomendado: QLoRA 4-bit con bitsandbytes")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print("MPS (Apple)  : OK")
            print("→ Modo recomendado: LoRA fp16/bf16 (sin bitsandbytes)")
        else:
            print("Device       : solo CPU")
            print("→ El training será lento; considera Colab T4 para el run completo.")
    except Exception as e:
        print(f"Device       : error ({e})")

    _header("Hugging Face Hub")
    try:
        from huggingface_hub import whoami
        info = whoami()
        print(f"Logged in as : {info.get('name', '?')}  (email: {info.get('email', '?')})")
    except Exception as e:
        print(f"No hay sesión iniciada.  ({e})")
        print("→ Corre: hf auth login   (CLI nuevo en huggingface_hub >= 1.0)")
        print("  Docs: https://huggingface.co/docs/huggingface_hub/v1.10.1/guides/cli")

    _header("bitsandbytes (opcional, solo CUDA)")
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes : {bnb.__version__}  (instalado)")
    except ImportError:
        print("bitsandbytes : no instalado  (esperado en Mac; requerido en Colab)")
    except Exception as e:
        print(f"bitsandbytes : instalado pero con error  ({e})")

    _header("cairosvg (renderizado SVG)")
    try:
        import cairosvg
        svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><circle cx="16" cy="16" r="12" fill="red"/></svg>'
        png_bytes = cairosvg.svg2png(bytestring=svg)
        print(f"cairosvg     : OK  (render -> {len(png_bytes)} bytes PNG)")
    except ImportError:
        print("cairosvg     : no instalado   -> pip install cairosvg")
        exit_code = 1
    except OSError as e:
        msg = str(e)
        if "libcairo" in msg or "cairo-2" in msg:
            print("cairosvg     : la libería nativa de Cairo no se encuentra.")
            if platform.system() == "Darwin":
                print("  Mac (Apple Silicon): instala y expone el path:")
                print("    brew install cairo pango libffi")
                print('    export DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix)/lib:${DYLD_FALLBACK_LIBRARY_PATH}"')
            else:
                print("  Linux: sudo apt-get install -y libcairo2")
        else:
            print(f"cairosvg     : error al renderizar  ({e})")
        exit_code = 1
    except Exception as e:
        print(f"cairosvg     : error al renderizar  ({e})")
        exit_code = 1

    _header("Resultado")
    if exit_code == 0:
        print("✓ Entorno listo.")
    else:
        print("✗ Hay problemas. Revisa los mensajes arriba e instala lo que falte.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
