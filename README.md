# qwen-svg-finetune

Fine-tune de **Qwen2.5-Coder-1.5B-Instruct** para generar SVG bien formado usando **TRL v1.0** + **QLoRA 4-bit** en **Google Colab Free**.

Parte del Issue Semanal #2 de [BitNeuronal](https://bitneuronal.com) — IA sin filtro para Latinoamérica.

> Este README es un placeholder. El tutorial completo se publicará junto con el issue del domingo 19 de abril 2026. Mientras tanto, consulta el `BRIEF.md` para el alcance.

## Stack

- Modelo base: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Dataset: `thesantatitan/deepseek-svg-dataset`
- Método: SFT + QLoRA 4-bit con TRL v1.0
- Hardware: Google Colab Free (T4 16GB) — $0

## Estructura (en progreso)

```
.
├── notebooks/        # Colab notebook principal
├── src/              # Scripts Python sueltos (train.py, eval.py)
├── samples/          # Prompts y SVGs de ejemplo
├── BRIEF.md          # Alcance del tutorial
└── README.md         # Quick-start (pendiente)
```

## Licencia

MIT — ver [LICENSE](./LICENSE).
