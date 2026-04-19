# qwen-svg-finetune

Fine-tune de **Qwen2.5-Coder-1.5B-Instruct** para generar SVG bien formado, usando **TRL v1.0** + **QLoRA 4-bit** en **Google Colab Free** ($0). Una sola GPU T4, ~2 horas de training, un adaptador de 72 MB listo para el Hub.

Parte del [Issue Semanal #2 de BitNeuronal](https://bitneuronal.com) — IA sin filtro para Latinoamérica.

📖 **Tutorial completo paso a paso**: [en el post del newsletter](https://bitneuronal.com/p/fine-tune-qwen-svg-trl-colab)

🤗 **Adapter en Hugging Face**: [`bitneuronal/qwen-svg-coder-lora`](https://huggingface.co/bitneuronal/qwen-svg-coder-lora)

---

## ¿Qué hace este repo?

Toma un modelo de lenguaje (Qwen2.5-Coder-1.5B) que "entiende" código pero **no está especializado en SVG**, y lo entrena para que, dado un prompt en inglés describiendo una imagen, devuelva código SVG bien formado y renderizable.

El modelo fine-tuned pasa de generar SVG caóticos (~15–25% renderizables) a producir SVG consistentes (~TBD% renderizables, ver tabla de resultados abajo) sin tocar los 1,500 millones de parámetros del base — solo 18 M parámetros nuevos en el adaptador LoRA.

## Resultados

Evaluación sobre 5 prompts held-out del test set (`thesantatitan/deepseek-svg-dataset`). Métricas sobre el output crudo de cada modelo con `temperature=0.7`, `top_p=0.95`:

| Métrica | Base (sin fine-tune) | Adapter (fine-tuned) |
|---|---|---|
| Contiene bloque `<svg>` | TBD% | TBD% |
| Well-formed XML | TBD% | TBD% |
| Renderizable con `cairosvg` | TBD% | TBD% |
| Tags SVG válidos | TBD% | TBD% |

<!-- TODO: completar tabla con números reales de outputs/eval_report.md -->

Ejemplos visuales (base \| adapter \| ground truth) en [`outputs/samples/`](./outputs/samples/).

## Quick start: usar el adapter entrenado

Tres líneas después del install:

```bash
pip install transformers peft bitsandbytes accelerate cairosvg
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    quantization_config=bnb, device_map="auto", dtype=torch.float16,
)
model = PeftModel.from_pretrained(base, "bitneuronal/qwen-svg-coder-lora")

messages = [
    {"role": "system", "content": "Respond in the following format:\n<think>\n...\n</think>\n...\n<generated_svg>\n...\n</generated_svg>"},
    {"role": "user", "content": "Generate svg code for an image that looks like: a red circle on a blue background"},
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
output = model.generate(inputs, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.9)
print(tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True))
```

Para renderizar el SVG generado:

```python
import re, cairosvg
svg = re.search(r"<svg.*?</svg>", output_text, re.DOTALL).group(0)
cairosvg.svg2png(bytestring=svg.encode(), write_to="out.png")
```

## Reproducir el entrenamiento

**Opción A — Colab Free (recomendado, $0):**

Abre [`notebooks/train_colab.ipynb`](./notebooks/train_colab.ipynb) en Colab, configura una GPU T4 (Runtime → Change runtime type → T4 GPU), y ejecuta todas las celdas. El notebook es end-to-end: instala, entrena, guarda, evalúa.

**Opción B — Local con GPU NVIDIA:**

```bash
git clone https://github.com/BitNeuronal/qwen-svg-finetune.git
cd qwen-svg-finetune

python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Si tienes CUDA, descomenta `bitsandbytes>=0.43.0` en requirements.txt

python src/train.py                 # training completo
python src/eval.py --n_samples 5    # evaluación comparativa
```

**Opción C — Local en Apple Silicon (MPS):**

Soportado pero **lento** (~7 min/step en M2 Max por la ausencia de cuantización 4-bit en MPS). Útil para iterar código, no para training real.

```bash
python src/train.py --smoke  # 50 steps, ~30 min en M2 Max
```

Detalles y los bugs específicos de MPS (NaN cascade con gradient_checkpointing) están explicados en [el tutorial completo](https://bitneuronal.com/p/fine-tune-qwen-svg-trl-colab).

## Estructura del repo

```
qwen-svg-finetune/
├── notebooks/
│   └── train_colab.ipynb     # Training + eval end-to-end en Colab T4
├── src/
│   ├── check_env.py          # Verifica device, Hub, Cairo
│   ├── explore_dataset.py    # Stats del dataset (features, distribuciones)
│   ├── format_dataset.py     # Inspeccionar chat_template aplicado
│   ├── train.py              # Training standalone (CUDA / MPS / CPU)
│   └── eval.py               # Evaluación adapter vs. base, 3 métricas
├── outputs/
│   ├── samples/              # PNGs + SVGs generados por eval.py
│   └── eval_report.md        # Reporte agregado de eval.py
├── samples/                  # Prompts y SVG de referencia (curados)
├── requirements.txt
├── LICENSE
└── README.md
```

## Arquitectura del training

La receta estándar QLoRA moderna (probada en Turing T4) queda así:

| Componente | Dtype | Cómo se configura |
|---|---|---|
| Base model weights | nf4 (4-bit) | `BitsAndBytesConfig(load_in_4bit=True)` |
| Compute (matmul) | fp16 | `bnb_4bit_compute_dtype=float16` + `fp16=True` en SFTConfig |
| Embeddings / norms / lm_head | fp16 | `dtype=torch.float16` en `from_pretrained` |
| LoRA adapters | fp32 | `prepare_model_for_kbit_training` + cast manual post-SFTTrainer |
| Optimizer states (AdamW) | fp32 | default |
| Gradient scaling | GradScaler (fp16 AMP) | automático con `fp16=True` |

El **por qué** del cast manual de LoRA a fp32 (bug sutil entre `bitsandbytes.Linear4bit`, PEFT y GradScaler) y la **razón** de `fp16` en Turing en vez de `bf16` (Tensor Cores) están desarrollados en el tutorial.

## Hyperparams usados

- `lora_r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `target_modules=[q,k,v,o,gate,up,down]_proj`
- `per_device_train_batch_size=2`, `gradient_accumulation_steps=4` → effective batch = 8
- `learning_rate=2e-4`, `lr_scheduler="cosine"`, `warmup_steps=20`, `num_train_epochs=1`
- `max_length=2048`, `packing=False`, `assistant_only_loss=True`
- `gradient_checkpointing=True` (CUDA), `=False` (MPS — causa NaN con Qwen2.5)
- `save_strategy="steps"`, `save_steps=100`, `save_total_limit=3`

## FAQ rápida

**¿Por qué Qwen2.5-Coder y no Llama/Mistral/Phi?**
Qwen2.5-Coder ya fue entrenado sobre ~4T tokens incluyendo HTML/XML, así que la estructura sintáctica de SVG le es familiar. El fine-tune refuerza el patrón, no lo enseña desde cero.

**¿Por qué 1.5B y no 7B?**
El 1.5B entra en la T4 con batch decente y entrena en ~2 h gratis. El mismo pipeline escala a 7B en una A100 o en Colab Pro, cambiando solo `MODEL_ID`.

**¿Por qué fp16 y no bf16 en Colab?**
Porque la T4 es Turing (compute capability 7.5) y sus Tensor Cores solo aceleran fp16. En bf16 cae a FP32 emulado y el training se vuelve 8× más lento (medido).

**¿Puedo usar otro dataset?**
Sí. Si ya viene en formato `messages` (chat), solo cambia `DATASET_ID`. Si viene como pares `{prompt, completion}`, hay que convertirlo a `messages` (ver [`src/format_dataset.py`](./src/format_dataset.py)).

## Contribuir

Pull requests bienvenidos. Áreas donde ayuda extra:

- Eval con un LLM-as-judge (calidad visual, no solo well-formedness).
- Experimentos con `r` más alto o `target_modules` distintos.
- Pipeline para dataset más grande (~50 k ejemplos) en una GPU mayor.
- Port a otros modelos base (SmolLM, Phi-3, Gemma-2).

## Links y referencias

- Tutorial completo: [BitNeuronal #2 — Fine-tune Qwen para SVG](https://bitneuronal.com/p/fine-tune-qwen-svg-trl-colab)
- TRL docs: https://huggingface.co/docs/trl
- QLoRA paper: https://arxiv.org/abs/2305.14314
- LoRA paper: https://arxiv.org/abs/2106.09685
- Dataset: [`thesantatitan/deepseek-svg-dataset`](https://huggingface.co/datasets/thesantatitan/deepseek-svg-dataset)
- Modelo base: [`Qwen/Qwen2.5-Coder-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)

## Licencia

Código bajo [MIT](./LICENSE). El adapter entrenado hereda la licencia Apache-2.0 del modelo base (Qwen2.5-Coder). El dataset `thesantatitan/deepseek-svg-dataset` viene bajo su propia licencia — revisa la model card antes de usarlo con fines comerciales.
