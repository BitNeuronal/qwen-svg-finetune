# qwen-svg-finetune

Fine-tune de **Qwen2.5-Coder-1.5B-Instruct** para generar SVG bien formado, usando **TRL v1.0** + **QLoRA 4-bit** en **Google Colab Free** ($0). Una sola GPU T4, ~2 horas de training, un adaptador de 72 MB listo para el Hub.

Parte del [Issue Semanal #2 de BitNeuronal](https://bitneuronal.com) — IA sin filtro para Latinoamérica.

📖 **Tutorial completo paso a paso**: [en el post del newsletter](https://bitneuronal.com/p/fine-tune-qwen-svg-trl-colab)

🤗 **Adapter en Hugging Face**: [`bitneuronal/qwen-svg-coder-lora`](https://huggingface.co/bitneuronal/qwen-svg-coder-lora)

> ⚠️ **Estado del proyecto (19 abril 2026)**: la **v1 del adapter colapsó** por contaminación del dataset base — 1.77% de las muestras (84 de 4,750) tenían `"No valid response from model."` como completion, y el fine-tune aprendió esa cadena como respuesta óptima. La **v2** está entrenando con el dataset filtrado; queda bloqueada por la cuota diaria de Colab Free. Ver ["Estado del adapter"](#estado-del-adapter-abril-2026) abajo. El código, el pipeline y la receta técnica son correctos y reutilizables — solo los números de eval del adapter actualmente publicado no son usables.

---

## ¿Qué hace este repo?

Toma un modelo de lenguaje (Qwen2.5-Coder-1.5B) que "entiende" código pero **no está especializado en SVG**, y lo entrena para que, dado un prompt en inglés describiendo una imagen, devuelva código SVG bien formado y renderizable — solo 18 M parámetros nuevos en el adaptador LoRA, sin tocar los 1,500 millones del base.

La parte interesante no es solo el pipeline (QLoRA 4-bit, TRL v1.0, una T4 gratis) sino también lo que salió mal: el primer adapter entrenado colapsó a una cadena basura que estaba en el 1.77% del dataset. Auditar y filtrar resuelve el problema. Todo documentado en el [tutorial](https://bitneuronal.com/p/fine-tune-qwen-svg-trl-colab).

## Resultados

### v1 — mode collapse (19 abril 2026)

Sanity check con 5 prompts distintos (greedy + sampling a temperature=0.7 con 5 seeds):

| Métrica | Adapter v1 |
|---|---|
| Contiene bloque `<svg>` | **0/5** |
| Well-formed XML | **0/5** |
| Renderizable con `cairosvg` | **0/5** |
| Output = `"No valid response from model."` | **5/5** |

No es azar del sampling: ante cualquier prompt, el modelo devuelve la misma cadena de 29 caracteres (ver [Estado del adapter](#estado-del-adapter-abril-2026)).

### v2 — en cola

Re-training con dataset filtrado (4,750 → ~4,666 muestras limpias) pendiente de cuota Colab. Los números finales se publican aquí y en la [model card del Hub](https://huggingface.co/bitneuronal/qwen-svg-coder-lora) cuando termine.

Ejemplos visuales (base \| adapter v2 \| ground truth) aparecerán en [`outputs/samples/`](./outputs/samples/) junto con los números.

## Estado del adapter (abril 2026)

El primer training (594 steps, loss 0.91 → 0.61, ~1h 51min) **técnicamente funcionó**: el adaptador quedó guardado, loss bajó, nada crasheó. Pero cuando le pedí un SVG simple con greedy decoding me respondió:

```
No valid response from model.
```

5 seeds distintos después, idéntico. Mode collapse. Auditoría del dataset:

```python
ds = load_dataset("thesantatitan/deepseek-svg-dataset", split="train")
bad = sum(1 for ex in ds if "No valid response from model" in ex["completion"])
print(f"{bad}/{len(ds)}")   # → 84/4750  (1.77%)
```

Los autores del dataset usaron DeepSeek para generar las respuestas. Cuando DeepSeek fallaba, el pipeline guardó el mensaje de error literal como si fuera una completion válida. 84 muestras contaminadas.

**Por qué el 1.77% colapsa al 100%**: con `assistant_only_loss=True`, la loss se concentra en los ~8 tokens del stub (vs 600–1,200 de un SVG real), el string es idéntico en las 84 muestras (atractor estable), y es trivial de memorizar (loss → 0 rápido). El optimizer encontró el atajo.

**Fix** — filtrar antes de entrenar:

```python
def is_clean(ex):
    c = ex["completion"]
    return (
        "No valid response from model" not in c
        and len(c) >= 200
        and "<svg" in c
    )

ds = ds.filter(is_clean)   # 4750 → 4666
```

El segundo training usa el dataset filtrado. Mismo código, mismos hyperparams, mismos ~2h de T4. Si v2 genera SVG válido con regularidad, el pipeline queda validado.

Desarrollo completo (causa raíz + diagnóstico con greedy + teoría de por qué 1.77% basta para colapsar): sección *"La trampa"* en el [post del newsletter](https://bitneuronal.com/p/fine-tune-qwen-svg-trl-colab).

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
# En transformers recientes, apply_chat_template(tokenize=True, return_tensors="pt")
# devuelve un BatchEncoding (dict), no un tensor. Usamos el pattern de dos pasos.
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=1024, do_sample=True,
                        temperature=0.7, top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id)
output_text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
print(output_text)
```

Para renderizar el SVG generado:

```python
import re, cairosvg
svg = re.search(r"<svg.*?</svg>", output_text, re.DOTALL).group(0)
cairosvg.svg2png(bytestring=svg.encode(), write_to="out.png")
```

## Reproducir el entrenamiento

> **Antes de entrenar:** auditá el dataset. El notebook y `src/train.py` aceptan un DatasetDict filtrado en `./dataset_clean/`; el tutorial explica cómo generarlo con un `filter()` de 4 líneas. Entrenar sobre el dataset público sin filtrar reproduce el mode collapse documentado arriba.

**Opción A — Colab Free (recomendado, $0):**

Abre [`notebooks/train_colab.ipynb`](./notebooks/train_colab.ipynb) en Colab, configura una GPU T4 (Runtime → Change runtime type → T4 GPU), y ejecuta todas las celdas. El notebook es end-to-end: instala, filtra dataset, entrena, guarda, evalúa.

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

En MPS hay que desactivar `gradient_checkpointing` (causa NaN cascade con Qwen2.5). `src/train.py` lo detecta automáticamente según el device.

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
Sí. Si ya viene en formato `messages` (chat), solo cambia `DATASET_ID`. Si viene como pares `{prompt, completion}`, hay que convertirlo a `messages` (ver [`src/format_dataset.py`](./src/format_dataset.py)). **Y audítalo antes de entrenar** — el mismo `is_clean()` funciona como plantilla: ajusta los predicados al formato de tu dataset específico.

**¿Cómo sé si mi modelo fine-tuneado tiene mode collapse?**
Post-training, antes de cualquier eval formal, corre un sanity check de 30 segundos: 3 prompts distintos, greedy decoding (`do_sample=False`). Si los 3 outputs son idénticos (o casi), o si todos son extremadamente cortos (<100 chars), hay colapso. La causa más común es contaminación del dataset con respuestas de fallo del pipeline que generó los datos. `src/eval.py --temperature 0 --n_samples 3` hace exactamente eso.

**¿El adapter del Hub funciona o no?**
En el momento de escribir esto (abril 2026): **no**. El commit actual es v1 con mode collapse. v2 con dataset filtrado está en cola; cuando se publique, la model card del Hub y esta sección quedarán actualizadas con los números reales.

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
