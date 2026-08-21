# CADGenesis-LM v2.0 — Complete User Guide

> A beginner-friendly guide to the whole project: what it is, why it was built,
> how each part works, and how to run it step by step.
>
> Companion documents:
> - [`../README.md`](../README.md) — quick-start commands.
> - [`project_report.md`](project_report.md) — deep technical report.
> - [`implementation_plan.md`](implementation_plan.md) — training readiness notes.

---

## 1. What Is This Project?

**CADGenesis-LM is a language model (LLM) that turns plain English text into
Parametric 3D CAD designs.**

Example input:

> "Create a steel box 50mm wide, 20mm high and 10mm deep."

Example output (a compact CAD construction sequence called **TOON**):

```text
id|feature|width|height|depth
int|str|float|float|float
1|BOX|50.0|20.0|10.0
```

In other words: you type what you want to design, and the model generates the
steps needed to build that 3D model in CAD software. It was created because
ordinary LLMs are:

1. **Expensive** — CAD designs written as long JSON/XML strings eat up lots of
   the "context window" (the amount of text the model can see), raising API cost.
2. **Unreliable** — generic LLMs frequently produce invalid geometry (e.g.
   negative wall thickness, impossible shapes).

CADGenesis-LM solves both problems with its own compact **TOON format** and a
**geometry-aware model trained with AI "teacher" models**.

---

## 2. Why Was It Built? (Purpose & Motivation)

The project has three main goals:

| Goal | Problem it fixes |
| :--- | :--- |
| **Save tokens (money)** | CAD data is normally written as verbose JSON/XML. This project invents **TOON**, a compact format that cuts token usage by **30–50%**. |
| **Generate valid geometry** | A **Quality Filter** checks every design and rejects invalid shapes, negative dimensions, unsafe parts, etc. |
| **Learn CAD without manual data** | A **Teacher-Student distillation** pipeline lets big AI models (GPT-4o, DeepSeek, Qwen) generate training data automatically, which a small "student" model then learns from. |

The final result is a **small, fast, specialized CAD model** trained to do one
job well, instead of paying a huge general-purpose model every time.

---

## 3. The Big Ideas (Explained Simply)

### 3.1 TOON — the compact data format
JSON repeats the same key names again and again:

```json
{"id":1, "feature":"BOX", "width":50.0, "height":20.0, "depth":10.0}
{"id":2, "feature":"BOX", "width":30.0, "height":10.0, "depth":5.0}
```

TOON writes the keys **once**, then only the values:

```text
id|feature|width|height|depth
int|str|float|float|float
1|BOX|50.0|20.0|10.0
2|BOX|30.0|10.0|5.0
```

Less text = fewer tokens = cheaper and faster LLM calls.

### 3.2 Teacher–Student Distillation
- **Teacher** = a big, smart AI (GPT-4o, DeepSeek, Qwen) that writes CAD designs.
- **Quality Filter** = a checker that deletes anything invalid.
- **Student** = your own small model (`GeometryAwareTransformer`) that learns
  from the teacher's good examples.
- **Soft loss (KL divergence)** = the student also learns the teacher's "soft
  confidence" for each answer, not just the final answer — this transfers more
  knowledge than plain copying.
- **Self-improvement loop** = when the student gets something wrong, the teacher
  is asked to re-explain the mistake, and the student learns from the correction.

### 3.3 A "self-designing" model
The model doesn't just train — it can also **improve its own architecture**:
- **NAS (Neural Architecture Search)** — tries different layer counts/sizes and
  keeps the best one.
- **Dynamic layer routing** — decides per token which layers to skip.
- **MoE (Mixture of Experts)** — grows extra "expert" sub-networks.
- **Layer pruning** — removes weak layers, reversibly (safe to undo).
- **Auto rollback** — if a change makes things worse, it restores the last good
  weights automatically.

### 3.4 Memory pools
Like human working memory: the model keeps 8 pools (working, session, project,
user, CAD, engineering, manufacturing, simulation) and **reads from and writes
back to them inside every layer**, so it can remember design context across the
whole generation.

### 3.5 Internal multi-agent system
Inside the model, 8 specialist "agents" (planner, geometry, constraint,
manufacturing, validation, optimization, assembly, simulation) each look at the
design from their own viewpoint and combine their opinions to produce the final
output.

---

## 4. Project Layout — What Every File Does

### Scripts (`scripts/`, `tools/`)
| File | What it does |
| :--- | :--- |
| `scripts/train.py` | Root training entry point (forwards to `cadgenesis.cli.train`). |
| `scripts/distill_train.py` | Runs the full Teacher–Student distillation pipeline. |
| `tools/openai_finetune.py` | Prepare / run OpenAI-style fine-tuning. |
| `tools/prepare_finetune_from_toon.py` | Turn TOON data into fine-tuning examples. |

### Examples & SDK (`examples/`, `sdk/`)
| File / folder | What it is |
| :--- | :--- |
| `examples/mini_demo/` | The tiny verification model (`data.py`, `model.py`, `generate.py`). |
| `examples/toon_demo.py`, `examples/langchain_integration.py` | Small demos of TOON usage + LangChain hooks. |
| `examples/app_fastapi.py` | FastAPI web service exposing TOON + LLM endpoints. |
| `examples/*.json`, `examples/*.jsonl` | Example / training data files. |
| `sdk/toon.py`, `sdk/toon_extended.py` | TOON serializer (Python). |
| `sdk/toon.ts`, `sdk/toon_node_demo.ts` | TOON serializer (TypeScript). |

### The core package `src/cadgenesis/`
| Folder | What it contains |
| :--- | :--- |
| `transformer/` | The transformer **built from scratch**: `geometry_transformer.py` (main model), `attention.py` (6 specialized heads), `transformer_block.py` (custom layers), `positional.py`, `moe.py`, `efficient_attention.py`, `interaction.py`, and `self_designing/` (NAS, routing, pruning, rollback). |
| `tokenizer/` | The **Autonomous CAD Tokenizer**: `vocabulary.py`, `evolution.py`, `statistics.py`, `versioning.py`, `toon_backend.py`, plus CAD-family token vocabularies. |
| `training/` | `trainer.py` — the training engine (`CADTrainer`). |
| `distillation/` | Teacher interface, quality filter, data generation, soft-KL loss, self-improvement loop. |
| `memory/` | `memory_pools.py` — the 8 layer-integrated memory pools. |
| `agents/` | `multi_agent_system.py` — the 8 internal specialist agents. |
| `execution/` | `execution_engine.py` — CAD geometry validation. |
| `alignment/` | `constitutional_ai.py` — safety rules (e.g. minimum safety factor). |
| `confidence/` | `confidence_engine.py` — model confidence scoring. |
| `adapters/` | `manager.py` (adapter system for fast fine-tuning) + `lora.py` (LoRA/QLoRA low-rank fine-tuning). |
| `reasoning/` | `neuro_symbolic.py` — neuro-symbolic reasoning helpers. |
| `config/` | `cad_config.py` — `CADConfig`, every setting/hyperparameter lives here. |
| `cli/train.py` | Package training CLI (`python -m cadgenesis.train`). |
| `inference/engine.py` | `CADInferenceEngine` — greedy/beam inference with TOON output. |

### Other
| File / folder | What it is |
| :--- | :--- |
| `tests/` | Pytest suite (300 passing tests). |
| `benchmarks/` | Speed benchmarks for attention and tokenizer. |
| `outputs/` | Saved checkpoints and training output. |
| `notebooks/CADGenesis_Mini_Colab.ipynb` | Colab notebook version of the mini model. |
| `docs/*.md`, `README.md` | Documentation (README, guide, project report, TOON docs, LoRA notes). |
| `pyproject.toml`, `requirements.txt` | Python packaging / dependencies. |

---

## 5. How It Is Built (From Scratch or Not?)

**Honest answer: the real model is built from scratch; one tiny test file uses a
ready-made PyTorch part.**

| Piece | Built from scratch? | Explanation |
| :--- | :--- | :--- |
| `src/cadgenesis/transformer/` (real model) | ✅ Yes | Self-attention, transformer blocks, positional encodings, MoE — all written by hand using only low-level PyTorch (`nn.Module`, `nn.Linear`, `nn.Embedding`, `torch.matmul`). |
| `examples/mini_demo/model.py` (mini test model) | ⚠️ One exception | `model.py` uses PyTorch's ready-made `torch.nn.Transformer`. You did *not* write that file's attention math yourself; you wrapped a built-in component. |
| Training, tokenizer, TOON, distillation, inference | ✅ Yes | All custom code. |

So you can honestly say: **"The production model and the entire pipeline are
built from scratch. Only the throwaway mini test model reuses a built-in
PyTorch transformer module."**

---

## 6. How to Install and Run

### 6.1 Requirements
- Python **3.10 or newer**
- PyTorch **2.0+**
- Optional extras: `tokenizers`, `transformers` (BPE), `fastapi`, `uvicorn`, `openai`, `pydantic`

```bash
# Install base dependencies
pip install torch>=2.0

# Install dev/test dependencies
pip install pytest pytest-cov

# Install API/service dependencies (optional)
pip install -r requirements.txt
```

### 6.2 Train the mini verification model (fast, CPU-friendly)
```bash
python -m cadgenesis.train
```
Custom settings:
```bash
python -m cadgenesis.train --epochs 10 --batch-size 32 --train-size 1000 --valid-size 200 --output-dir outputs/cadgenesis_train
```

### 6.3 Train the full v2.0 model (needs a GPU for real training)
```bash
python -m cadgenesis.train --model-size full --epochs 5 --batch-size 8 --train-size 200 --valid-size 50 --output-dir outputs/cadgenesis_train_full
```

### 6.4 Resume training from a checkpoint
```bash
python -m cadgenesis.train --resume-from outputs/cadgenesis_train/best_checkpoint.pt
```

### 6.5 Run the Teacher–Student distillation pipeline
```bash
python scripts/distill_train.py --num-samples 10 --epochs 1
```
This runs the 5 phases: teacher setup → data generation → student model →
distillation training → self-improvement loop.

### 6.6 Run the tests
```bash
python -m pytest -q
```
Expect **300 passing tests**.

### 6.7 Run the benchmarks (optional)
```bash
python benchmarks/attention_benchmarks.py --max-len 256 --reps 2
python benchmarks/tokenizer_benchmarks.py --reps 10
```

### 6.8 Serve the FastAPI app (optional)
```bash
uvicorn examples.app_fastapi:app --reload --port 8080
```

---

## 7. How to Use the Model for Inference

Use the inference engine to generate a CAD design from text:

```python
from cadgenesis.config import CADConfig
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer
from cadgenesis.tokenizer.cad_tokenizer import AutonomousCADTokenizer
from cadgenesis.inference import CADInferenceEngine

# Build model + tokenizer
cfg = CADConfig.mini()
tokenizer = AutonomousCADTokenizer.build_mini()
model = GeometryAwareTransformer(cfg)

# Create the inference engine
engine = CADInferenceEngine(model, tokenizer)

# Generate a CAD sequence
result = engine.greedy("create a steel box 50mm wide", max_len=64)
# or engine.beam("...", beam_width=3)

print(result.tokens)  # generated CAD token list
print(result.confidence)  # model confidence (0..1)
print(result.toon)  # TOON-serialized result
```

### Self-designing model (advanced)
```python
from cadgenesis.transformer.self_designing import SelfDesigningTransformer

model = SelfDesigningTransformer(cfg)
model.grow_experts(1)  # add an expert to every MoE block
model.prune_layers(0.25)  # reversibly prune weak layers
best, score, summary = model.search_architecture(dataset)  # run NAS
model.snapshot(metric)
model.check_performance(metric)  # auto rollback
```

### Use the Autonomous CAD Tokenizer
```python
tok = AutonomousCADTokenizer.build()
report = tok.evolve(sequences)  # grow vocabulary from data
tok.serialize_to_toon(tok.encode_cad_sequence(["PRIM_BOX", "EXTRUDE"]))
```

---

## 8. Glossary (Simple Word Meanings)

| Term | Simple meaning |
| :--- | :--- |
| **LLM** | Large Language Model — an AI that predicts the next word/token from text. |
| **Token** | A small chunk of text (a word or part of a word) the model reads/writes. |
| **Transformer** | The architecture (structure) most modern LLMs use — it lets every word "attend" to every other word. |
| **Attention** | The mechanism a transformer uses to decide which other words matter most. |
| **Embedding** | Turning a word/token into a list of numbers the model can do math on. |
| **Encoder** | The part that reads the input text. |
| **Decoder** | The part that writes the output sequence. |
| **Encoder-decoder** | A model that reads language input and generates output — exactly what CADGenesis-LM is. |
| **Tokenization** | Splitting text into tokens. |
| **Vocabulary** | The set of all tokens the model knows. |
| **BPE** | Byte-Pair Encoding — a common way to build a token vocabulary. |
| **TOON** | The compact pipe-delimited format invented here (Token-Optimized Object Notation). |
| **Parametric CAD** | CAD design based on parameters (width, height, depth) rather than fixed shapes. |
| **B-Rep** | Boundary Representation — the math way CAD stores 3D shape surfaces. |
| **Manifold / non-manifold** | Manifold = physically valid solid surface; non-manifold = broken/overlapping edges a real object can't have. |
| **Distillation** | Teaching a small model by copying a big model's knowledge. |
| **Teacher model** | The big, smart AI (GPT-4o, DeepSeek, Qwen) used to generate knowledge. |
| **Student model** | The small model (yours) that learns from the teacher. |
| **KL divergence** | A math measure of how different two probability distributions are; used in the soft distillation loss. |
| **Fine-tuning / PEFT** | Continuing to train a model on a specialized task; PEFT = lightweight fine-tuning. |
| **LoRA / QLoRA** | Cheap ways to fine-tune by training tiny low-rank adapters instead of all weights. |
| **MoE** | Mixture of Experts — many small "expert" networks with a router that picks the best ones per token. |
| **NAS** | Neural Architecture Search — automatically finding the best model structure. |
| **AMP** | Automatic Mixed Precision — faster training by using low-precision numbers where safe. |
| **Checkpoint** | A saved snapshot of the model's weights. |
| **Resume** | Continuing training from a checkpoint instead of starting over. |
| **Loss** | A number telling how wrong the model is; training reduces it. |
| **Hyperparameter** | A setting chosen before training (learning rate, layer count, batch size...). |

---

## 9. Verification Summary (What Is Proven to Work)

- ✅ Full **training pipeline** verified end-to-end (mini + full architecture) on
  synthetic data — see `implementation_plan.md`.
- ✅ **300 passing unit tests** via `python -m pytest -q`.
- ✅ **TOON** Python + TypeScript serializers with round-trip tests.
- ✅ **Distillation pipeline** runs all 5 phases and reports loss + pass rate.
- ✅ **Inference engine** (greedy + beam) with confidence scoring and TOON output.
- ✅ **Self-designing** subsystems (NAS, routing, pruning, MoE growth, rollback)
  covered by 28 dedicated tests.
- ✅ Attention and tokenizer **micro-benchmarks** in `benchmarks/`.

---

## 10. Frequently Asked Questions

**Q: Did you build the LLM from scratch?**
A: The production model (`src/cadgenesis/transformer/`) and the whole pipeline
are built from scratch with PyTorch. The only exception is the throwaway mini
test model in `examples/mini_demo/model.py`, which reuses PyTorch's built-in
`torch.nn.Transformer`.

**Q: Do I need a GPU?**
A: The mini model trains fine on CPU. The full model is designed for GPU
training (CUDA), with automatic CPU fallback for small runs.

**Q: How is this different from a normal LLM?**
A: Normal LLMs read text only. CADGenesis-LM is trained on CAD token
sequences, understands geometry coordinates, validates designs, uses compact
TOON output, and can redesign its own architecture.

**Q: How do I check the model's confidence in a design?**
A: Use the inference engine — `result.confidence` gives the mean model
confidence (0–1) over the generated tokens.

**Q: Where do I start first?**
A: Run `python -m pytest -q` to verify the environment, then
`python -m cadgenesis.train` for a mini training run, then try the inference
example in Section 7.
