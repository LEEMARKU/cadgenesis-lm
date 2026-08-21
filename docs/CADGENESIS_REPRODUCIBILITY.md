# CADGenesis-LM Reproducibility Procedure

Status: **Verified** — `research/reproducibility.py` toolkit tested; synthetic dataset deterministic for a given seed.

## 1. Purpose

Every experiment must be repeatable: same inputs + same environment → same outputs (bit-identical where determinism allows). This document is the operational procedure.

## 2. Environment Capture (required for every run)

Before any training/eval run, capture the environment to `environment.json`:

```python
from cadgenesis.research.reproducibility import EnvironmentCapture, capture_pip_freeze

env = EnvironmentCapture.capture()        # python version, platform, packages, env vars (secrets redacted), cwd, command
env.save("outputs/experiments/<run>/environment.json")
with open("outputs/experiments/<run>/requirements-frozen.txt", "w") as fh:
    fh.write(capture_pip_freeze())
```

- `REDACT_TOKENS = ("KEY", "SECRET", "PASSWORD", "TOKEN", "AUTH")` — env vars containing these are written as `***`.
- Verify manually: `python --version`, `nvidia-smi`, `pip list`.

## 3. Seeding (required for every run)

```python
from cadgenesis.research.reproducibility import DeterministicTraining, SeedRegistry

with DeterministicTraining(seed=42):
    ...  # training / generation
```

`DeterministicTraining` sets Python/NumPy/torch/CUDA seeds, enables
`torch.use_deterministic_algorithms(True)`, and sets float32 matmul precision to
`"highest"`. Use `SeedRegistry(42).seed_for("dataset")` / `seed_for("model")` for
stable per-component derived seeds.

CLI entry points already accept `--seed` (default 42): `cadgenesis train --seed`,
`cadgenesis eval --seed`.

## 4. Dataset Determinism (verified)

```python
from cadgenesis.datasets.cad_program_synth import build_synthetic_records

r1 = build_synthetic_records(100, seed=7)
r2 = build_synthetic_records(100, seed=7)   # byte-identical to r1
```

- Verified: same seed → identical records; different seed → different records.
- 18 templates; only validator-accepted programs enter the dataset (honest rejection of
  unsupported token grammars — FILLET/BOLT/WEIGHT/SHAFT templates are excluded by the
  analytic validator, by design).
- Record the dataset seed + template set hash in the run manifest.

## 5. Reproducibility Checklist

| Step | Action | Evidence |
|------|--------|----------|
| 1 | Freeze env: `environment.json` + pip freeze | file in run dir |
| 2 | Seed: `DeterministicTraining(seed)` | seed in manifest |
| 3 | Dataset: record `seed`, template count, token coverage | 18 templates, 51 unique tokens (500 records) |
| 4 | Checkpoint: store `config` dict + `vocab_tokens` + `model_state_dict` | checkpoint layout |
| 5 | Eval: fixed `seed` + fixed benchmark task list | benchmark doc |
| 6 | Note hardware/driver (CUDA determinism not guaranteed) | `environment.json` |

## 6. Known Limits

- GPU (CUDA) runs are not guaranteed bit-identical across devices/drivers; CPU verified.
- Distributed (FSDP) runs need a rank-ordered seed scheme (P2 item).
- No golden-master training snapshot yet (requires P2 hardware).

## 7. Tooling

| Tool | Purpose | Command |
|------|---------|---------|
| `EnvironmentCapture` | snapshot env | `cadgenesis.research.reproducibility` |
| `SeedRegistry` | derived per-key seeds | same module |
| `DeterministicTraining` | enforce determinism ctx | same module |
| `cadgenesis train --seed N` | seeded training CLI | CLI |
| `build_synthetic_records(n, seed)` | seeded dataset | `cadgenesis.datasets.cad_program_synth` |