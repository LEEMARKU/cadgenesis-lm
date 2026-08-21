# Pillar 3 — Multimodal Understanding

Implementation report for the **Multimodal Understanding** pillar of the
CADGenesis-LM v6.0 roadmap (`docs/v6_roadmap.md`, milestone **M19**).
Delivers one shared engineering embedding space for eleven modalities plus
cross-modal attention, five fusion strategies, and platform integration.

## 1. Scope

| # | Capability | Module |
|---|-----------|--------|
| 1 | Modality enum, specs, default feature dims, family registry | `multimodal/common.py` |
| 2 | Shared engineering embedding space (modality adapters, projection heads) | `multimodal/embeddings.py` |
| 3 | Per-modality encoders (TEXT, CAD, DRAWING, SKETCH, IMAGE, PDF, POINT_CLOUD, MESH, AUDIO, VIDEO, SENSOR) | `multimodal/encoders/` |
| 4 | Cross-modal attention layers, registry, engine | `multimodal/cross_modal.py` |
| 5 | Five fusion strategies (concat, mean, weighted, attention, gated) | `multimodal/fusion.py` |
| 6 | Facade `MultimodalSystem` (encode / embed / fuse / project) | `multimodal/multimodal.py` |
| 7 | Integration with CAD docs, memory, world model, reasoning | `multimodal/integration.py` |
| 8 | Metrics, benchmark, tests, config | `evaluation/multimodal_metrics.py`, `benchmarks/multimodal_benchmarks.py`, `tests/multimodal/` |

## 2. Module layout

```
multimodal/
├── __init__.py          # public exports
├── common.py            # Modality, ModalitySpec, supported_families, DEFAULT_FEATURE_DIMS
├── embeddings.py        # SharedEngineeringEmbeddingSpace, ModalityAdapter(+Registry), ProjectionHead
├── cross_modal.py       # CrossModalLayer(+Registry), CrossModalAttention, CrossModalEngine
├── fusion.py            # FusionEngine (Strategy: CONCAT/MEAN/WEIGHTED/ATTENTION/GATED)
├── multimodal.py        # MultimodalSystem facade + MultimodalEncoding
└── integration.py       # MultimodalIntegrator (CADDocument <-> modality features)
encoders/
├── base.py              # EncoderProtocol, ModalityEncoder base
├── text.py              # mean-pooled embedding encoder
├── cad.py               # feature/material/mass/sensor-family encoder
├── drawing.py / sketch.py / image.py / pdf.py
├── point_cloud.py / mesh.py
├── audio.py / video.py / sensor.py
└── __init__.py
```

## 3. API tables

### `Modality` (enum)

`TEXT, CAD, DRAWING, SKETCH, IMAGE, PDF, POINT_CLOUD, MESH, AUDIO, VIDEO, SENSOR`
(11 values, `modality_from_name` maps strings both directions).

### `MultimodalSystem`

| Method | Description |
|--------|-------------|
| `from_config(MultimodalConfig)` | Build system from the config sub-object |
| `encode(features)` | Raw per-modality features keyed by `Modality` |
| `embed(modal_features)` | Project every modality into the shared space |
| `fuse(embedded)` | Combine via the configured fusion strategy |
| `project(fused, target)` | Project fused vector onto one modality subspace |
| `raw_feature_dims()` | Feature dimensions per modality (config or defaults) |

### `SharedEngineeringEmbeddingSpace`

| Method | Description |
|--------|-------------|
| `embed(modality, features)` | `(B, D) -> (B, E)` through adapter + projection |
| `similarity(a, b)` | Cosine similarity matrix `(B_a, B_b)` |
| `registered()` / `register_adapter` | Adapter registry management |

### `CrossModalEngine`

| Method | Description |
|--------|-------------|
| `forward(embedded, pairs=HEADLINE_PAIRS)` | Attention over aligned modality pairs |
| `stacks()` | Enumerate the registered pair stacks |
| `embeddings(encoded)` | Return per-modality embeddings after attention |

### `FusionEngine`

`forward(embedded) -> FusionResult(.fused)` under `FusionStrategy` ∈
`CONCAT, MEAN, WEIGHTED, ATTENTION, GATED`.

## 4. Encoder family map

| Modality | Input document | Output features |
|----------|---------------|-----------------|
| TEXT | `str` / `list[str]` | mean-pooled token embeddings |
| CAD | `CADDocument` | feature-kind, material, mass, sensor-family one-hots |
| DRAWING / SKETCH | `DrawingDocument` / `SketchDocument` | entity-kind + geometry counts |
| IMAGE | `ImageDocument(C,H,W)` | pooled CNN features |
| PDF | `PDFDocument(pages=[...])` | per-page text hashes + page count |
| POINT_CLOUD | `PointCloudDocument` | occupancy/centroid histograms |
| MESH | `MeshDocument(vertices, triangles)` | sampled-geometry histograms |
| AUDIO | `AudioDocument(N,C)` | frame-padded descriptor (fixed frame grid) |
| VIDEO | `VideoDocument(frames=[(3,H,W)])` | frame sampling + pooled stats |
| SENSOR | `SensorDocument((N,C) tensor)` | last-frame + sequence stats |

## 5. Sequence (typical pipeline)

```
MultimodalSystem.encode
   -> encoders[modality].encode(doc)              # raw features
MultimodalSystem.embed
   -> space.embed(modality, features)             # (B,E) shared space
CrossModalEngine.forward
   -> attention over aligned pairs                 # e.g. CAD<->TEXT
MultimodalSystem.fuse -> FusionResult.fused       # (B,E)
MultimodalSystem.project(fused, target)           # (B, E_target)
MultimodalIntegrator.from_document(doc)           # raw CADDocument -> features
```

## 6. Verification

- `tests/multimodal/test_encoders.py`, `test_fusion.py` — 22 tests green.
- `benchmarks/multimodal_benchmarks.py` — encoder 0.33–7.3 ms, fusion 0.39–2.19 ms,
  cross-modal 4.4–7.5 ms, end-to-end ~5.4 ms per call (reps=3, batch=8).
- `evaluation/multimodal_metrics.py` — retrieved_to_ground_truth,
  aligned_embedding_distance, modality_coverage, fusion_consistency,
  run_multimodal_benchmark.
- `src/cadgenesis/config/__init__.py` — `MultimodalConfig` (embed_dim,
  feature_dims, fusion_strategy, cross_modal pairs) exported.
- Ruff clean; mypy clean.
