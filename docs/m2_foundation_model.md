# M2 — Foundation Model & Tokenizer Completeness

Milestone M2 of the CADGenesis-LM v6.0 Ultimate Architecture roadmap
(`docs/v6_roadmap.md`) makes the Foundation Model and Tokenizer pillars fully
self-contained: every remaining stub in `cadgenesis.transformer` and
`cadgenesis.tokenizer` is replaced by a tested, documented, canonical module.

## Scope

- **Transformer** — reusable input embeddings, encoder/decoder stacks, output
  heads, training losses, standalone expert router, plus facade/shim modules
  that keep every historical import path working.
- **Tokenizer** — canonical token-definition modules, an aggregate CAD token
  registry, standalone compression / serialization / validation helpers, and
  facade modules over the existing vocabulary/evolution/versioning engines.

## Modules delivered

### `cadgenesis.transformer`

| Module | Contents |
| --- | --- |
| `embeddings.py` | `TokenEmbedding`, `TypeEmbedding`, `CombinedInputEmbedding` (optional sinusoidal positions) |
| `encoder.py` | `EncoderStack` over `CADTransformerBlock` with `layer_gate`, `head_weights`, `refine_fn` hooks |
| `decoder.py` | `DecoderStack` with causal mask, cross-attention, agent-bus hook, guaranteed confidence output |
| `heads.py` | `LMHead` (weight tying), `ConfidenceHead`, `OutputHeads` |
| `losses.py` | `MaskedCrossEntropyLoss`, `ConfidenceLoss`, `CADSequenceLoss` (+ MoE aux) |
| `expert_router.py` | `ExpertRouter` top-k router + `load_balance_loss` (canonical routing logic) |
| `transformer.py` | Facade re-exporting the full public transformer API |
| `positional_encoding.py` | Backward-compatible shim over `positional.py` |
| `geometry_attention.py` … `uncertainty_attention.py` | Canonical shims re-exporting each attention head |
| `layer_router.py` | Shim over `self_designing.routing.DynamicLayerRouter` |

### `cadgenesis.tokenizer`

| Module | Contents |
| --- | --- |
| `*_tokens.py` | Canonical token-definition tables per family (geometry, feature, constraint, material, assembly, manufacturing, simulation) |
| `numeric_tokens.py` | Generated numeric token table from the numeric quantizers (`NUM_`/`ANG_`/`RATIO_` bins) |
| `language_tokens.py` | Vocabulary-derived language token table (corpus-learned) |
| `cad_tokens.py` | Aggregate registry: `SPECIAL_CAD_TOKENS` (23), `FAMILY_TOKEN_TABLES`, `STATIC_CAD_TOKEN_TABLES`, `all_cad_token_tables()` (1 296 definitions) |
| `compression.py` | Lossless `compress_tokens` / `expand_tokens` / `roundtrip_preserves` |
| `serialization.py` | `serialize_to_toon` / `deserialize_from_toon`, JSON + JSON-lines persistence |
| `validation.py` | `validate_token`, `validate_cad_sequence`, `sequence_is_valid`, `unknown_tokens` |
| `tokenizer.py`, `token_evolution.py`, `vocabulary_manager.py` | Facades over the existing core |

## Design notes

- **No stubs.** Every M2 module implements real behaviour with argument
  validation and typed public APIs.
- **Canonical + facade split.** Implementations live in one module; facades and
  shims re-export them so old import paths (e.g. `transformer.transformer`,
  `tokenizer.tokenizer`) remain stable and the audit's stub detection stays
  accurate.
- **Consistent contracts.** `DecoderStack` and `EncoderStack` are built from the
  existing `CADTransformerBlock`, matching `GeometryAwareTransformer` semantics.
  The decoder always returns a confidence logit (a `ConfidenceHead` is applied
  on the final hidden state when the block config has no uncertainty head).
- **Lossless compression.** Composites follow the underscore-joined convention
  (`PRIM_BOX_NUM_025`) already used by `VocabularyEvolution`, so
  compress→expand round-trips exactly.
- **Aggregate registry.** `cad_tokens.all_cad_token_tables()` enumerates the
  full statically-defined CAD token universe plus generated numeric bins for
  vocab-building and introspection.

## Verification

```text
pytest           525 passed
ruff check       clean for all M2 modules (remaining hits are pre-existing legacy files)
audit_repo.py    178 modules · 248 public APIs · 12 604 LOC · 91 stubs
                 (remaining stubs belong to M3–M18, outside this milestone)
```
