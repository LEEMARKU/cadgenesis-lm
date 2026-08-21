"""cadgenesis.cad.integration.transformer_bridge
==============================================
Encode CAD designs into the batched tensor form the transformer consumes.

The bridge turns a ``CADTokenSequence`` (produced by
:class:`~cadgenesis.cad.integration.tokenizer_bridge.TokenizerBridge`) into a
:class:`MultiModalBatch` via the tokenizer's ``collate``.  ``to_torch()`` on
the batch yields the exact dict of tensors the CAD transformer expects.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.cad.integration.tokenizer_bridge import TokenizerBridge


class TransformerBridge:
    """Adapts CAD designs to transformer-ready batches."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.tokens = TokenizerBridge(tokenizer)

    def encode_design(self, design: dict[str, Any], text: str = "") -> Any:
        """Return a ``CADTokenSequence`` for a single design."""
        return self.tokens.to_sequence(design, text=text)

    def encode_batch(
        self,
        designs: list[dict[str, Any]],
        texts: list[str] | None = None,
        max_src: int | None = None,
        max_tgt: int | None = None,
    ) -> Any:
        """Return a padded ``MultiModalBatch`` for many designs."""
        if texts is None:
            texts = [""] * len(designs)
        sequences = [
            self.encode_design(design, text) for design, text in zip(designs, texts, strict=False)
        ]
        return self.tokenizer.collate(sequences, max_src=max_src, max_tgt=max_tgt)

    def batch_to_torch(
        self, designs: list[dict[str, Any]], texts: list[str] | None = None
    ) -> dict[str, Any]:
        """Directly produce the ``{text_ids, cad_ids, type_ids, attention_mask}`` tensor dict."""
        batch = self.encode_batch(designs, texts)
        return batch.to_torch()

    def decode_sequence(self, cad_ids: list[int]) -> list[str]:
        """Decode model output ids back to CAD token strings."""
        return self.tokenizer.decode_cad_sequence(cad_ids)


__all__ = ["TransformerBridge"]
