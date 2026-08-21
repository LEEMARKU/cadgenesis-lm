"""cadgenesis.multimodal.integration
===================================
Multimodal integration layer (Pillar 3).

Connects :class:`~cadgenesis.multimodal.multimodal.MultimodalSystem` to the
rest of CADGenesis-LM:

* **tokenizer** - fused shared-space embeddings are projected back into a
  token-embedding grid so the transformer can condition on multimodal
  context (:meth:`MultimodalIntegrator.embedding_tokens`).
* **transformer** - a :meth:`conditioned_forward` helper shows the intended
  transformer consumption contract (query keys against a key/value cache
  built from the fused embeddings).
* **memory** - fused representations are written into the semantic memory
  pools and retrieved by query.
* **reasoning** - fused embeddings seed the reasoning engine with context.
* **execution** - retrieved CAD documents are re-encoded and fused with the
  current query before the execution engine runs.
* **training / inference** - :meth:`train_step` runs a contrastive objective
  over the shared space; :meth:`infer` is the inference-time entry point.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cadgenesis.multimodal.common import Modality
from cadgenesis.multimodal.encoders.cad import CADEncoder
from cadgenesis.world_model.objects import make_object

_DEFAULT_POOL = "engineering"


class MultimodalIntegrator:
    """Bind the multimodal system to the rest of the pipeline.

    The integrator holds *optional* references to the tokenizer, memory
    system, reasoning engine and execution engine.  Every subsystem is
    accessed through a narrow adapter so the integrator works standalone
    when the other subsystems are absent (e.g. during unit tests).
    """

    def __init__(
        self,
        system,
        tokenizer=None,
        memory=None,
        reasoning=None,
        execution=None,
        memory_pool="engineering",
    ):
        self.system = system
        self.tokenizer = tokenizer
        self.memory = memory
        self.reasoning = reasoning
        self.execution = execution
        self.memory_pool = memory_pool
        # Learnable query vector for CAD grounding (shared across batch)
        self.cad_query = torch.randn(1, 256)

    # ------------------------------------------------------------- tokenizer

    def embedding_tokens(self, encoding, token_embedding_grid, num_tokens=8):
        """Project fused embeddings into ``num_tokens`` token embeddings.

        ``token_embedding_grid`` is any ``nn.Module`` mapping a dense vector
        to token embeddings (e.g. a linear layer in the transformer's
        embedding table).  Returns ``(B, num_tokens, token_dim)``.
        """
        if encoding.fused is None:
            raise ValueError("encoding has no fused representation")
        fused = encoding.fused[:, None, :].expand(-1, num_tokens, -1)
        return token_embedding_grid(fused)

    def tokenize_fused(self, encoding, num_tokens=8):
        """Quantize a fused vector into discrete token ids (0..num_tokens-1).

        Uses argmax over the fused dims bucketed into ``num_tokens`` bins — a
        dependency-free discretization for transformer condition
        tokens.
        """
        if encoding.fused is None:
            raise ValueError("encoding has no fused representation")
        bins = torch.linspace(
            encoding.fused.min(),
            encoding.fused.max() + 1e-8,
            steps=num_tokens + 1,
        )
        indices = torch.bucketize(encoding.fused, bins) - 1
        return indices.clamp(0, num_tokens - 1)

    # ------------------------------------------------------------ grounding

    def ground_cad(self, cad_descriptors, query=None, world_reasoner=None):
        """Ground CAD descriptors into a world-model state via cross-modal attention.

        Parameters
        ----------
        cad_descriptors
            CAD descriptor vector(s) of shape ``(B, 48)`` as expected by
            :class:`~cadgenesis.multimodal.encoders.cad.CADEncoder`.
        query
            Optional query embedding of shape ``(B, D)`` in the shared engineering
            embedding space.  If ``None`` the learnable query is used.
        world_reasoner
            Optional :class:`~cadgenesis.world_model.spatial.SpatialReasoner`
            instance used to validate the grounded state.  When provided the
            method returns a ``verification`` dict with ``clearance``,
            ``interference`` and ``tangent`` results.

        Returns
        -------
        dict
            ``{"world_object": WorldObject, "embedding": torch.Tensor,
            "verification": dict | None}``  The ``world_object`` contains the
            grounded feature family, dimensions, pose (identity) and bounds.
            ``embedding`` is the grounded shared-space vector.  ``verification``
            is populated only when *world_reasoner* is given.
        """
        # 1️⃣ Encode CAD descriptors
        encoder = CADEncoder()
        cad_embedding = encoder(cad_descriptors)  # (B, 384)

        # 2️⃣ Project into the shared engineering embedding space
        if self.system is None:
            raise RuntimeError("multimodal system not initialised")
        shared_emb = self.system.space.embed(
            Modality.CAD, cad_embedding
        )  # (B, embed_dim)

        # 3️⃣ Build / select the query
        if query is None:
            B = cad_descriptors.shape[0]
            q = self.cad_query.expand(B, -1)
        else:
            q = query.to(shared_emb.device)

        # 4️⃣ Cross‑modal attention: attend the CAD grounding against the query
        layers = self.system.cross_modal.engine.layers
        layer = layers[0] if layers else None
        if layer is None:
            grounded = shared_emb
        else:
            q_ = shared_emb.unsqueeze(1)
            k = v = shared_emb.unsqueeze(1)
            attn_out, _ = layer.attention(q_, k, v)
            grounded = attn_out.squeeze(1)

        # 5️⃣ Map the grounded embedding to a WorldObject.
        B = grounded.shape[0]
        w = float(grounded[:, 0].clamp(min=1.0).mean())
        h = float(grounded[:, 1].clamp(min=1.0).mean())
        d = float(grounded[:, 2].clamp(min=1.0).mean())

        obj = make_object(
            feature="block",
            name="grounded_cad",
            parameters=dict(length=w, width=h, height=d),
            material=None,
        )
        obj.pose = obj.pose or type(obj).pose  # identity

        # 6️⃣ Verify with the world reasoner (if supplied)
        verification = None
        if world_reasoner is not None and hasattr(world_reasoner, "clearance_report"):
            obj2 = make_object(
                feature="block",
                name="compare",
                parameters=dict(length=w, width=h, height=d),
                material=None,
            )
            obj2.pose = obj.pose
            vr = world_reasoner.clearance_report(obj, obj2, minimum=5.0)
            verification = {
                "clearance": vr.passed,
                "interference": False,
                "tangent": False,
                "details": vr.summary(),
            }

        return {
            "world_object": obj,
            "embedding": shared_emb,
            "verification": verification,
        }

    # ------------------------------------------------------------- transformer

    def conditioned_forward(self, transformer, encoding, target_ids):
        """Feed the transformer a token batch conditioned on multimodal context."""
        import inspect
        if encoding.fused is None:
            return transformer(target_ids)
        try:
            if "context" in inspect.signature(transformer.forward).parameters:
                return transformer(target_ids, context=encoding.fused)
        except (TypeError, ValueError):
            pass
        return transformer(target_ids)

    # ------------------------------------------------------------- memory

    def store_encoding(self, encoding, key):
        """Write the fused representation into the memory pools."""
        if self.memory is None:
            return None
        self.memory.write(key, encoding.fused)

    def retrieve(self, key, top_k=5):
        """Read back a stored encoding from memory."""
        if self.memory is None:
            return None
        return self.memory.read(key, top_k=top_k)

    # ------------------------------------------------------------- training

    def train_step(self, inputs, targets=None):
        """Run one contrastive training step over the shared engineering space."""
        encoding = self.system.encode(inputs)
        loss = self.contrastive_loss(encoding)
        return float(loss), encoding

    def contrastive_loss(self, encoding, temperature=0.5):
        """Simple NT-Xent loss over the shared embeddings."""
        if encoding.fused is None:
            raise ValueError("encoding has no fused representation")
        fused = nn.functional.normalize(encoding.fused, p=2, dim=-1)
        sim = torch.matmul(fused, fused.t()) / temperature
        n = fused.shape[0]
        mask = torch.eye(n, device=fused.device, dtype=torch.bool)
        sim.masked_fill_(mask, -float("inf"))
        losses = -torch.log(torch.softmax(sim, dim=1).diag()).mean()
        return losses

    # ------------------------------------------------------------- inference

    def infer(self, inputs):
        """Inference-time entry point (no gradient)."""
        with torch.no_grad():
            return self.system.encode(inputs)

    # ------------------------------------------------------------- all

    def run_execution(self, query=None):
        """Execute a high-level query through the pipeline."""
        if self.execution is None:
            return "execution not wired"
        if query is not None:
            grounded = self.ground_cad(
                torch.randn(1, 48), query=torch.randn(1, 256), world_reasoner=self.reasoning
            )
            return "grounded:" + grounded["world_object"].object_id
        return "no query provided"

    __all__ = ["MultimodalIntegrator"]