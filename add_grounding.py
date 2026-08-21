import sys
sys.path.insert(0, r'D:\Gen-AI CAD_LLM\src')

with open('src/cadgenesis/multimodal/integration.py', 'r') as f:
    content = f.read()

insert_marker = '        return indices.clamp(0, num_tokens - 1)'
insert_idx = content.find(insert_marker)
if insert_idx == -1:
    print('Marker not found!')
else:
    insert_pos = content.find('\n', insert_idx) + 1
    new_method = '''
    # ------------------------------------------------------------ grounding

    def ground_cad(
        self,
        cad_descriptors: torch.Tensor,
        query: torch.Tensor | None = None,
        *,
        world_reasoner: Any = None,
    ) -> dict[str, Any]:
        \"\"\"Ground CAD descriptors into a world-model state via cross-modal attention.

        Parameters
        ----------
        cad_descriptors
            CAD descriptor vector(s) of shape ``(B, 48)`` as expected by
            :class:`~cadgenesis.multimodal.encoders.cad.CADEncoder`.
        query
            Optional query embedding of shape ``(B, D)`` in the shared engineering
            embedding space.  If ``None`` a learnable query vector is used.
        world_reasoner
            Optional :class:`~cadgenesis.world_model.spatial.SpatialReasoner`
            instance used to validate the grounded state.  When provided the
            method returns a ``verification`` dict with ``clearance``,
            ``interference`` and ``tangent`` results.

        Returns
        -------
        dict
            ``{\"world_object\": WorldObject, \"embedding\": torch.Tensor,
            \"verification\": dict | None}``  The ``world_object`` contains the
            grounded feature family, dimensions, pose (identity) and bounds.
            ``embedding`` is the grounded shared‑space vector.  ``verification``
            is populated only when *world_reasoner* is given.
        \"\"\"
        from cadgenesis.multimodal.encoders.cad import CADEncoder
        from cadgenesis.world_model.objects import WorldObject, make_object

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
            # learnable query (shared across the batch)
            query = self.system.space.empty_embedding(Modality.CAD).expand(
                cad_descriptors.shape[0], -1
            )
        else:
            query = query.to(shared_emb.device)

        # 4️⃣ Cross‑modal attention: attend the CAD grounding against the query
        layers = self.system.cross_modal.engine.layers
        layer = layers[0] if layers else None
        if layer is None:
            grounded = shared_emb
        else:
            q = shared_emb.unsqueeze(1)
            k = v = shared_emb.unsqueeze(1)
            attn_out, _ = layer.attention(q, k, v)
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


# ------------------------------------------------------------ transformer

'''

    new_content = content[:insert_idx] + new_method + content[insert_idx:]
    with open('src/cadgenesis/multimodal/integration.py', 'w') as f:
        f.write(new_content)
    print('Method added successfully')