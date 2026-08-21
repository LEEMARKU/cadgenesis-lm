"""cadgenesis.world_model.integration
====================================
World-model integration (Pillar 4).

:class:`WorldModelIntegration` bridges the world model to the rest of the
system:

* **Pillar 3 multimodal** — converts the object graph into CAD / text /
  sensor inputs and embeds them with the multimodal system.
* **Memory** — persists and retrieves world snapshots.
* **Datasets** — produces :class:`~cadgenesis.datasets.multimodal.MultimodalSample`
  instances for training the fusion stack on world-model states.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.datasets.multimodal import MultimodalSample
from cadgenesis.multimodal.common import Modality
from cadgenesis.multimodal.encoders.cad import CADDocument, CADFileFormat
from cadgenesis.multimodal.encoders.sensor import SensorDocument
from cadgenesis.world_model.objects import ObjectGraph, WorldObject

# Map world-model primitive families onto the CAD feature vocabulary.
_FAMILY_TO_CAD: dict[str, str] = {
    "block": "extrude",
    "cylinder": "revolve",
    "sphere": "revolve",
    "cone": "revolve",
    "torus": "sweep",
    "prism": "extrude",
    "hole": "hole",
    "fillet": "fillet",
    "chamfer": "chamfer",
    "extrusion": "extrude",
    "revolve": "revolve",
    "loft": "loft",
}


class WorldModelIntegration:
    """Bridges the world model to multimodal, memory and datasets."""

    # ------------------------------------------------------------ to cad

    def to_cad_document(
        self,
        graph: ObjectGraph,
        name: str = "world",
    ) -> CADDocument:
        """Project an object graph onto a CAD document representation."""
        doc = CADDocument(format=CADFileFormat.FUSION360, name=name)
        for obj in graph.objects:
            kind = _FAMILY_TO_CAD.get(obj.feature, obj.feature)
            doc.add_feature(kind, **dict(obj.parameters))
            if obj.material is not None:
                doc.materials.append(obj.material.name)
            if obj.parent is not None:
                doc.assemblies.append({"parent": obj.parent, "child": obj.object_id})
        return doc

    # -------------------------------------------------------- multimodal

    def to_multimodal_inputs(
        self,
        graph: ObjectGraph,
    ) -> dict[Modality, Any]:
        """Build encoder-ready inputs for each modality present in the world."""
        cad_doc = self.to_cad_document(graph)
        text_parts = [
            f"{o.feature} {o.name}" + (f" of {o.material.name}" if o.material is not None else "")
            for o in graph.objects
        ]
        text = "; ".join(text_parts) if text_parts else "empty world"
        stats = self._world_stats(graph)
        # shape (N, 1) so each stat is a time step of a single "world" channel
        sensor_doc = SensorDocument(
            data=[[value] for value in stats],
            channels=["world"],
            source="world_model",
        )
        return {
            Modality.TEXT: text,
            Modality.CAD: cad_doc,
            Modality.SENSOR: sensor_doc,
        }

    @staticmethod
    def _world_stats(graph: ObjectGraph) -> list[float]:
        if not graph.objects:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        count = float(len(graph.objects))
        mean_volume = sum(o.volume_estimate() for o in graph.objects) / count
        mean_mass = sum(o.mass() for o in graph.objects) / count
        mean_confidence = sum(o.confidence for o in graph.objects) / count
        dof_total = sum(max(0, 6 - len(o.relations.get("constraints", []))) for o in graph.objects)
        return [count, mean_volume, mean_mass, mean_confidence, float(dof_total)]

    def embed_world(
        self,
        graph: ObjectGraph,
        multimodal: Any,
    ) -> Any:
        """Encode a world snapshot through the multimodal system."""
        if multimodal is None:
            raise ValueError("multimodal system is not connected")
        return multimodal.encode(self.to_multimodal_inputs(graph))

    # ------------------------------------------------------------- sample

    def to_multimodal_sample(
        self,
        graph: ObjectGraph,
        label: Any = None,
    ) -> MultimodalSample:
        """A training sample from a world snapshot."""
        return MultimodalSample(
            inputs=self.to_multimodal_inputs(graph),
            label=label,
        )

    # -------------------------------------------------------------- memory

    def store(
        self,
        graph: ObjectGraph,
        memory: Any,
        key: str | None = None,
    ) -> str | None:
        """Persist the object graph into the memory system."""
        if memory is None:
            return None
        payload = graph.to_dict()
        if key is not None:
            return memory.remember("engineering", key, payload)
        return memory.remember("engineering", "snapshot", payload)

    def retrieve(
        self,
        query: str,
        memory: Any,
        top_k: int = 4,
    ) -> list[Any]:
        """Recall the most relevant world snapshots for a query."""
        if memory is None:
            return []
        result = memory.retrieve(query, top_k=top_k, pool_names=("engineering",))
        hits = getattr(result, "hits", None)
        if hits is None:
            return []
        return list(hits)[:top_k]

    # ---------------------------------------------------------- conditioned

    def conditioned_reason(
        self,
        capability: str,
        query: str,
        world: Any,
        **kwargs: Any,
    ) -> Any:
        """Run a world reasoner, augmenting it with recalled snapshots.

        Recalled snapshots are merged as prior objects so downstream
        reasoning sees historical context before answering.
        """
        for hit in self.retrieve(query, world.memory or None):
            entry = getattr(hit, "entry", None)
            data = getattr(entry, "content", None)
            if not isinstance(data, dict):
                continue
            for obj_data in data.get("objects", []):
                try:
                    prior = WorldObject.from_dict(obj_data)
                except (TypeError, ValueError):
                    continue
                if world.graph.get(prior.object_id) is None:
                    world.graph.add(prior)
        return world.reason(capability, **kwargs)


__all__ = ["WorldModelIntegration"]
