"""cadgenesis.distillation.synthetic
=================================
Synthetic data generation for distillation.

Rule-based generation of (prompt, TOON) training pairs: the prompt
templates reused from ``cadgenesis.distillation.distill_pipeline`` are
crossed with randomized BOX/CYLINDER/SPHERE/EXTRUDE_PROFILE objects whose
parameters carry a jitter around base ranges.  Serialization goes through
``sdk.toon_extended.to_toon`` so every sample is a well-formed TOON payload
(header + optional schema + rows).

Determinism
-----------
:meth:`SyntheticDataGenerator.generate` and
:meth:`SyntheticDataGenerator.apply_perturbation` draw every random value
from their own ``random.Random(seed)`` instance, so identical inputs with
the same seed produce byte-identical output regardless of the global RNG
state.
"""

from __future__ import annotations

import random
import re
from typing import Any

from sdk import toon_extended

__all__ = ["DEFAULT_PROMPT_TEMPLATES", "SyntheticDataGenerator"]

#: Prompt templates shared with ``AutomatedDatasetGenPipeline``.
DEFAULT_PROMPT_TEMPLATES: tuple[str, ...] = (
    "Design a mounting bracket with 4 M5 screw holes.",
    "Create a hollow cylindrical housing with 2mm wall thickness.",
    "Generate an extruded enclosure box with rounded 2mm fillets.",
    "Design a structural support beam with I-profile cross section.",
    "Create a stepped shaft adapter with keyway slot.",
)

_PRIMITIVES: tuple[str, ...] = ("BOX", "CYLINDER", "SPHERE", "EXTRUDE_PROFILE")
_PERTURB_KEYS: tuple[str, ...] = ("width", "height", "depth", "fillet")
_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


class SyntheticDataGenerator:
    """Rule-based synthetic (prompt, TOON) sample generator."""

    def __init__(self, prompts: list[str] | None = None, param_jitter: float = 0.15) -> None:
        if not 0.0 <= param_jitter <= 1.0:
            raise ValueError(f"param_jitter must be in [0, 1], got {param_jitter}")
        self.prompts = list(prompts) if prompts else list(DEFAULT_PROMPT_TEMPLATES)
        self.param_jitter = param_jitter

    # ------------------------------------------------------------ sampling

    def generate(self, n: int, seed: int | None = None) -> list[dict[str, Any]]:
        """Generate ``n`` deterministic samples ``{prompt, toon, objects}``.

        Every draw comes from ``random.Random(seed)``, so ``generate(n,
        seed)`` is reproducible.  Each sample contains 1-2 objects with
        jittered ``width``/``height``/``depth``/``fillet`` parameters
        serialized via ``toon_extended.to_toon(..., include_schema=True)``.
        """
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        rng = random.Random(seed)
        dataset: list[dict[str, Any]] = []
        for _ in range(n):
            prompt = rng.choice(self.prompts)
            objects = self._random_objects(rng)
            dataset.append(
                {
                    "prompt": prompt,
                    "toon": toon_extended.to_toon(objects, include_schema=True),
                    "objects": objects,
                }
            )
        return dataset

    def _random_objects(self, rng: random.Random) -> list[dict[str, Any]]:
        return [
            {
                "id": obj_id,
                "feature": rng.choice(_PRIMITIVES),
                "width": self._jittered(rng, rng.uniform(20.0, 80.0)),
                "height": self._jittered(rng, rng.uniform(10.0, 60.0)),
                "depth": self._jittered(rng, rng.uniform(5.0, 40.0)),
                "fillet": self._jittered(rng, rng.uniform(0.5, 4.0)),
            }
            for obj_id in range(1, rng.randint(1, 2) + 1)
        ]

    def _jittered(self, rng: random.Random, base: float) -> float:
        factor = 1.0 + self.param_jitter * rng.uniform(-1.0, 1.0)
        return round(max(base * factor, 0.01), 2)

    # --------------------------------------------------------- perturbation

    def apply_perturbation(
        self, toon: str, noise_scale: float = 0.1, seed: int | None = None
    ) -> str:
        """Jitter the numeric values of the width/height/depth/fillet keys.

        The header line is used to locate the columns of those keys; every
        numeric value in those columns of each data row is then multiplied
        by ``(1 + noise_scale * U(-1, 1))`` (rounded to 2 decimals) via
        regex replacement.  The output remains a parseable TOON payload
        with the same structure as the input.  Deterministic for a given
        ``(toon, noise_scale, seed)``.
        """
        if noise_scale < 0.0:
            raise ValueError(f"noise_scale must be >= 0, got {noise_scale}")
        lines = toon.splitlines()
        if len(lines) < 2:
            return toon
        header_fields = re.split(r"[|]", lines[0])
        perturb_indices = [i for i, field in enumerate(header_fields) if field in _PERTURB_KEYS]
        if not perturb_indices:
            return toon

        out_lines = [lines[0]]
        row_start = 1
        schema_types = {
            "int",
            "integer",
            "float",
            "double",
            "number",
            "str",
            "string",
            "bool",
            "boolean",
        }
        if len(lines) > 1 and all(
            part.lower() in schema_types for part in re.split(r"[|]", lines[1])
        ):
            out_lines.append(lines[1])
            row_start = 2

        rng = random.Random(seed)
        for line in lines[row_start:]:
            parts = re.split(r"[|]", line)
            for index in perturb_indices:
                if index >= len(parts):
                    continue
                match = _NUMBER_PATTERN.search(parts[index])
                if match is None:
                    continue
                value = float(match.group())
                jittered = value * (1.0 + noise_scale * rng.uniform(-1.0, 1.0))
                if jittered == value:
                    continue  # zero noise: preserve the original token text
                parts[index] = (
                    parts[index][: match.start()] + f"{jittered:.2f}" + parts[index][match.end() :]
                )
            out_lines.append("|".join(parts))
        return "\n".join(out_lines)
