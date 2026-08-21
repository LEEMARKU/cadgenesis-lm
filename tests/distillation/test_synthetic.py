"""tests/distillation/test_synthetic.py"""

from __future__ import annotations

from cadgenesis.distillation.synthetic import DEFAULT_PROMPT_TEMPLATES, SyntheticDataGenerator
from sdk import toon_extended


def test_generate_is_deterministic_given_seed():
    a = SyntheticDataGenerator().generate(5, seed=42)
    b = SyntheticDataGenerator().generate(5, seed=42)
    assert a == b


def test_generate_differs_across_seeds():
    a = SyntheticDataGenerator().generate(3, seed=1)
    b = SyntheticDataGenerator().generate(3, seed=2)
    assert a != b


def test_generate_entries_are_parseable_toon():
    dataset = SyntheticDataGenerator().generate(10, seed=7)
    assert len(dataset) == 10
    for entry in dataset:
        assert set(entry) == {"prompt", "toon", "objects"}
        assert entry["prompt"] in DEFAULT_PROMPT_TEMPLATES
        objects = toon_extended.from_toon(entry["toon"])
        assert len(objects) == len(entry["objects"])
        assert all(obj["feature"] for obj in objects)


def test_generate_positive_parameters():
    dataset = SyntheticDataGenerator().generate(20, seed=3)
    for entry in dataset:
        for obj in entry["objects"]:
            for key in ("width", "height", "depth", "fillet"):
                assert float(obj[key]) > 0


def test_generate_zero_and_negative_counts():
    assert SyntheticDataGenerator().generate(0, seed=0) == []
    try:
        SyntheticDataGenerator().generate(-1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_apply_perturbation_is_deterministic_and_changes_values():
    gen = SyntheticDataGenerator()
    dataset = gen.generate(1, seed=5)
    original = dataset[0]["toon"]
    perturbed_a = gen.apply_perturbation(original, noise_scale=0.5, seed=11)
    perturbed_b = gen.apply_perturbation(original, noise_scale=0.5, seed=11)
    assert perturbed_a == perturbed_b
    assert perturbed_a != original
    assert toon_extended.from_toon(perturbed_a)


def test_apply_perturbation_preserves_structure():
    gen = SyntheticDataGenerator()
    dataset = gen.generate(1, seed=5)
    original = dataset[0]["toon"]
    perturbed = gen.apply_perturbation(original, noise_scale=0.2, seed=9)
    original_objs = toon_extended.from_toon(original)
    perturbed_objs = toon_extended.from_toon(perturbed)
    assert len(perturbed_objs) == len(original_objs)
    for before, after in zip(original_objs, perturbed_objs, strict=True):
        assert after["feature"] == before["feature"]
        for key in ("width", "height", "depth", "fillet"):
            assert float(after[key]) != float(before[key])


def test_apply_perturbation_zero_noise_is_identity():
    dataset = SyntheticDataGenerator().generate(1, seed=5)
    perturbed = SyntheticDataGenerator().apply_perturbation(
        dataset[0]["toon"], noise_scale=0.0, seed=1
    )
    assert perturbed == dataset[0]["toon"]


def test_apply_perturbation_returns_input_when_no_numeric_columns():
    plain = "id|feature\n1|BOX"
    assert SyntheticDataGenerator().apply_perturbation(plain, noise_scale=0.5, seed=1) == plain


def test_rejects_invalid_jitter_and_noise():
    try:
        SyntheticDataGenerator(param_jitter=2.0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        SyntheticDataGenerator().apply_perturbation("id|width\n1|5.0", noise_scale=-1.0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
