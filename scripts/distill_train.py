"""
distill_train.py
================
CLI Runner for CADGenesis-LM v2.0 Teacher-Student Distillation & Quality Pipeline.

Demonstrates:
1. Teacher Model Interface (GPT-4o, DeepSeek, Qwen)
2. Automated Dataset Generation Pipeline with TOON
3. Quality Filtering & Geometry Topology Validation
4. Multi-Teacher Soft KL Divergence Distillation Loss
5. Self-Improvement Iterative Feedback Loop
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from cadgenesis.config import CADConfig
from cadgenesis.distillation.distill_pipeline import (
    AutomatedDatasetGenPipeline,
    DistillationLossPipeline,
    QualityFilteringEngine,
    SelfImprovementLoop,
    TeacherModelInterface,
)
from cadgenesis.tokenizer.cad_tokenizer import AutonomousCADTokenizer
from cadgenesis.tokenizer.legacy_shim import build_dataset
from cadgenesis.training.trainer import CADTrainer, MultiModalCADDataset, cad_collate_fn
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM->LLM Teacher-Student Distillation Pipeline."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of teacher synthetic CAD samples to generate.",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Training epochs for student model distillation."
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training.")
    parser.add_argument(
        "--temperature", type=float, default=2.0, help="Distillation soft label temperature."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight balance between hard loss and soft KL loss.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=================================================================")
    print("CADGenesis-LM v2.0 — Teacher-Student Distillation Pipeline")
    print("=================================================================")

    # Step 1: Initialize Teacher Model Interface & Quality Filtering
    print("\n[Phase 1] Initializing Teacher Interface & Quality Filter...")
    teacher = TeacherModelInterface(provider="openai")
    quality_filter = QualityFilteringEngine()

    # Step 2: Automated Synthetic Data Generation Loop
    print(f"\n[Phase 2] Running Automated Data Generation Pipeline ({args.num_samples} samples)...")
    dataset_gen = AutomatedDatasetGenPipeline(
        teacher_interface=teacher, quality_filter=quality_filter
    )
    verified_dataset = dataset_gen.generate_dataset(num_samples=args.num_samples)

    # Step 3: Initialize Student Model & Distillation Loss
    print("\n[Phase 3] Instantiating Student CADGenesis-LM Model & Distillation Engine...")
    config = CADConfig.mini()
    config.training.max_epochs = args.epochs
    config.training.batch_size = args.batch_size

    tokenizer = AutonomousCADTokenizer.build_mini()
    student_model = GeometryAwareTransformer(config)

    distill_pipeline = DistillationLossPipeline(temperature=args.temperature, alpha=args.alpha)
    print(
        f"Distillation Engine initialized with Temperature={args.temperature}, Alpha={args.alpha}"
    )

    # Prepare PyTorch Dataloader
    raw_data = build_dataset(len(verified_dataset), lang_tok=tokenizer.lang_tok)
    train_ds = MultiModalCADDataset(raw_data, tokenizer)
    train_dl = DataLoader(
        train_ds, batch_size=config.training.batch_size, shuffle=True, collate_fn=cad_collate_fn
    )

    trainer = CADTrainer(config=config, model=student_model, tokenizer=tokenizer)

    # Step 4: Run Student Training with Soft-Target Distillation Loss
    print("\n[Phase 4] Training Student Model with Distillation Loss...")
    for epoch in range(1, args.epochs + 1):
        trainer.model.train()
        total_loss = 0.0
        for batch in train_dl:
            src, tgt = batch
            src, tgt_in, tgt_out, tgt_type, src_mask, tgt_mask = trainer._prepare_batch(src, tgt)

            # Student forward pass
            student_logits, _ = trainer.model(
                src_ids=src,
                tgt_in_ids=tgt_in,
                tgt_type_ids=tgt_type,
                src_key_padding_mask=src_mask,
                tgt_key_padding_mask=tgt_mask,
            )

            # Simulated teacher soft probabilities
            teacher_logits = student_logits.detach() + (torch.randn_like(student_logits) * 0.1)

            # Compute combined Hard Loss + Soft KL Loss
            loss = distill_pipeline.compute_loss(student_logits, teacher_logits, tgt_out)

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        print(f"Epoch {epoch}/{args.epochs} | Distillation Loss (Hard + Soft KL): {avg_loss:.4f}")

    # Step 5: Self-Improvement Loop & Failure Feedback
    print("\n[Phase 5] Executing Self-Improvement & Failure Feedback Loop...")
    self_improvement = SelfImprovementLoop(
        teacher=teacher, quality_filter=quality_filter, student_model=student_model
    )
    test_prompts = [
        "Design an enclosure with 0.5mm wall thickness.",
        "Create a mounting plate with non-manifold overlapping faces.",
        "Generate a cylindrical pin with safety factor >= 2.0.",
    ]
    hard_examples, pass_rate = self_improvement.run_iteration(test_prompts)

    print("\n=================================================================")
    print("Teacher-Student LLM Distillation Pipeline Execution Complete!")
    print(f"Final Student Pass Rate: {pass_rate:.1%}")
    print(
        f"Generated {len(hard_examples)} hard failure correction prompts for teacher re-querying."
    )
    print("=================================================================")


if __name__ == "__main__":
    main()
