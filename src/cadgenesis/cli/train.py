"""
cadgenesis.train
================
Training Script for CADGenesis-LM v2.0 Foundation Model.

Executes end-to-end training of GeometryAwareTransformer on multi-modal CAD token streams.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader

from cadgenesis.config import CADConfig
from cadgenesis.tokenizer.cad_tokenizer import AutonomousCADTokenizer
from cadgenesis.tokenizer.legacy_shim import LangTokenizer, build_dataset
from cadgenesis.training.trainer import CADTrainer, MultiModalCADDataset, cad_collate_fn
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CADGenesis-LM v2.0 on synthetic multi-modal CAD data."
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override the number of training epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override the training batch size."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints and training output.",
    )
    parser.add_argument(
        "--train-size", type=int, default=800, help="Number of synthetic training examples."
    )
    parser.add_argument(
        "--valid-size", type=int, default=200, help="Number of synthetic validation examples."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for dataset generation and training."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use for training, e.g. cpu or cuda."
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["mini", "full"],
        default="mini",
        help=(
            "Training model size / architecture mode. 'full' uses the"
            " complete CADGenesis v2 architecture."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=================================================================")
    print("CADGenesis-LM v2.0 — Training Foundation Model Core")
    print("=================================================================")

    config = CADConfig.mini() if args.model_size == "mini" else CADConfig()
    if args.epochs is not None:
        config.training.max_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    if args.model_size == "mini":
        tokenizer = AutonomousCADTokenizer.build_mini()
    else:
        tokenizer = AutonomousCADTokenizer.build()

    raw_train = build_dataset(args.train_size, lang_tok=cast(LangTokenizer, tokenizer.lang_tok))
    raw_val = build_dataset(args.valid_size, lang_tok=cast(LangTokenizer, tokenizer.lang_tok))

    if args.model_size == "full":
        config.tokenizer.lang_vocab_size = tokenizer.lang_vocab_size

    print(
        f"[Config] d_model={config.model.d_model}, nhead={config.model.nhead}, "
        f"encoder_layers={config.model.num_encoder_layers}, "
        f"decoder_layers={config.model.num_decoder_layers}, "
        f"epochs={config.training.max_epochs}, batch_size={config.training.batch_size}, "
        f"mode={args.model_size}"
    )
    print(
        f"[Tokenizer] Autonomous CAD Tokenizer initialized."
        f" Total Vocab: {tokenizer.vocab_size:,} tokens"
    )

    train_ds = MultiModalCADDataset(raw_train, tokenizer)
    val_ds = MultiModalCADDataset(raw_val, tokenizer)

    train_dl = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=cad_collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=cad_collate_fn,
    )

    model = GeometryAwareTransformer(config)
    trainer = CADTrainer(config=config, model=model, tokenizer=tokenizer, device=args.device)
    trainer.configure_scheduler(len(train_dl))

    if args.model_size == "full" and trainer.device == "cpu":
        print(
            "[Warning] Full CADGenesis-LM mode is extremely large on CPU. "
            "Training may be slow or may require a GPU with sufficient memory."
        )

    resumed_checkpoint = None
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume_from is not None:
        resumed_checkpoint = trainer.load_checkpoint(args.resume_from)
        start_epoch = resumed_checkpoint.get("epoch", 0) + 1
        best_val_loss = resumed_checkpoint.get("validation_loss", best_val_loss)
        print(
            f"[Resume] Loaded checkpoint from: {args.resume_from} (starting at epoch {start_epoch})"
        )
        if start_epoch > config.training.max_epochs:
            print(
                f"[Resume] Checkpoint already contains epoch {resumed_checkpoint.get('epoch')} "
                f"which is >= max_epochs={config.training.max_epochs}. Nothing to do."
            )
            return

    print(
        f"[Model] GeometryAwareTransformer instantiated"
        f" ({sum(p.numel() for p in model.parameters()):,} parameters)"
    )
    print(f"[Device] Training on device: {trainer.device}")
    print(f"[Output] Saving checkpoints to: {config.output_dir}")

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, config.training.max_epochs + 1):
        train_loss = trainer.train_epoch(train_dl)
        val_loss = trainer.validate(val_dl)
        checkpoint_path = os.path.join(
            config.output_dir,
            f"checkpoint_epoch_{epoch}.pt",
        )
        trainer.save_checkpoint(
            checkpoint_path,
            epoch=epoch,
            step=epoch * len(train_dl),
            validation_loss=val_loss,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config.output_dir, "best_checkpoint.pt")
            trainer.save_checkpoint(
                best_path,
                epoch=epoch,
                step=epoch * len(train_dl),
                validation_loss=val_loss,
            )
            print(
                f"[Best] New best validation loss: {best_val_loss:.4f}."
                f" Saved best checkpoint: {best_path}"
            )
        print(
            f"Epoch {epoch:2d}/{config.training.max_epochs} "
            f"| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
            f"| checkpoint saved: {checkpoint_path}"
        )

    print("=================================================================")
    print("Training run complete. Checkpoints and logs are saved.")
    print("=================================================================")


if __name__ == "__main__":
    main()
