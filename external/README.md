# CADGenesis-Data (External)

This directory is a **reference** for the external CADGenesis data store.

Large artifacts — datasets, checkpoints, pretrained models, adapters, logs,
experiments, generated CAD models, simulations, and backups — are kept
**outside** the source repository by design (they can grow beyond 100 GB).

Create the data root next to the repository (or anywhere you like) and point
CADGenesis-LM at it via the `CADGENESIS_DATA_ROOT` environment variable
(see `.env.example`).

```
CADGenesis-Data/
├── datasets/          # raw and processed datasets
├── checkpoints/       # model checkpoints
├── pretrained_models/ # downloaded / shared base models
├── adapters/          # trained adapter weights
├── logs/              # training and serving logs
├── cache/             # intermediate caches
├── experiments/       # experiment outputs
├── tensorboard/       # tensorboard event logs
├── generated_models/  # model-generated CAD files
├── cad_outputs/       # exported CAD artifacts
├── simulations/       # FEA / CFD results
└── backups/           # backups of important artifacts
```
