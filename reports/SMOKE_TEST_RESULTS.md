# SMOKE TEST RESULTS (Phase 12, CPU)

| stage | status | duration_s | key metric |
| --- | --- | --- | --- |
| stage1_forward_backward | PASS | 1.0 | 6.260519504547119 |
| stage2_tiny_dataset | PASS | 2.1 | 5.716460466384888 |
| stage3_overfit | PASS | 38.1 | 0.49460455775260925 |
| stage4_dev_run | PASS | 9.8 | 6.065155506134033 |

## stage1_forward_backward

- batch_shape: [4, 64, 32]
- duration_s: 1.0
- gradients_updated: True
- loss: 6.260519504547119
- parameters: 2597660
- status: PASS

## stage2_tiny_dataset

- duration_s: 2.12
- epochs: 1
- final_train_loss: 6.012326649257115
- final_val_loss: 5.716460466384888
- initial_val_loss: 6.215238809585571
- records: 50
- status: PASS

## stage3_overfit

- curve: [6.260775, 4.652886, 3.421381, 2.472662, 1.743518, 1.145314, 0.717821]
- duration_s: 38.07
- final_loss: 0.49460455775260925
- initial_loss: 6.260775089263916
- status: PASS
- steps_used: 137
- target_loss: 0.5
- target_reached: True

## stage4_dev_run

- best_val_loss: 6.065155506134033
- checkpoint_epoch: 1
- checkpoint_path: outputs\smoke\stage4\last.pt
- curve: [{'epoch': 0.0, 'train_loss': 6.222339079930232, 'val_loss': 6.182922840118408}, {'epoch': 1.0, 'train_loss': 6.147637293888972, 'val_loss': 6.065155506134033}]
- duration_s: 9.78
- epochs: 2
- final_train_loss: 6.147637293888972
- final_val_loss: 6.065155506134033
- metrics_path: outputs\smoke\stage4\metrics\metrics.jsonl
- records: 200
- status: PASS

## Verdict

**ALL STAGES PASS** - training pipeline is ready for the PRE-TRAINING READINESS REVIEW.