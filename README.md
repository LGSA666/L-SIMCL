# L-SIMCL: label-guided semantic interaction with multi-dimensional contrastive learning for hierarchical text classification

## Requirements

For detailed configuration, please refer to requirements.txt

## Data preparation

Please obtain the original dataset and then run the script provided by HPT (https://github.com/wzh9969/HPT). Our data processing method is completely consistent with HPT.

## Train

```
usage: train.py [-h] [--lr LR] [--data DATA] [--batch BATCH]
                [--early-stop EARLY_STOP] [--device DEVICE] --name NAME
                [--update UPDATE] [--swanlab] [--arch ARCH] [--layer LAYER]
                [--graph GRAPH] [--low-res] [--seed SEED]
                [--lambda_1 LAMBDA_1] [--lambda_2 LAMBDA_2] [--temp TEMP]
                [--heads HEADS] [--dropout_cl_rate DROPOUT_CL_RATE]
                [--loss {zlpr,bce,hbm}] [--hbm_alpha HBM_ALPHA]
                [--hbm_margin HBM_MARGIN] [--hbm_loss_mode {unit,hierarchy}]
                [--threshold THRESHOLD] [--epoch EPOCH]
optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Learning rate (default: 3e-5)
  --data DATA           Dataset name (default: WebOfScience)
  --batch BATCH         Batch size for training (default: 24)
  --early-stop EARLY_STOP
                        Early stopping patience (default: 10)
  --device DEVICE       Device to use (default: cuda)
  --name NAME           Name for a specific run. Required.
  --update UPDATE       Gradient accumulation steps (default: 1)
  --swanlab             Whether to use SwanLab logging (default: False)
  --arch ARCH           Pretrained model architecture (default: bert-base-uncased)
  --layer LAYER         Number of GNN layers (default: 1)
  --graph GRAPH         Graph type (GAT, GCN, etc.) (default: GAT)
  --low-res             Use low-resource setting (default: False)
  --seed SEED           Random seed (default: 3)
  --lambda_1 LAMBDA_1   Sample-level CL weight (default: 0.1)
  --lambda_2 LAMBDA_2   Label-level CL weight (default: 0.1)
  --temp TEMP           Contrastive Learning Temperature (default: 0.1)
  --heads HEADS         Number of Label-Aware Attention Heads (default: 4)
  --dropout_cl_rate DROPOUT_CL_RATE
                        Dropout rate (Sample-level CL) (default: 0.1)
  --loss {zlpr,bce,hbm}
                        Classification Loss Function: zlpr (default), bce, or hbm
  --hbm_alpha HBM_ALPHA
                        HBM loss alpha (default: 1.0)
  --hbm_margin HBM_MARGIN
                        HBM loss margin (default: 0.1)
  --hbm_loss_mode {unit,hierarchy}
                        HBM Loss Mode (default: unit)
  --threshold THRESHOLD
                        Classification Threshold during Inference (default: 0.5)
  --epoch EPOCH         Max Training Epochs (default: 100)
```
