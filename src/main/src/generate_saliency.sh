#!/bin/bash
# Generate saliency experiments
python3 saliency.py --model "../models/baseline_cnn_lab4_frequent_metrics.pth" --arch basefm --out "baseline CNN"
python3 saliency.py --model "../models/mesonet_imp_v3_1_epoch_15.pth" --arch mesi3 --out "MesoNet v3"

