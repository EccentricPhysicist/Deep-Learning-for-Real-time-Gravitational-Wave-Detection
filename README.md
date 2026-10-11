# Deep Filtering CNN for Binary Black Hole Detection (H1 + L1)

A PyTorch implementation of the **Deep Filtering** classifier from George & Huerta, *"Deep Learning for real-time gravitational wave detection and parameter estimation: Results with Advanced LIGO data"* ([arXiv:1711.03121](https://arxiv.org/abs/1711.03121)), trained to separate **binary black hole (BBH) signals injected into real LIGO noise** from noise-only data, using two detectors (H1 and L1) as input channels.

The project has two notebooks:

| Notebook | Purpose |
|---|---|
| `BBH_Gravitational_Wave_Dataset_Generation_H1L1.ipynb` | Builds the dataset: whitened H1/L1 strain windows, with and without simulated BBH injections |
| `BBH_GW_Deep_Filtering_CNN_Training.ipynb` | Trains the CNN on that dataset and evaluates it on a held-out test set |

## Results

Test set: 2000 examples (balanced noise / signal), signals injected at an optimal H1 SNR of 4–10. The numbers below come from one training run (seed 42). The restored checkpoint was **epoch 3** (selected by lowest validation loss; see [Known issues](#known-issues)).

| Metric | Value |
|---|---|
| ROC-AUC | **0.968** (95 % bootstrap CI 0.960–0.975) |
| Average precision | 0.976 (0.970–0.981) |
| Accuracy at P(signal) > 0.5 | 84.3 % (82.7–85.9 %) |
| TPR / FPR at P > 0.5 | 0.686 / 0.001 |
| Detection efficiency at ~1 % false-alarm rate | **78.4 %** (75.7–80.8 %) |
| Test FPR obtained with the validation-chosen 1 % threshold | 0.9 % (0.5–1.7 %) |

**Detection efficiency vs. SNR** (fixed 1 % FPR threshold, from the efficiency plot of this run):
efficiency rises steeply with SNR, from roughly 30 % near network SNR 6, to roughly 70 % near 8.5, above 90 % near 11, and approximately 100 % from about 14 upward. It is lowest for the weakest signals, as expected for a real detector, and does not look like a dataset shortcut (a shortcut would give flat efficiency at all SNRs).

> The 0.5-probability accuracy is held down by a conservative threshold (very few false alarms, many missed weak signals) rather than by poor ranking, which is why the AUC is high. For detection, the fixed-false-alarm result is the more meaningful number.

## Model

Dilated 1-D CNN, exactly the architecture of the paper's Fig. 3, with a 2-channel input:

| Block | Layers | Output |
|---|---|---|
| 1 | Conv1d(2→64, k=16, dil=1) → MaxPool(4) → ReLU | 64 × 2044 |
| 2 | Conv1d(64→128, k=16, dil=2) → MaxPool(4) → ReLU | 128 × 503 |
| 3 | Conv1d(128→256, k=16, dil=2) → MaxPool(4) → ReLU | 256 × 118 |
| 4 | Conv1d(256→512, k=32, dil=2) → MaxPool(4) → ReLU | 512 × 14 |
| Head | Flatten (7168) → Linear 128 → ReLU → Linear 64 → ReLU → Linear 2 | 2 logits |

About 5.78 M trainable parameters. The input is 2 s at 4096 Hz = 8192 samples, the same length as the paper's 1 s at 8192 Hz input, so every layer size matches the paper. The only change is the first convolution taking 2 channels instead of 1. The training notebook asserts all 18 layer output sizes against the paper.

### Deviations from the paper

- **Curriculum learning** is approximated by data ordering (highest-SNR injections first), because the dataset's SNR is fixed.
- **No transfer learning** from a mass-regression network, and **no noise-mixing augmentation**. A global sign flip (a physically valid change of polarisation angle) is used instead.
- Optimiser is Adam as in the paper; learning rate (1e-4), batch size (64) and the rest are our own choices.

## Data layout

The training notebook expects the output of the dataset notebook:

```
gw_bbh_dataset/
├── train/
│   ├── X.npy           # (N, 2, 8192) float32, whitened H1/L1 strain
│   ├── y.npy           # (N,) 0 = noise, 1 = signal
│   └── metadata.csv    # gps_time, mass1, mass2, snr_H1, snr_L1, snr_network, ...
└── test/
    └── (same files)
```

The training notebook validates shapes, labels, finiteness, and that `metadata.csv` rows line up with `X`/`y`, and stops with a clear error if anything is inconsistent.

## Quick start

### Locally

```bash
pip install torch numpy pandas matplotlib scikit-learn scipy jupyter
```

1. Run the dataset notebook to create `gw_bbh_dataset/`.
2. Open `BBH_GW_Deep_Filtering_CNN_Training.ipynb` in the same directory and run all cells.

A GPU is strongly recommended. On a laptop CPU one epoch took on the order of ten minutes; on a Colab T4 it took 15–25 s.

### On Google Colab (recommended)

1. Zip the dataset so that `gw_bbh_dataset/` is the top-level folder, upload it to Google Drive, and set sharing to **Anyone with the link**.
2. Upload the training notebook to Colab and choose **Runtime → Change runtime type → T4 GPU**.
3. Add this as the first cell:

   ```python
   import os, subprocess
   import gdown

   DRIVE_LINK = "PASTE_YOUR_PUBLIC_LINK_HERE"

   if not os.path.isdir("gw_bbh_dataset"):
       gdown.download(DRIVE_LINK, "gw_bbh_dataset.zip", quiet=False, fuzzy=True)
       subprocess.run(["unzip", "-q", "-o", "gw_bbh_dataset.zip"], check=True)
       os.remove("gw_bbh_dataset.zip")
   ```
4. Run everything. Colab storage is temporary, so at the end download the results:

   ```python
   import shutil
   from google.colab import files
   shutil.make_archive("gw_cnn_outputs", "zip", "gw_cnn_outputs")
   files.download("gw_cnn_outputs.zip")
   ```

## Training and evaluation protocol

- **Validation split:** held out from the training set in contiguous 64 s time blocks, with an asserted guard so no validation window overlaps a training window. A random split would leak noise, because the dataset draws overlapping crops from the same noise stretches.
- **SNR curriculum:** for the first `CURRICULUM_EPOCHS` (10) epochs the loudest injections plus an equal number of noise examples are used, expanding to the full set.
- **Early stopping and LR schedule** monitor validation loss; checkpoints are restored at the end.
- **Test set** is used once, after training, for no decisions.
- **Detection statistic:** the logit margin `z_signal − z_noise` (`> 0` is equivalent to P(signal) > 0.5, but it does not saturate, so low-FPR thresholds are well defined).
- **Fixed-false-alarm operating point:** the threshold is the 99th percentile of **validation** noise scores; the false-alarm rate it actually gives on test noise is reported with a Wilson interval.

### Outputs (`gw_cnn_outputs/`)

| File | Content |
|---|---|
| `deep_filtering_cnn_best.pt` | Best model weights (`state_dict`) |
| `training_history.csv`, `config.json` | Per-epoch metrics and run configuration |
| `test_metrics.json`, `test_predictions.csv` | All test metrics and per-example scores |
| `01_training_curves.png` | Loss, accuracy, validation AUC, learning rate |
| `02_confusion_matrices.png` | At P > 0.5 and at the fixed-FPR threshold |
| `03_roc_pr.png` | ROC (linear and log-FPR) and precision-recall |
| `04_scores_calibration.png` | Score histograms and reliability diagram |
| `05_efficiency_vs_snr.png` | Detection efficiency vs. H1, L1 and network SNR |
| `06_efficiency_vs_mass.png` | Efficiency vs. total mass, mass ratio, and the (m1, m2) plane |
| `07_misclassified_examples.png` | Highest-scoring false alarms and loudest missed signals |

## Known issues

- **Checkpoint selection can pick an under-trained model.** The notebook restores the epoch with the lowest validation *loss*. In the reported run the loss was lowest at epoch 3 (inside the curriculum) and rose afterwards while validation accuracy kept improving to about 95 %, so the restored model is not the most accurate one that was trained. Selecting by validation ROC-AUC after the curriculum ends is the likely fix; the reported results were produced with the loss-based rule.
- **Weights are only written after training ends.** If the kernel dies or training is interrupted, the best weights exist only in memory.
- **Mild overfitting appears with longer training** (about 99 % train vs. 95 % validation accuracy by epoch 17 in the reported run), expected for a 5.8 M-parameter network and roughly 6.8 k training examples. A larger training set is the most effective remedy.

## Limitations

- The merger is always at the centre of the window in both train and test data, so robustness to time shifts is untested. The paper randomly shifted the peak. For a sliding-window search on continuous data, regenerate injections with shifted merger times.
- The test noise comes from the same noise files as training (a later time split), so generalisation to other epochs or detector conditions is untested.
- Noise windows are random crops with no data-quality vetoes, so real glitches (and possibly real events) can appear in either class.
- A 1 % false-alarm probability is far from the rates a real search needs. This is a proof-of-concept classifier, not a search pipeline.
- Results come from a single seed; compare changes across several seeds.

## Reference

George, D. & Huerta, E. A. *Deep Learning for real-time gravitational wave detection and parameter estimation: Results with Advanced LIGO data.* Physics Letters B 778 (2018) 64–70. [arXiv:1711.03121](https://arxiv.org/abs/1711.03121)

