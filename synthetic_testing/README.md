# Synthetic Testing

This directory contains a controlled synthetic experiment for evaluating probabilistic
disaggregation methods. Instead of using real ground-truth labels, a generative model
is used to synthesise per-pixel labels from the image itself, so the true distribution
is known and results can be evaluated exactly.

## Overview

A U-Net (or a simple 1×1 convolution baseline) is trained to predict a per-pixel
Gaussian distribution. The aggregation step compresses these pixel-level predictions
into region-level predictions. Two aggregation strategies are compared:

| Method | Description |
|---|---|
| **analytical** | Analytically sums pixel-level Gaussians over each region (sum of means, sum of variances). |
| **uniform** | Assumes all pixels in a region share the same value; upsamples region averages back to pixel resolution before computing the loss. |

## Dataset

**EuroSAT** multi-spectral Sentinel-2 imagery is used. The ground-truth labels are
*synthetically generated* from the image itself using the following generative model:

```
mu_true  = (R_norm + G_norm) / 2
std_true = softplus(2 * B_norm - 1)
label ~ Normal(mu_true, std_true)
```

Images are normalised with pre-computed dataset-wide RGB mean/std statistics.

### Setup

Set the `EUROSAT_ROOT` environment variable (or pass `--root`) to point to the EuroSAT
`.tif` directory:

```bash
export EUROSAT_ROOT=/path/to/EuroSAT/ds/images/remote_sensing/otherDatasets/sentinel_2/tif
```

## Files

| File | Description |
|---|---|
| `dataset.py` | `Eurosat` dataset wrapper; also contains a `dataset_cifar` class for CIFAR-10 experiments. |
| `train_new.py` | **Main training script.** Lightweight 1×1-conv baseline with `AnalyticalRegionAggregator` and `Uniform_model`. |
| `train_small.py` | Alternative training script using the full U-Net architecture. |
| `unet.py` | U-Net implementation (encoder/decoder with skip connections). |
| `test_new.py` | Evaluation script: loads a checkpoint and reports MSE, log-probability, and coverage at several radii. |
| `train.sh` | Convenience wrapper to train a single method/kernel-size combination. |
| `full_stack.sh` | Runs all kernel-size × method combinations in sequence. |

## Training

### Single run

```bash
# Analytical aggregation, kernel size 8
./train.sh analytical 8

# Uniform aggregation, kernel size 16
./train.sh uniform 16
```

### Full sweep (all kernel sizes)

```bash
bash full_stack.sh
```

### Manual invocation

```bash
python train_new.py \
    --method analytical \
    --kernel_size 8 \
    --seed 80 \
    --max_epochs 150 \
    --batch_size 128 \
    --gpus 1
```

Checkpoints are saved under `<seed>/logtest/<method>/<kernel_size>/`.

## Evaluation

```bash
python test_new.py \
    --method analytical \
    --kernel_size 8 \
    --ckpt_path /path/to/checkpoint.ckpt \
    --device cuda
```

Results are appended to `logg.txt` in the current directory. The script reports:

- **MSE** – mean squared error of predicted means vs. true labels  
- **Average log-probability** – mean NLL of the predicted distribution  
- **Coverage at ±0.15 / ±0.25 / ±0.35 / ±0.50 / ±1.0** – fraction of true labels inside the predicted interval

## Dependencies

```
pytorch-lightning
torch
torchvision
albumentations
scikit-image
scikit-learn
scipy
matplotlib
pandas
numpy
```
