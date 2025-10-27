# Traceformer
Traceformer is a transformer-based side-channel analysis framework for masked AES on ASCAD. It uses a BERT plaintext pathway with late fusion, a fixed 1,400-sample window, key-rank-0 evaluation, cross-desynchronization testing, and saved indices/normalization for fully reproducible experiments (including ablations and per-byte heatmaps).


# Traceformer — BERT for Side-Channel Analysis on ASCAD

Transformer-based side-channel analysis (SCA) on ASCAD v1 **EM traces**.
This repo accompanies the thesis “Exploring the Performance of BERT Transformers on ASCAD Variable Datasets” and evaluates a BERT-based late-fusion model for masked software AES-128.

## Highlights
- **BERT + late fusion:** plaintext is encoded by a pretrained BERT; traces are linearly projected; embeddings are fused and classified over 256 masked S-box values.
- **ASCAD variable-key (desync 0/50/100):** fixed **1,400-sample** window around the first SubBytes; **no re-windowing** across settings.
- **Key-rank evaluation (KR-0):** accumulate per-class **log-likelihoods** over attack traces; report **top-rank=0**; include full-key and per-byte results.
- **Cross-desync testing:** **train on desync=0**, **test on 50/100** with **no fine-tuning**.
- **Reproducibility:** z-score statistics taken **only from profiling**; attack split indices and normalization saved; fixed seeds; W&B logging.
- **Multiple plaintext encodings:** default hex tokenization (00–ff per byte); variants (ascii/byte) available in configs.


## Datasets
Built for **ASCAD v1 (ANSSI)** EM datasets:
- `ascad-variable.h5` (desync=0)
- `ascad-variable_desync50.h5` (desync=50)
- `ascad-variable_desync100.h5` (desync=100)
- *(optional)* `ascad-fixed.h5`

> Datasets are **not included**. Please obtain them from the official ASCAD release. Do not redistribute.

## Evaluation Protocol (TL;DR)
- **Window:** single 1,400 samples for all bytes and desync levels.
- **Normalization:** z-score using **profiling** set stats, applied to attack sets.
- **KR-0:** accumulate log-likelihoods over attack traces; rank true key among 256 hypotheses; top-rank = 0.
- **Full-key success:** all 16 byte-wise ranks reach 0 within the same trace budget.
- **Cross-desync:** train@0 → test@50/100; **no** re-windowing/fine-tuning.

## Reproducibility
- Fixed random seeds; saved normalization arrays and attack split indices.
- W&B logging for runs; configs under `configs/`.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()


