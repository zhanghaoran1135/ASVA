# ASVA: Attack path and Syntax logic analysis for Vulnerability severity Assessment

ASVA is a vulnerability severity assessment framework that integrates attack path modeling with multi-scale syntactic logic analysis at the source-code level.

This repository contains the core implementation of ASVA and resources for reproducing the main experiments in our paper.

## Installation

```bash
sudo apt install openjdk-11-jdk
conda env create -f environment.yml
```

You should also 
download the pre-trained CodeBERT model and place it in `data/codebert-base/`(see `data/README.md` for instructions).
download the joern tool and place it in `joern/`(see `tools/README.md` for instructions).


## Train and Evaluate Models

### 0. Download datasets

Place the datasets in the `data/`

### 1. Prepare data

```bash
python prepare_csv.py  # wait for joern to run
python prepare_data.py --config configs/default.yaml
```

### 2. Train

```bash
python train.py --config configs/default.yaml
```

### 3. Evaluate

```bash
python evaluate.py \
  --config configs/default.yaml \
  --checkpoint artifacts/checkpoints/best.pt \
  --split test
```
