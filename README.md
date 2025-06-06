# CS-493s Final Project: Knowledge Distillation for Language Models

This project implements knowledge distillation for language models on the HellaSwag and GSM8K datasets. The goal is to train a smaller "student" model to mimic a larger "teacher" model, using both label and logits distillation techniques.

## Features
- Teacher-student distillation (label and logits)
- Support for HellaSwag and GSM8K datasets
- Training, evaluation, and logging utilities
- Model checkpointing and loss visualization

## Project Structure
```
scripts/
    dataloader.py         # Dataset loading utilities
    distill_utils.py      # Device selection, parameter counting, etc.
    models.py             # Teacher and student model definitions
    train.py              # Main training script for distillation
    model_evaluate.py     # Evaluation utilities
    runall.py             # (Optional) Script to run all experiments
requirements.txt          # Python dependencies
README.md                 # Project documentation
```

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
- Download the HellaSwag and GSM8K datasets. Update the paths in `scripts/dataloader.py` if needed to point to your data files.

### 3. Train Student Models
```bash
python scripts/train.py
```
This will train student models using both label and logits distillation on both datasets. Model checkpoints and logs will be saved as configured in the scripts.

### 4. Evaluate Models
Use `scripts/model_evaluate.py` to evaluate trained student models.

## Requirements
- Python 3.10+
- PyTorch
- matplotlib
- (Other dependencies as listed in `requirements.txt`)

## Citation
If you use this code for academic purposes, please cite the original datasets and any relevant model papers.