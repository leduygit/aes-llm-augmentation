# Improving Automated Essay Scoring with Targeted LLM-Based Data Augmentation under Imbalanced Data

[![Paper](https://img.shields.io/badge/Paper-KES%202026-blue)](#)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](#)

This repository provides the implementation artifact for the KES 2026 paper **"Improving Automated Essay Scoring with Targeted LLM-Based Data Augmentation under Imbalanced Data."** It contains a lightweight, standalone pipeline for training transformer-based Automated Essay Scoring (AES) models and generating targeted synthetic IELTS Writing Task 2 essays with Large Language Models.

The codebase is designed for local execution and modular reproduction of the paper workflow: train an AES model, diagnose weak IELTS bands, generate targeted synthetic essays, add them to the training split, and retrain. It intentionally avoids remote experiment tracking so the workflow stays simple to run in local or Kaggle environments.

## Reference Results

Our targeted augmentation framework reduces model bias toward majority mid-range bands by adding synthetic samples where validation errors are highest. We evaluate this strategy across multiple transformer backbones and report both representative absolute performance and average performance deltas.

### Absolute Performance (Before vs. After Augmentation)

The table below reports baseline performance without augmentation for each backbone and the representative augmented result for the BERT-based model. The augmented BERT setting achieves higher agreement with human raters and lower error while showing tighter variance across runs.

| Data Setting | Model | QWK (↑) | MAE (↓) | MSE (↓) |
| :--- | :--- | :--- | :--- | :--- |
| **No Augmentation** | DeBERTa-Base | 0.53 ± 0.30 | 0.75 ± 0.14 | 0.82 ± 0.32 |
| **No Augmentation** | RoBERTa-Base | 0.60 ± 0.06 | 0.73 ± 0.06 | 0.77 ± 0.14 |
| **No Augmentation** | BERT-Large-Uncased | 0.67 ± 0.06 | 0.68 ± 0.05 | 0.69 ± 0.09 |
| **Augmentation** | **Our model (Backbone: BERT)** | **0.77 ± 0.03** | **0.58 ± 0.01** | **0.52 ± 0.03** |

### Average Performance Differences

To summarize generalization across architectures, the following table reports average metric changes after adding targeted synthetic data to each model's training pipeline.

| Model | Mean MAE Diff (↓) | Mean MSE Diff (↓) | Mean QWK Diff (↑) |
| :--- | :--- | :--- | :--- |
| **DeBERTa-Base** | -0.159 | -0.272 | +0.235 |
| **RoBERTa-Base** | -0.119 | -0.212 | +0.159 |
| **BERT-Large-Uncased** | -0.100 | -0.164 | +0.101 |
| **All Models (Average)** | **-0.126** | **-0.216** | **+0.165** |

> **Note on Reproducibility:** The metrics above are reference results from the full paper pipeline and multiple independent runs. Local results may differ depending on dataset version, model backbone, provider/model choice, generated samples, random seeds, hardware, and the number of augmentation rounds.

## Repository Structure

```text
llm-aes-aug/
├── configs/                   # YAML training configuration files
├── outputs/
│   ├── metrics/               # Generated evaluation metrics
│   └── synthetic/             # Generated synthetic IELTS CSV files
├── synthetic_assets/          # Questions, few-shot exemplars, and band descriptions
├── generate_synthetic_data.py # LLM synthesis script for OpenAI and Claude
├── pipeline_training.py       # Training and evaluation pipeline
├── requirements.txt           # Python dependencies
└── setup.sh                   # Environment setup script
```

## Installation

From the project directory:

```bash
cd /kaggle/working/llm-aes-aug
bash setup.sh
source venv/bin/activate
```

The setup script creates `venv/`, installs the dependencies in `requirements.txt`, and verifies the main runtime packages.

## API Keys

Create a local `.env` file in the repository root. This file should not be committed.

```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional model overrides
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_MODEL=claude-sonnet-4-0
```

The repository defaults to `gpt-4o-mini` for lower-cost OpenAI generation. For paper-style OpenAI generation, pass `--model gpt-4o` in the generation command. For Anthropic, use the configured Claude Sonnet model available to your account.

## Iterative Augmentation Workflow

The paper workflow follows an error-driven loop:

```text
Evaluation -> Diagnosis -> Targeted Generation -> Data Integration -> Retraining
```

This repository provides standalone scripts for each step. The original full experiment loop was orchestrated externally, so the loop is reproduced here manually.

### Step 1: Train The Baseline Model

```bash
venv/bin/python pipeline_training.py
```

The training script loads `configs/base.yaml`, then merges `configs/training_worst_band.yaml` by default. It saves local artifacts such as:

```text
best_model.pth
scaler_config.pkl
outputs/metrics/per_band_val_metrics.csv
```

### Step 2: Identify The Weakest Band

Inspect the terminal output or open:

```text
outputs/metrics/per_band_val_metrics.csv
```

Select the validation band with the highest per-band MAE. This is the target band for the next synthetic generation round.

### Step 3: Generate Targeted Synthetic Essays

Generate 32 essays for the weakest band with 3 refinement iterations. For example, if Band 5 is the weakest band:

Using OpenAI with the paper-style model:

```bash
venv/bin/python generate_synthetic_data.py 5 --provider openai --model gpt-4o --num-essays 32 --max-iterations 3
```

Using Claude:

```bash
venv/bin/python generate_synthetic_data.py 5 --provider claude --num-essays 32 --max-iterations 3
```

Generated files are written to:

```text
outputs/synthetic/band_<bands>_<timestamp>.csv
```

Each generated CSV contains:

```text
Question,Essay,Overall
```

### Step 4: Integrate Synthetic Data

Copy the generated CSV into the training folder configured in `configs/base.yaml`:

```bash
cp outputs/synthetic/band_5_*.csv /path/to/data/original_24-09-20-04-09/training_worst_band/
```

The generator produces flat CSV files only. It does not create train, validation, or test split folders automatically.

### Step 5: Retrain And Repeat

```bash
venv/bin/python pipeline_training.py
```

Repeat Steps 2-5 for the desired number of augmentation rounds.

## Data Configuration

Training data paths are configured in `configs/base.yaml`. The pipeline expects this directory layout:

```text
<data_path>/<data_folder>/
├── <train_folder>/       # Original training CSVs plus injected synthetic CSVs
├── <validation_folder>/  # Validation CSVs
└── <test_folder>/        # Test CSVs
```

Example configuration:

```yaml
data_path: /kaggle/input/mielband-ielts-data-train
data_folder: original_24-09-20-04-09
train_folder: training_worst_band
validation_folder: validation
test_folder: test
```

With this configuration, the script reads CSV files from:

```text
/kaggle/input/mielband-ielts-data-train/original_24-09-20-04-09/training_worst_band/
/kaggle/input/mielband-ielts-data-train/original_24-09-20-04-09/validation/
/kaggle/input/mielband-ielts-data-train/original_24-09-20-04-09/test/
```

Input CSV files should contain:

```text
Question,Essay,Overall
```

The loader also accepts lowercase `question` and `answer` columns, which are renamed internally to `Question` and `Essay`.

## Training Configuration

Use environment overrides for quick checks:

```bash
EPOCHS=1 BATCH_SIZE=16 venv/bin/python pipeline_training.py
```

Use YAML settings for paper-style runs:

```yaml
bert_model_name: microsoft/deberta-base
max_seq_length: 512
batch_size: 32
epochs: 200
learning_rate: 0.0000458115
patience: 50
use_sigmoid: false
checkpoint_path: ./best_model.pth
scaler_path: ./scaler_config.pkl
```

To use a different config override file, set `DATA_AUGMENTATION_METHOD` to the filename without `.yaml`:

```bash
DATA_AUGMENTATION_METHOD=training_worst_band venv/bin/python pipeline_training.py
```

## Synthetic Generation Examples

Generate one Band 5 essay with OpenAI:

```bash
venv/bin/python generate_synthetic_data.py 5 --provider openai --num-essays 1 --max-iterations 1
```

Generate one Band 5 essay with Claude:

```bash
venv/bin/python generate_synthetic_data.py 5 --provider claude --num-essays 1 --max-iterations 1
```

Generate three essays for each of Bands 5, 6, and 7 with OpenAI:

```bash
venv/bin/python generate_synthetic_data.py 5,6,7 --provider openai --num-essays 3 --max-iterations 1
```

## Citation

If you use this codebase or generated data in your research, please cite the paper:

```bibtex
@inproceedings{le2026aesllmaugmentation,
  title={Improving Automated Essay Scoring with Targeted LLM-Based Data Augmentation under Imbalanced Data},
  author={Le, Duy Anh and Vo Thanh, Nghia and Trieu, Huy and Nam, Van Chi and Nguyen, Huy Tien and Le, Tung},
  booktitle={30th International Conference on Knowledge-Based and Intelligent Information \& Engineering Systems (KES 2026)},
  year={2026},
  note={To appear}
}
```
