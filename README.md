# AES LLM Augmentation

Utilities for generating synthetic IELTS Task 2 essays with OpenAI or Claude, then training an AES model on configured IELTS data.

## Setup

From the project directory:

```bash
cd /kaggle/working/llm-aes-aug
bash setup.sh
source venv/bin/activate
```

The setup script creates `venv/`, installs PyTorch and the dependencies in `requirements.txt`, and verifies the main packages.

## Environment Variables

Create a local `.env` file in the repo root. Do not commit this file.

For OpenAI generation:

```bash
OPENAI_API_KEY=your_openai_key
```

For Claude generation:

```bash
ANTHROPIC_API_KEY=your_anthropic_key
```

Optional model overrides:

```bash
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_MODEL=claude-sonnet-4-0
```

## Generate Synthetic Data

The generator reads prompt assets from `synthetic_assets/` and writes flat CSV files to `outputs/synthetic/`.

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

Generated files use this pattern:

```text
outputs/synthetic/band_<bands>_<timestamp>.csv
```

Each generated CSV contains:

```text
Question,Essay,Overall
```


## Configure Training YAML

Training configuration lives in `configs/base.yaml`. The file named by `data_augmentation_method` is merged on top of it; by default that is `configs/training_worst_band.yaml`.

The most important part is the data path layout. The training script builds paths like this:

```text
<data_path>/<data_folder>/<train_folder>/*.csv
<data_path>/<data_folder>/<validation_folder>/*.csv
<data_path>/<data_folder>/<test_folder>/*.csv
```

Example:

```yaml
data_path: /kaggle/input/mielband-ielts-data-train
data_folder: original_24-09-20-04-09
train_folder: training_worst_band
validation_folder: validation
test_folder: test
```

With that config, the script expects CSV files under:

```text
/kaggle/input/mielband-ielts-data-train/original_24-09-20-04-09/training_worst_band/
/kaggle/input/mielband-ielts-data-train/original_24-09-20-04-09/validation/
/kaggle/input/mielband-ielts-data-train/original_24-09-20-04-09/test/
```

Each CSV should contain essay data with columns compatible with the loader. The common format is:

```text
Question,Essay,Overall
```

The loader also accepts lowercase source columns `question` and `answer`; it renames them to `Question` and `Essay` during preprocessing.

If you want to train with generated synthetic data, copy the generated CSV from `outputs/synthetic/` into the configured training folder, for example:

```bash
cp outputs/synthetic/band_5_15-05-26-11-08.csv /path/to/data/original_24-09-20-04-09/training_worst_band/
```

You still need valid `validation/` and `test/` folders. The generator does not create those splits.

Other useful config examples:

```yaml
bert_model_name: microsoft/deberta-base
max_seq_length: 512
batch_size: 32
epochs: 50
learning_rate: 0.0000458115
patience: 50
use_sigmoid: false
checkpoint_path: ./best_model.pth
scaler_path: ./scaler_config.pkl
```

To use a different override file, set `DATA_AUGMENTATION_METHOD` to the filename without `.yaml`:

```bash
DATA_AUGMENTATION_METHOD=training_worst_band venv/bin/python pipeline_training.py
```


## Manual Paper Reproduction Workflow

This repository exposes the paper pipeline as manual standalone steps. The original full experiment loop was orchestrated externally, so this repo does not automatically run the entire train-diagnose-generate-retrain cycle for you.

The manual loop is:

1. Train a baseline model.
2. Read validation metrics from the terminal output.
3. Identify the weakest validation band.
4. Generate synthetic essays for that target band.
5. Copy the generated CSV into the configured training folder.
6. Retrain the model.
7. Repeat for the desired number of augmentation rounds.

Train the baseline:

```bash
venv/bin/python pipeline_training.py
```

Choose the weakest band from validation analysis. For example, if Band 5 is the weakest band, generate 32 targeted essays with 3 refinement iterations:

```bash
venv/bin/python generate_synthetic_data.py 5 --provider openai --num-essays 32 --max-iterations 3
```

Claude version:

```bash
venv/bin/python generate_synthetic_data.py 5 --provider claude --num-essays 32 --max-iterations 3
```

Copy the generated CSV into your configured training folder:

```bash
cp outputs/synthetic/band_5_*.csv /path/to/data/original_24-09-20-04-09/training_worst_band/
```

Retrain after adding the generated data:

```bash
venv/bin/python pipeline_training.py
```

For the paper-style loop, repeat this process for multiple augmentation rounds. In the paper experiments, the targeted generation step used a batch of 32 essays and 3 refinement iterations per selected band.

### Choosing The Worst Band

The current training script prints overall validation and test metrics. For exact manual reproduction of the paper workflow, the training script should also print or save per-band validation MAE so users can directly select the weakest band from this repo.

Until per-band validation metrics are added here, use this manual workflow when you already know the target band from external validation analysis or prior experiment logs.

## Reference Paper Results

The paper reports the following average improvements from the full experimental setup:

```text
Average QWK improvement: +0.165
Average MAE reduction:   -0.126
Average MSE reduction:   -0.216
```

These are reference results from the full paper pipeline, not guaranteed outputs from a single local README command. Exact values depend on dataset version, model backbone, provider/model, generated samples, random seeds, and the number of augmentation rounds.

## Train The Model

Run the training pipeline:

```bash
venv/bin/python pipeline_training.py
```

The training script loads `configs/base.yaml`, then merges `configs/training_worst_band.yaml` by default. It expects the configured data path to contain train, validation, and test split folders.

Training saves local artifacts such as:

```text
best_model.pth
scaler_config.pkl
```

## Config Notes

Synthetic generation produces flat CSV files only. It does not automatically create the train/validation/test folder layout expected by the training pipeline.

To train on generated data, place or copy the generated CSV into the training folder configured by `configs/base.yaml`, while keeping compatible validation and test folders available.
