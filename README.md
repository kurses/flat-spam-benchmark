# flat-spam-benchmark
PyTorch implementation of Feature-Level Adversarial Training (FLAT) for robust spam detection. Benchmarks BERT, ALBERT, and DistilBERT against synonym substitution attacks (PWWS).

# FLAT Benchmarking Suite

A comprehensive benchmarking framework for evaluating transformer models (BERT, ALBERT, DistilBERT) using FLAT (Feature Learning for Adversarial Training) methodology. This suite measures model accuracy, robustness to adversarial attacks, and interpretability metrics.

## Overview

The FLAT methodology combines adversarial training with feature learning to improve both model robustness and interpretability. This benchmarking suite automates:

- Training: Multi-iteration FLAT training with adversarial example generation
- Evaluation: Clean accuracy, after-attack accuracy, and interpretability metrics
- Robustness: PWWS adversarial attack evaluation
- Interpretability: Kendall's Tau correlation and Top-K intersection analysis
- Visualization: Loss curves, performance comparison, and confusion matrix analysis

## Features

- Multi-model support (BERT, ALBERT, DistilBERT)
- Adversarial robustness evaluation using TextAttack
- Interpretability metrics via embedding saliency
- Automatic checkpoint saving and recovery
- Detailed CSV reporting and PNG visualizations
- GPU-accelerated training
- Progress monitoring and logging

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA support recommended)
- GPU (optional but highly recommended for faster training)

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:kurses/flat-spam-benchmark.git
cd flat-spam-benchmark
```

### 2. Create a Conda Environment

```bash
conda create -n flat-env python=3.10
conda activate flat-env
```

### 3. Install PyTorch with CUDA

Visit [PyTorch's official installation page](https://pytorch.org/get-started/locally/) and select your CUDA version. Example for CUDA 12.1:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

For CPU-only (not recommended):
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import AutoTokenizer; print('Transformers OK')"
python -c "import textattack; print('TextAttack OK')"
```

## Quick Start

### Train Default Models

```bash
python benchmark_suite.py
```

This trains BERT, ALBERT, and DistilBERT using FLAT methodology (default: 5 iterations × 2 epochs).

### Evaluate Saved Checkpoints (No Training)

```bash
python benchmark_suite.py --eval-only
```

Automatically detects `flat_*.pt` checkpoint files and regenerates metrics and plots.

### Skip Robustness Evaluation (Faster)

```bash
python benchmark_suite.py --eval-only --skip-robustness
```

Useful for quick evaluation without TextAttack robustness testing. Note: `AA_Accuracy`, `Attack_Success_Rate`, and interpretability metrics will be 0.0.

### Train Specific Models

```bash
python benchmark_suite.py --models bert-base-uncased,albert-base-v2
```

## Command-Line Arguments

```
optional arguments:
  -h, --help             Show help message
  --eval-only            Evaluate saved checkpoints without training
  --skip-robustness      Skip TextAttack robustness evaluation (faster)
  --models MODEL1,MODEL2 Comma-separated list of models to train/evaluate
```

## Configuration

Edit these variables in `benchmark_suite.py` to customize training:

```python
# Models to benchmark
MODELS = ['bert-base-uncased', 'albert-base-v2', 'distilbert-base-uncased']

# FLAT hyperparameters
NUM_FLAT_ITERATIONS = 5          # Number of FLAT iterations
NUM_EPOCHS_PER_ITERATION = 2     # Epochs per iteration
GAMMA = 0.001                     # Adversarial weight
TRAIN_ATTACK_SAMPLES = 200        # Samples for adversarial generation (-1 for full)

# Evaluation
EVAL_SUBSET_SIZE = 50             # Samples for robustness evaluation
MAX_QUERIES_TA = 750              # TextAttack query budget per sample
ATTACK_BATCH_SIZE = 8             # Batch size for attacks

# Data
MAX_SEQ_LENGTH = 256              # Max sequence length
```

## Output Files

After successful completion:

```
├── flat_bert-base-uncased.pt                    # BERT checkpoint
├── flat_albert-base-v2.pt                       # ALBERT checkpoint
├── flat_distilbert-base-uncased.pt              # DistilBERT checkpoint
├── training_curves_bert-base-uncased.csv        # Per-epoch metrics
├── training_curves_albert-base-v2.csv
├── training_curves_distilbert-base-uncased.csv
├── benchmark_report.csv                         # Summary metrics table
└── img/
    ├── loss_curves.png                          # Training loss over epochs
    ├── performance_comparison.png               # Accuracy vs robustness
    └── confusion_matrix_components.png          # TP/TN/FP/FN by model
```

## Benchmark Report Columns

| Column | Description |
|--------|-------------|
| `Model` | Model name |
| `Accuracy` | Final accuracy on clean test data |
| `Precision` | True positives / (TP + FP) |
| `Recall` | True positives / (TP + FN) |
| `F1` | Harmonic mean of precision & recall |
| `AA_Accuracy` | Accuracy after adversarial attack |
| `Attack_Success_Rate` | 1 - AA_Accuracy |
| `Kendall_Tau` | Correlation of token importance rankings |
| `Top_K_Intersection` | Overlap in top-10 important tokens |

## Dataset

The script uses the **Spam/Ham Email Dataset** from Kaggle:
- Source: [venky73/spam-mails-dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset)
- Task: Binary classification (spam vs. legitimate emails)
- Split: 70% train / 15% dev / 15% test

The dataset is automatically downloaded via `kagglehub` on first run.

## Performance & Runtime

Times vary based on:
- GPU capabilities
- Batch sizes
- `TRAIN_ATTACK_SAMPLES` (adversarial generation is slow)
- `EVAL_SUBSET_SIZE` (robustness evaluation via TextAttack is expensive)

**Faster Testing:**
```bash
# Reduce iterations and samples for quick testing
# Edit benchmark_suite.py:
NUM_FLAT_ITERATIONS = 1
TRAIN_ATTACK_SAMPLES = 50
EVAL_SUBSET_SIZE = 10
```

## Troubleshooting

### Out of Memory (OOM)

```python
# In benchmark_suite.py, reduce:
ATTACK_BATCH_SIZE = 4          # Default: 8
TRAIN_ATTACK_SAMPLES = 100     # Default: 200
NUM_FLAT_ITERATIONS = 2        # Default: 5
```

Or run one model at a time:
```bash
python benchmark_suite.py --models bert-base-uncased
python benchmark_suite.py --models albert-base-v2
```

### Slow Training

- Reduce `TRAIN_ATTACK_SAMPLES` (adversarial generation is the bottleneck)
- Reduce `NUM_FLAT_ITERATIONS`
- Use `--skip-robustness` for evaluation

### TextAttack/Robustness Timeout

```bash
# Use faster evaluation
python benchmark_suite.py --eval-only --skip-robustness

# Or reduce query budget in benchmark_suite.py:
MAX_QUERIES_TA = 300
EVAL_SUBSET_SIZE = 20
```

### CUDA/PyTorch Issues

Check CUDA availability:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Reinstall PyTorch for your CUDA version (see [PyTorch Installation](https://pytorch.org/get-started/locally/)).

## Usage Examples

### Complete Benchmark Run

```bash
conda activate flat-env
python benchmark_suite.py
```

### Analyze Results

```python
import pandas as pd

# Load report
df = pd.read_csv('benchmark_report.csv')

# Find best model by accuracy
best_accuracy = df.loc[df['Accuracy'].idxmax()]
print(f"Best accuracy: {best_accuracy['Model']}")

# Find most robust
most_robust = df.loc[df['AA_Accuracy'].idxmax()]
print(f"Most robust: {most_robust['Model']}")

# Compare all metrics
print(df.to_string())
```

### Load Trained Model

```python
import torch
from benchmark_suite import FLATUniversalModel

# Load BERT checkpoint
model = FLATUniversalModel('bert-base-uncased', vocab_size=30522)
model.load_state_dict(torch.load('flat_bert-base-uncased.pt'))
model.eval()

# Use for inference
with torch.no_grad():
    outputs = model(input_ids, attention_mask)
```

## Citation

If you use this benchmark suite in your research, please cite:

```bibtex
@software{flat_benchmark,
  title = {FLAT Benchmarking Suite},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/kurses/flat-spam-benchmark}
}
```

## References

```bibtex
@inproceedings{flat_paper,
  title={Adversarial training for improving model robustness? Look at both prediction and interpretation},
  author={Chen, Hanjie and Ji, Yangfeng},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={36},
  number={10},
  pages={10463--10472},
  year={2022}
}

@article{bert_paper,
  title={Bert: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle={Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers)},
  pages={4171--4186},
  year={2019}
}

@article{albert_paper,
  title={Albert: A lite bert for self-supervised learning of language representations},
  author={Lan, Zhenzhong and Chen, Mingda and Goodman, Sebastian and Gimpel, Kevin and Sharma, Piyush and Soricut, Radu},
  journal={arXiv preprint arXiv:1909.11942},
  year={2019}
}

@article{distilbert,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  journal={arXiv preprint arXiv:1910.01108},
  year={2019}
}

@inproceedings{pwws_paper,
  title={Generating natural language adversarial examples through probability weighted word saliency},
  author={Ren, Shuhuai and Deng, Yihe and He, Kun and Che, Wanxiang},
  booktitle={Proceedings of the 57th annual meeting of the association for computational linguistics},
  pages={1085--1097},
  year={2019}
}

@inproceedings{textattack,
  title={Textattack: A framework for adversarial attacks, data augmentation, and adversarial training in nlp},
  author={Morris, John and Lifland, Eli and Yoo, Jin Yong and Grigsby, Jake and Jin, Di and Qi, Yanjun},
  booktitle={Proceedings of the 2020 conference on empirical methods in natural language processing: System demonstrations},
  pages={119--126},
  year={2020}
}

@inproceedings{transformers,
  title={Transformers: State-of-the-art natural language processing},
  author={Wolf, Thomas and Debut, Lysandre and Sanh, Victor and Chaumond, Julien and Delangue, Clement and Moi, Anthony and Cistac, Pierric and Rault, Tim and Louf, Remi and Funtowicz, Morgan and others},
  booktitle={Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations},
  pages={38--45},
  year={2020}
}
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a clear description

## Issues & Support

For bugs, questions, or feature requests, please open an issue on GitHub.
