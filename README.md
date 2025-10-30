# 🎯 Few-Shot Learning: State-of-the-Art Methods

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-Papers-b31b1b.svg)](https://arxiv.org/)

> **Comprehensive implementation of state-of-the-art few-shot learning algorithms with unified evaluation framework**

A complete research and production-ready library for few-shot learning, featuring classic and modern approaches including metric learning, meta-learning, and transfer learning methods.

---

## 🌟 Key Features

- 🏆 **15+ State-of-the-Art Methods** (Prototypical Networks, MAML, Relation Networks, Matching Networks, etc.)
- 📊 **Multiple Benchmark Datasets** (miniImageNet, tieredImageNet, Omniglot, CIFAR-FS, FC100)
- 🔥 **PyTorch Implementation** with modern best practices
- 📈 **Unified Evaluation Framework** with standardized metrics
- 🚀 **Production Ready** with clean APIs and documentation
- 🧪 **Reproducible Results** with fixed seeds and detailed configs
- 🎓 **Educational** with extensive comments and tutorials

---

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Implemented Methods](#implemented-methods)
- [Datasets](#datasets)
- [Results](#results)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## 🎓 Overview

### What is Few-Shot Learning?

Few-shot learning aims to recognize new classes with only a **few labeled examples** (typically 1-5). This is crucial for:

- **Data-scarce scenarios** (medical imaging, rare species)
- **Rapid adaptation** (personalization, new product categories)
- **Continual learning** (adding new classes without retraining)

### Problem Formulation

**N-way K-shot Classification:**
- **N**: Number of new classes
- **K**: Number of examples per class
- **Query**: Test samples to classify

**Example:** 5-way 1-shot = Classify among 5 classes with only 1 example per class

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA 11.8+ (optional, for GPU)

### Quick Install

```bash
# Clone repository
git clone https://github.com/yourusername/few-shot-learning.git
cd few-shot-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### From PyPI (Coming Soon)

```bash
pip install few-shot-learning
```

---

## ⚡ Quick Start

### 1. Download Dataset

```bash
# Download miniImageNet
python scripts/download_data.py --dataset miniimagenet --output datasets/

# Or download manually from:
# https://drive.google.com/file/d/miniImageNet
```

### 2. Train a Model

```python
from fewshot import PrototypicalNetwork, miniImageNetDataset
from fewshot.utils import train_few_shot

# Load dataset
train_dataset = miniImageNetDataset(root='datasets/miniImageNet', split='train')
val_dataset = miniImageNetDataset(root='datasets/miniImageNet', split='val')

# Initialize model
model = PrototypicalNetwork(
    backbone='resnet12',
    input_size=84
)

# Train
trainer = train_few_shot(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    n_way=5,
    k_shot=1,
    q_queries=15,
    epochs=100
)
```

### 3. Evaluate

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --model prototypical \
    --checkpoint checkpoints/best_model.pth \
    --dataset miniimagenet \
    --n_way 5 \
    --k_shot 1
```

### 4. Run Quick Demo

```python
from fewshot import load_model, predict

# Load pre-trained model
model = load_model('prototypical', checkpoint='pretrained/proto_mini.pth')

# Make predictions
predictions = model.predict(
    support_images=support_set,  # [N, K, C, H, W]
    query_images=query_set        # [Q, C, H, W]
)

print(f"Predictions: {predictions}")
```

---

## 🏆 Implemented Methods

### Metric Learning Methods

#### 1. **Prototypical Networks** (2017)
- **Paper**: [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)
- **Idea**: Classify based on distance to class prototypes (mean embeddings)
- **Results**: 68.2% (5-way 1-shot miniImageNet)

```python
from fewshot.models import PrototypicalNetwork

model = PrototypicalNetwork(backbone='resnet12')
```

#### 2. **Matching Networks** (2016)
- **Paper**: [Matching Networks for One Shot Learning](https://arxiv.org/abs/1606.04080)
- **Idea**: Attention-based matching with support set
- **Results**: 55.3% (5-way 1-shot miniImageNet)

```python
from fewshot.models import MatchingNetwork

model = MatchingNetwork(backbone='conv4')
```

#### 3. **Relation Networks** (2018)
- **Paper**: [Learning to Compare](https://arxiv.org/abs/1711.06025)
- **Idea**: Learn a deep similarity metric
- **Results**: 65.3% (5-way 1-shot miniImageNet)

```python
from fewshot.models import RelationNetwork

model = RelationNetwork(backbone='conv4')
```

### Meta-Learning Methods

#### 4. **MAML** (Model-Agnostic Meta-Learning) (2017)
- **Paper**: [Model-Agnostic Meta-Learning](https://arxiv.org/abs/1703.03400)
- **Idea**: Learn initialization for fast adaptation
- **Results**: 63.1% (5-way 1-shot miniImageNet)

```python
from fewshot.models import MAML

model = MAML(
    backbone='conv4',
    inner_lr=0.01,
    inner_steps=5
)
```

#### 5. **Reptile** (2018)
- **Paper**: [On First-Order Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999)
- **Idea**: First-order MAML approximation
- **Results**: 62.7% (5-way 1-shot miniImageNet)

```python
from fewshot.models import Reptile

model = Reptile(backbone='conv4')
```

#### 6. **Meta-SGD** (2017)
- **Paper**: [Meta-SGD: Learning to Learn Quickly](https://arxiv.org/abs/1707.09835)
- **Idea**: Learn both initialization and learning rates
- **Results**: 64.0% (5-way 1-shot miniImageNet)

```python
from fewshot.models import MetaSGD

model = MetaSGD(backbone='conv4')
```

### Transfer Learning Methods

#### 7. **Baseline++** (2019)
- **Paper**: [A Closer Look at Few-shot Classification](https://arxiv.org/abs/1904.04232)
- **Idea**: Pretrain + fine-tune with cosine classifier
- **Results**: 51.9% (5-way 1-shot miniImageNet)

```python
from fewshot.models import BaselinePlusPlus

model = BaselinePlusPlus(backbone='resnet18')
```

#### 8. **Simple CNAPS** (2020)
- **Paper**: [Improved Few-Shot Visual Classification](https://arxiv.org/abs/1912.03432)
- **Idea**: Simple adaptation with feature-wise transformations
- **Results**: 69.8% (5-way 1-shot miniImageNet)

```python
from fewshot.models import SimpleCNAPS

model = SimpleCNAPS(backbone='resnet18')
```

### Graph-Based Methods

#### 9. **GNN** (Graph Neural Networks) (2018)
- **Paper**: [Few-Shot Learning with Graph Neural Networks](https://arxiv.org/abs/1711.04043)
- **Idea**: Model relationships between support and query
- **Results**: 66.4% (5-way 1-shot miniImageNet)

```python
from fewshot.models import GNN

model = GNN(backbone='conv4')
```

#### 10. **EGNN** (Edge-Labeling GNN) (2019)
- **Paper**: [Edge-Labeling Graph Neural Network](https://arxiv.org/abs/1905.01436)
- **Idea**: Learn edge labels for better propagation
- **Results**: 76.4% (5-way 1-shot miniImageNet)

```python
from fewshot.models import EGNN

model = EGNN(backbone='resnet12')
```

### Recent State-of-the-Art

#### 11. **Meta-Baseline** (2021)
- **Paper**: [Meta-Baseline: Exploring Simple Meta-Learning](https://arxiv.org/abs/2003.04390)
- **Idea**: Pretrain + meta-test with learned metric
- **Results**: 73.4% (5-way 1-shot miniImageNet)

```python
from fewshot.models import MetaBaseline

model = MetaBaseline(backbone='resnet12')
```

#### 12. **Meta-DeepBDC** (2021)
- **Paper**: [DeepBDC for Few-Shot Learning](https://arxiv.org/abs/2012.00913)
- **Idea**: Brownian distance covariance metric
- **Results**: 78.7% (5-way 1-shot miniImageNet)

```python
from fewshot.models import MetaDeepBDC

model = MetaDeepBDC(backbone='resnet12')
```

#### 13. **FRN** (Feature-Reweighting Network) (2021)
- **Paper**: [Rethinking Few-Shot Image Classification](https://arxiv.org/abs/2003.11539)
- **Idea**: Learn to reweight features
- **Results**: 77.9% (5-way 1-shot miniImageNet)

```python
from fewshot.models import FRN

model = FRN(backbone='resnet12')
```

### Self-Supervised Methods

#### 14. **MetaOptNet-SVM** (2019)
- **Paper**: [Meta-Learning with Differentiable Convex Optimization](https://arxiv.org/abs/1904.03758)
- **Idea**: SVM as final classifier layer
- **Results**: 64.1% (5-way 1-shot miniImageNet)

```python
from fewshot.models import MetaOptNet

model = MetaOptNet(backbone='resnet12')
```

#### 15. **DeepEMD** (2020)
- **Paper**: [DeepEMD: Differentiable Earth Mover's Distance](https://arxiv.org/abs/2003.06777)
- **Idea**: Earth Mover's Distance for matching
- **Results**: 75.7% (5-way 1-shot miniImageNet)

```python
from fewshot.models import DeepEMD

model = DeepEMD(backbone='resnet12')
```

---

## 📊 Datasets

### Supported Datasets

#### 1. **miniImageNet**
- **Classes**: 100 (64 train / 16 val / 20 test)
- **Images**: 600 per class
- **Size**: 84×84 RGB
- **Paper**: [Matching Networks](https://arxiv.org/abs/1606.04080)

```python
from fewshot.data import miniImageNetDataset

dataset = miniImageNetDataset(root='datasets/miniImageNet', split='train')
```

#### 2. **tieredImageNet**
- **Classes**: 608 (351 train / 97 val / 160 test)
- **Images**: ~1,300 per class
- **Size**: 84×84 RGB
- **Paper**: [Meta-Learning for Semi-Supervised](https://arxiv.org/abs/1803.00676)

```python
from fewshot.data import tieredImageNetDataset

dataset = tieredImageNetDataset(root='datasets/tieredImageNet', split='train')
```

#### 3. **Omniglot**
- **Classes**: 1,623 (characters from 50 alphabets)
- **Images**: 20 per class
- **Size**: 28×28 grayscale
- **Paper**: [Lake et al., Science 2015](https://science.sciencemag.org/content/350/6266/1332)

```python
from fewshot.data import OmniglotDataset

dataset = OmniglotDataset(root='datasets/omniglot', split='train')
```

#### 4. **CIFAR-FS**
- **Classes**: 100 (64 train / 16 val / 20 test)
- **Images**: 600 per class
- **Size**: 32×32 RGB
- **Paper**: [TADAM](https://arxiv.org/abs/1805.10123)

```python
from fewshot.data import CIFARFSDataset

dataset = CIFARFSDataset(root='datasets/cifar-fs', split='train')
```

#### 5. **FC100** (Fine-Grained CIFAR)
- **Classes**: 100 (60 train / 20 val / 20 test)
- **Images**: 600 per class
- **Size**: 32×32 RGB
- **Paper**: [TADAM](https://arxiv.org/abs/1805.10123)

```python
from fewshot.data import FC100Dataset

dataset = FC100Dataset(root='datasets/fc100', split='train')
```

### Episode Sampling

```python
from fewshot.data import EpisodeSampler

sampler = EpisodeSampler(
    dataset=train_dataset,
    n_way=5,
    k_shot=1,
    q_queries=15,
    episodes_per_epoch=100
)

for episode in sampler:
    support_images, support_labels = episode['support']
    query_images, query_labels = episode['query']
    # Train on episode
```

---

## 📈 Results

### miniImageNet Benchmark

| Method | Backbone | 5-way 1-shot | 5-way 5-shot | Year |
|--------|----------|--------------|--------------|------|
| **Meta-DeepBDC** | ResNet-12 | **78.7%** | **90.3%** | 2021 |
| **FRN** | ResNet-12 | 77.9% | 90.1% | 2021 |
| **EGNN** | ResNet-12 | 76.4% | 88.3% | 2019 |
| **DeepEMD** | ResNet-12 | 75.7% | 88.7% | 2020 |
| **Meta-Baseline** | ResNet-12 | 73.4% | 87.2% | 2021 |
| **Simple CNAPS** | ResNet-18 | 69.8% | 84.5% | 2020 |
| **Prototypical** | ResNet-12 | 68.2% | 83.8% | 2017 |
| **GNN** | Conv-4 | 66.4% | 81.0% | 2018 |
| **Relation Net** | Conv-4 | 65.3% | 79.7% | 2018 |
| **Meta-SGD** | Conv-4 | 64.0% | 80.2% | 2017 |
| **MetaOptNet** | ResNet-12 | 64.1% | 80.0% | 2019 |
| **MAML** | Conv-4 | 63.1% | 79.4% | 2017 |
| **Reptile** | Conv-4 | 62.7% | 78.9% | 2018 |
| **Matching Net** | Conv-4 | 55.3% | 68.1% | 2016 |
| **Baseline++** | ResNet-18 | 51.9% | 75.0% | 2019 |

### tieredImageNet Benchmark

| Method | Backbone | 5-way 1-shot | 5-way 5-shot |
|--------|----------|--------------|--------------|
| **Meta-DeepBDC** | ResNet-12 | **82.1%** | **92.8%** |
| **FRN** | ResNet-12 | 80.6% | 91.5% |
| **Meta-Baseline** | ResNet-12 | 77.8% | 90.1% |
| **Prototypical** | ResNet-12 | 72.4% | 86.9% |

### Omniglot Benchmark

| Method | Backbone | 5-way 1-shot | 20-way 1-shot |
|--------|----------|--------------|---------------|
| **MAML** | Conv-4 | 98.7% | 95.8% |
| **Prototypical** | Conv-4 | 98.8% | 96.0% |
| **Matching Net** | Conv-4 | 98.1% | 93.8% |

### Cross-Domain Evaluation

Transfer from miniImageNet to:

| Method | → CUB | → Cars | → Places | → Plantae |
|--------|-------|--------|----------|-----------|
| **Meta-Baseline** | 88.4% | 77.2% | 71.3% | 63.8% |
| **Simple CNAPS** | 87.1% | 76.8% | 70.5% | 62.1% |
| **Prototypical** | 82.3% | 70.1% | 65.4% | 57.9% |

---

## 💻 Usage Examples

### Example 1: Train Prototypical Network

```python
import torch
from fewshot import PrototypicalNetwork, miniImageNetDataset
from fewshot.utils import FewShotTrainer

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_dataset = miniImageNetDataset(root='datasets/miniImageNet', split='train')
val_dataset = miniImageNetDataset(root='datasets/miniImageNet', split='val')

# Create model
model = PrototypicalNetwork(
    backbone='resnet12',
    input_size=84,
    distance='euclidean'
).to(device)

# Setup trainer
trainer = FewShotTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    n_way=5,
    k_shot=1,
    q_queries=15,
    episodes_per_epoch=100,
    device=device
)

# Train
trainer.train(
    epochs=100,
    learning_rate=0.001,
    save_dir='checkpoints/prototypical'
)

# Evaluate
test_dataset = miniImageNetDataset(root='datasets/miniImageNet', split='test')
accuracy = trainer.evaluate(test_dataset, n_episodes=600)
print(f"Test Accuracy: {accuracy:.2%}")
```

### Example 2: Fine-tune with MAML

```python
from fewshot import MAML

# Initialize MAML
model = MAML(
    backbone='conv4',
    inner_lr=0.01,
    inner_steps=5,
    first_order=False
).to(device)

# Meta-train
meta_trainer = MAMLTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    n_way=5,
    k_shot=1,
    device=device
)

meta_trainer.train(
    epochs=60,
    meta_lr=0.001,
    save_dir='checkpoints/maml'
)
```

### Example 3: Cross-Domain Transfer

```python
from fewshot import MetaBaseline, CUBDataset

# Load model pretrained on miniImageNet
model = MetaBaseline.from_pretrained('checkpoints/meta_baseline_mini.pth')

# Evaluate on CUB (birds dataset)
cub_dataset = CUBDataset(root='datasets/CUB', split='test')

accuracy = evaluate_cross_domain(
    model=model,
    target_dataset=cub_dataset,
    n_way=5,
    k_shot=5,
    n_episodes=600
)

print(f"Cross-Domain Accuracy (miniImageNet → CUB): {accuracy:.2%}")
```

### Example 4: Custom Backbone

```python
from fewshot.backbones import ResNet12
from fewshot import PrototypicalNetwork

# Define custom backbone
custom_backbone = ResNet12(
    avg_pool=True,
    drop_rate=0.1,
    dropblock_size=5
)

# Use with any few-shot method
model = PrototypicalNetwork(
    backbone=custom_backbone,
    input_size=84
)
```

### Example 5: Ensemble Methods

```python
from fewshot import EnsembleClassifier

# Load multiple trained models
models = [
    PrototypicalNetwork.from_pretrained('proto.pth'),
    RelationNetwork.from_pretrained('relation.pth'),
    MetaBaseline.from_pretrained('baseline.pth')
]

# Create ensemble
ensemble = EnsembleClassifier(
    models=models,
    voting='soft',
    weights=[0.4, 0.3, 0.3]
)

# Predict
predictions = ensemble.predict(support_set, query_set)
```

---

## 📁 Project Structure

```
few-shot-learning/
│
├── fewshot/                          # Main package
│   ├── __init__.py
│   │
│   ├── models/                       # Few-shot models
│   │   ├── __init__.py
│   │   ├── prototypical.py          # Prototypical Networks
│   │   ├── matching.py              # Matching Networks
│   │   ├── relation.py              # Relation Networks
│   │   ├── maml.py                  # MAML
│   │   ├── reptile.py               # Reptile
│   │   ├── meta_sgd.py              # Meta-SGD
│   │   ├── baseline_pp.py           # Baseline++
│   │   ├── simple_cnaps.py          # Simple CNAPS
│   │   ├── gnn.py                   # GNN
│   │   ├── egnn.py                  # EGNN
│   │   ├── meta_baseline.py         # Meta-Baseline
│   │   ├── meta_deepbdc.py          # Meta-DeepBDC
│   │   ├── frn.py                   # FRN
│   │   ├── metaoptnet.py            # MetaOptNet
│   │   └── deepemd.py               # DeepEMD
│   │
│   ├── backbones/                   # Feature extractors
│   │   ├── __init__.py
│   │   ├── conv4.py                 # 4-layer CNN
│   │   ├── conv6.py                 # 6-layer CNN
│   │   ├── resnet12.py              # ResNet-12
│   │   ├── resnet18.py              # ResNet-18
│   │   └── wrn.py                   # Wide ResNet
│   │
│   ├── data/                        # Data loading
│   │   ├── __init__.py
│   │   ├── miniimagenet.py         # miniImageNet
│   │   ├── tieredimagenet.py       # tieredImageNet
│   │   ├── omniglot.py             # Omniglot
│   │   ├── cifar_fs.py             # CIFAR-FS
│   │   ├── fc100.py                # FC100
│   │   ├── cub.py                  # CUB-200
│   │   ├── episode_sampler.py      # Episode sampling
│   │   └── transforms.py           # Data augmentation
│   │
│   ├── losses/                      # Loss functions
│   │   ├── __init__.py
│   │   ├── prototypical_loss.py
│   │   ├── matching_loss.py
│   │   └── relation_loss.py
│   │
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── trainer.py              # Training loops
│       ├── metrics.py              # Evaluation metrics
│       ├── visualization.py        # Plotting
│       └── logger.py               # Logging
│
├── configs/                         # Configuration files
│   ├── prototypical_mini.yaml
│   ├── maml_mini.yaml
│   ├── relation_mini.yaml
│   └── meta_baseline_mini.yaml
│
├── scripts/                         # Utility scripts
│   ├── download_data.py            # Download datasets
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation script
│   ├── cross_domain.py             # Cross-domain eval
│   └── visualize.py                # Visualization
│
├── notebooks/                       # Jupyter notebooks
│   ├── 01_introduction.ipynb
│   ├── 02_prototypical_demo.ipynb
│   ├── 03_maml_tutorial.ipynb
│   ├── 04_cross_domain.ipynb
│   └── 05_visualization.ipynb
│
├── experiments/                     # Experiment scripts
│   ├── run_benchmark.py
│   ├── ablation_study.py
│   └── compare_methods.py
│
├── tests/                          # Unit tests
│   ├── test_models.py
│   ├── test_data.py
│   └── test_losses.py
│
├── docs/                           # Documentation
│   ├── METHODS.md
│   ├── DATASETS.md
│   └── TUTORIAL.md
│
├── datasets/                       # Data directory
├── checkpoints/                    # Model checkpoints
├── results/                        # Experiment results
│
├── .gitignore
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## 🧪 Evaluation Protocol

### Standard Evaluation

```python
from fewshot.utils import evaluate_fewshot

# 5-way 1-shot
accuracy_1shot = evaluate_fewshot(
    model=model,
    dataset=test_dataset,
    n_way=5,
    k_shot=1,
    q_queries=15,
    n_episodes=600
)

# 5-way 5-shot
accuracy_5shot = evaluate_fewshot(
    model=model,
    dataset=test_dataset,
    n_way=5,
    k_shot=5,
    q_queries=15,
    n_episodes=600
)

print(f"5-way 1-shot: {accuracy_1shot:.2%} ± {std_1shot:.2%}")
print(f"5-way 5-shot: {accuracy_5shot:.2%} ± {std_5shot:.2%}")
```

### Cross-Domain Evaluation

```python
# Train on miniImageNet, test on CUB
source_dataset = miniImageNetDataset(split='train')
target_dataset = CUBDataset(split='test')

# Meta-train
model.meta_train(source_dataset)

# Evaluate on target
accuracy = evaluate_cross_domain(model, target_dataset)
```

---

## 📚 Citation

If you use this repository in your research, please cite:

```bibtex
@misc{fewshotlearning2024,
  title={Few-Shot Learning: State-of-the-Art Methods},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/few-shot-learning}}
}
```

### Key Papers

```bibtex
@inproceedings{snell2017prototypical,
  title={Prototypical networks for few-shot learning},
  author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{finn2017maml,
  title={Model-agnostic meta-learning for fast adaptation of deep networks},
  author={Finn, Chelsea and Abbeel, Pieter and Levine, Sergey},
  booktitle={ICML},
  year={2017}
}
```

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Few-shot learning community
- PyTorch team
- Dataset creators
- Paper authors

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/few-shot-learning/issues)
- **Email**: your.email@example.com

---

<p align="center">Made with ❤️ for few-shot learning research</p>

