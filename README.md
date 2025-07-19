# Rhyme Probe Experiments with Sparse Autoencoders

A comprehensive research project investigating rhyming behavior in large language models using sparse autoencoders (SAEs) for causal probing.

## 🎯 Project Overview

This project implements rhyme probe experiments to identify and causally probe rhyming behavior in language models like Gemma. The research uses sparse autoencoders to discover interpretable features that encode rhyming patterns and enables causal interventions.

## 📁 Project Structure

```
Gemma/
├── src/                          # Source code
│   ├── data/                     # Dataset creation and processing
│   ├── metrics/                  # Rhyme detection and analysis
│   ├── models/                   # Model implementations
│   ├── experiments/              # Experiment runners
│   └── utils/                    # Utility functions
├── data/                         # Datasets and analysis
│   ├── sample_*.json            # Sample datasets (included)
│   └── analysis/                # Quality reports
├── configs/                      # Configuration files
├── scripts/                      # Deployment and setup scripts
└── docs/                         # Documentation
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd Gemma

# Install dependencies
pip install -r requirements.txt

# Set up Hugging Face authentication
python setup_hf_auth.py
```

### 2. Generate Datasets

```bash
# Create high-quality rhyming and non-rhyming datasets
python create_final_datasets.py

# Test dataset quality
python test_dataset_quality.py
```

### 3. Run Experiments

```bash
# Local testing
python run_experiments.py --model gemma-2b --local

# GCP deployment
bash create_gcp_instance.sh
bash deploy_to_gcp.sh
python run_experiments.py --model gemma-7b
```

## 📊 Dataset Generation

The project includes sophisticated dataset generation with:

- **Rhyming datasets**: AABB, ABAB, ABBA, AAAA patterns
- **Non-rhyming datasets**: Carefully crafted to avoid rhymes
- **Quality validation**: Automated rhyme detection and verification
- **Multiple topics**: Nature, emotions, abstract concepts
- **Various styles**: Nature poems, emotional poems, abstract poems

### Dataset Format

```json
{
  "text": "The tree stands tall and bright\nWhile bird sings in the light",
  "lines": ["The tree stands tall and bright", "While bird sings in the light"],
  "rhyme_pattern": "AABB",
  "rhyme_type": "perfect",
  "rhyme_density": 0.025,
  "features": {
    "rhyme_density": 0.025,
    "end_rhyme_ratio": 0.5,
    "internal_rhyme_ratio": 0.0
  },
  "metadata": {
    "topic": "nature",
    "style": "nature_poem",
    "num_lines": 4
  }
}
```

## 🔬 Experiment Pipeline

### Phase 1: Feature Discovery
- Train sparse autoencoders on model activations
- Identify features that encode rhyming behavior
- Analyze feature interpretability

### Phase 2: Causal Probing
- Perform causal interventions on discovered features
- Measure impact on rhyming behavior
- Validate causal relationships

### Phase 3: Analysis
- Statistical analysis of results
- Visualization of feature activations
- Comparison across different models

## 🛠️ Key Components

### Rhyme Detection (`src/metrics/rhyme_metrics.py`)
- Advanced rhyme pattern detection
- Multiple rhyme types (perfect, slant, assonance)
- Rhyme density calculation

### Dataset Creation (`src/data/dataset_creation.py`)
- Template-based generation
- Quality verification
- Multiple rhyme patterns and styles

### Experiment Runner (`run_experiments.py`)
- End-to-end experiment orchestration
- Model loading and inference
- Result collection and analysis

## 📈 Quality Metrics

The project includes comprehensive quality testing:

- **Rhyme density separation**: Ensures clear distinction between rhyming/non-rhyming
- **Pattern verification**: Validates correct rhyme patterns
- **Topic balance**: Checks for balanced topic distribution
- **Manual verification**: Human-readable sample validation

## 🚀 Deployment

### Local Development
```bash
# Test with smaller models
python run_experiments.py --model gemma-2b --local --samples 100
```

### GCP GPU Deployment
```bash
# Launch GPU instance
bash create_gcp_instance.sh

# Deploy code
bash deploy_to_gcp.sh

# Run experiments
python run_experiments.py --model gemma-7b --gpu
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-compatible GPU (for large models)
- Hugging Face account with Gemma access

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model configurations
- Experiment parameters
- Dataset settings
- Output paths

## 📊 Results

Results are saved in:
- `outputs/`: Experiment outputs
- `results/`: Analysis results
- `logs/`: Experiment logs
- `data/analysis/`: Quality reports

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for research purposes. Please respect the licensing terms of the models used (Gemma, etc.).

## 🙏 Acknowledgments

- Google for the Gemma models
- Hugging Face for the transformers library
- The sparse autoencoder research community

## 📞 Contact

For questions or collaboration, please open an issue or contact the maintainers.

---

**Note**: This project requires Hugging Face authentication for accessing gated models like Gemma. See `setup_hf_auth.py` for setup instructions. 