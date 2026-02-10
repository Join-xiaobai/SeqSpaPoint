# PointCloud-DTA/DTI/MOA Prediction Project
SeqSpaPoint: A multitask framework based on sequence-derived point clouds and cross-modal attention

## Project Overview
This project is a deep learning-based multitask prediction framework for bioinformatics, core supporting three key tasks: **Drug-Target Affinity prediction (DTA)**, **Drug-Target Interaction prediction (DTI)**, and **Mechanism of Action prediction (MOA)**. It innovatively integrates point cloud geometric features with sequence embedding features, adapts to three practical application scenarios (warm start, drug cold start, target cold start), and achieves "zero code modification" for task and dataset switching through a single entry script and centralized configuration, balancing usability and generalization ability.

## Core Features
- **Unified Multitask Framework**: A single script supports three major tasks, sharing preprocessing and training pipelines to reduce development and maintenance costs
- **Innovative Point Cloud Features**: Constructs 3D point cloud representations of drugs/targets to capture spatial structural information and improve prediction accuracy
- **Full-Scenario Adaptation**: Supports warm start, drug cold start, and target cold start to cover prediction needs for new drugs/new targets
- **High Robustness Design**: Integrates Focal Loss, dynamic λ scheduling, numerical stability and other mechanisms to alleviate class imbalance and training oscillations
- **Reproducibility Guarantee**: Fixed random seeds, unified configuration parameters, and complete log records ensure accurate reproducibility of experimental results

## Project Structure
```
point_cloud_model/
├── ablation_experiment/     # Ablation experiment module
│   ├── ablation_model/      # 4 groups of ablation models (variants with removed point clouds/features)
│   │   ├── model_seqspa_point_nopoint.py      # Remove point cloud features
│   │   ├── model_seqspa_point_nofeature.py   # Remove normal vector features
│   │   ├── model_seqspa_point_randomFeature.py # Random feature vectors
│   │   └── model_seqspa_point_randomPoint.py  # Random noise point clouds
│   ├── results/             # Ablation experiment result logs
│   └── visualization_of_ablation_experiment*.py # Attention visualization scripts (4 corresponding experiments)
├── case_study/          # MTC clinical case analysis module
│   ├── chembl29/                # Original dataset
│   ├── MTC/                # Case dataset (DAVIS subset)
│   ├── pdbbind/                # Original dataset
│   ├── UniProt/                # Original dataset
│   ├── step1_build_mtc_dta_benchmark.py # Benchmark dataset construction
│   ├── step2.1_tuning_training_MTC.py # Model fine-tuning training
│   ├── step2.2_mtc_finetuned_predict.py # Model prediction
│   └── step3_analysis_results.py # Result analysis and visualization
├── data/                    # Original dataset directory (to be stored by users)
│   ├── dta/                 # DTA datasets (davis, kiba)
│   ├── dti/                 # DTI datasets (hetionet, yamanishi_08)
│   └── moa/                 # MOA datasets (inhibition, activation)
├── data_preprocessing/      # Data preprocessing module
│   └── point_cloud_coordinate_construction.py  # Point cloud construction + embedding generation
│   └── model/               # Sequence generation pre-trained models
├── dataset/                   # Utility function module (added detailed description)
│   └── interaction_dataset.py      # Data processing utilities: feature normalization, point cloud sampling, embedding vector loading
├── evalMetrics/                   # Data validation and evaluation module
│   ├── classification_metrics.py      # Evaluation metric utilities: DTI/MOA (AUPRC/AUROC/F1) calculation
│   ├── eval_utils.py       # Model evaluation: evaluation testing
│   └── visualization_utils.py # Evaluation metric utilities: DTA (CI/Pearson/RMSE) calculation
├── model/                   # Core model
│   └── SeqSpaPoint.py      # SeqSpaPoint model definition module
├── result/                   # Automatically generated training output directory
│   ├── dta_experiment.py         # DTA task results (models, logs, metrics)
│   ├── dti_experiment.py         # DTI task results
│   └── moa_experiment.py         # MOA task results
├── train/                   # Task training module
│   ├── train_dta.py         # DTA task training logic (regression)
│   ├── train_dti.py         # DTI task training logic (classification)
│   └── train_moa.py         # MOA task training logic (classification)
├── utils/                   # Utility function module
│   └── common_utils.py      # Processing utilities
├── visualization/           # Result visualization scripts (generate FigureA-G)
├── main.py                  # Unified main entry script (core for task switching)
└── requirements.txt         # Dependencies list
```

## Environment Configuration
### Dependency Installation
Python 3.8-3.10 is recommended. Create a virtual environment with conda and install dependencies (see `requirements.txt` for details):
```bash
conda create -n pointcloud-bio python=3.9
conda activate pointcloud-bio
pip install -r requirements.txt
```

### Core Dependency Description
| Category | Key Dependencies | Version Requirements |
|----------|------------------|----------------------|
| Deep Learning Framework | PyTorch | ≥1.12.0 |
| 3D Point Cloud Processing | Open3D, plyfile | Open3D≥0.15.0 |
| Data Processing | NumPy, Pandas, SciPy | NumPy≥1.24.3 |
| Model Evaluation | scikit-learn | ≥1.2.2 |
| Sequence Embedding | Transformers | ≥4.30.2 |
| Utilities | tqdm, json5 | - |

## Quick Start
### 1. Data Preparation
(1) Create dataset folders by task under the `data/` directory with the following example structure:
```
data/
└── dta/
    └── davis/  # Dataset name must match configuration
        └── related datasets  # Original data must include ID columns, sequence columns, label columns
```
(2) Download ESM-2 t33_650M_UR50D (target embedding model) and ChemBERTa (drug embedding model) files, and save them to the `data_preprocessing/models/` folder (download addresses and related instructions can be found in `data_preprocessing/models/models_readme.md`; core necessary parts are briefly introduced below):

#### I. ESM-2 t33_650M_UR50D (Target Embedding Model)
##### 1. Storage Path
`data_preprocessing/models/esm2_t33_650M_UR50D/`

##### 2. Full List of Files to Store and Their Functions (Download link: https://huggingface.co/facebook/esm2_t33_650M_UR50D/tree/main)
| File Name                | Core Function                     |
|--------------------------|-----------------------------------|
| config.json              | Defines model structure (layers, dimensions, etc.) |
| model.safetensors        | Safe storage format file for model weights |
| pytorch_model.bin        | PyTorch-format model weight file (generates target embeddings) |
| README.md                | Model description document        |
| special_tokens_map.json  | Special token mapping configuration file |
| tf_model.h5              | TensorFlow-format model weight file |
| tokenizer_config.json    | Protein sequence encoding rule configuration |
| vocab.txt                | Amino acid symbol vocabulary      |

#### II. ChemBERTa (Drug Embedding Model)
##### 1. Storage Path
`data_preprocessing/models/chemberta/`

##### 2. Full List of Files to Store and Their Functions (Download link: https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/tree/main)
| File Name                | Core Function                     |
|--------------------------|-----------------------------------|
| added_tokens.json        | Added token configuration file    |
| config.json              | Defines model structure (layers, dimensions, etc.) |
| merges.txt               | Tokenizer subword merging rule file |
| pytorch_model.bin        | PyTorch-format model weight file (generates drug embeddings) |
| special_tokens_map.json  | Special token mapping configuration file |
| tokenizer_config.json    | SMILES sequence encoding rule configuration |
| tokenizer.json           | Complete Tokenizer configuration file |
| training_args.bin        | Training parameter storage file   |
| vocab.json               | Chemical symbol vocabulary        |

Original Data Requirements:
- DTA: Must include `drug_id`, `protein_id`, `affinity` (affinity value)
- DTI: Must include `drug_id`, `protein_id`, `label` (0/1)
- MOA: Must include `DrugID`, `TargetID`, `label` (0/1)

### 2. Task Configuration
Modify core configurations in `main.py` (no need to change other code):
```python
# Step 1: Select task (dta/dti/moa)
TASK_NAME = "dta"  # Task switching only requires modifying this line

# Step 2: Configure dataset and hyperparameters (optimal values preset)
CONFIG = {
    "dta": {
        "dataset_name": "davis",  # Optional: davis/kiba
        "label_col": "affinity",
        "id_cols": ["drug_id", "protein_id"],
        "training_args": {
            "split_types": ["warm", "drug_cold", "target_cold"],  # Scenarios to enable
            "num_folds": 5,  # Number of cross-validation folds
            "num_epochs": 300,
            "batch_size": 128,
            "lr": 1e-4,
            # Other hyperparameters optimized for datasets, no modification needed
        }
    }
    # DTI/MOA configurations follow the same logic with optimal parameters preset
}
```

### 3. Run the Project
Execute the main script directly to automatically complete preprocessing and training:
```bash
python main.py
```
Automatic script workflow:
1. Detect and generate drug/target embeddings (automatically run if missing)
2. Construct feature CSV and normalized point cloud PLY files
3. Start cross-validation training according to configuration
4. Results are automatically saved to the `./result/` directory

### 4. Result Viewing
Training output directory structure (taking DTA-davis as an example):
```
result/dta_experiment/davis/warm/
├── models/          # Optimal model weights (saved by fold)
├── train_log.txt    # Complete training log (including per-epoch metrics)
└── metrics.csv      # Final evaluation metric summary
```
Core Metric Description:
- DTA: CI, Pearson, RMSE, MAE, R²
- DTI/MOA: AUPRC, AUROC, F1-score, Accuracy

## Ablation Experiment Description
Four ablation models are provided to verify the effectiveness of core modules:
- `model_seqspa_point_nopoint.py`: Removes point cloud features, retains only sequence features
- `model_seqspa_point_nofeature.py`: Removes additional features, retains only point cloud coordinates
- `model_seqspa_point_randomFeature.py`: Uses randomly generated feature vectors
- `model_seqspa_point_randomPoint.py`: Uses random noise point cloud coordinates

Experimental results are saved in `ablation_experiment/results/`, supporting direct comparison of module contribution.

## Notes
1. Data paths must strictly follow the `data/{task}/{dataset}/` format, otherwise preprocessing will fail
2. The first run will generate point cloud and embedding files (time-consuming), subsequent runs will automatically skip this step
3. To reproduce experiments, keep `seed=42` (fixed) to avoid the impact of random factors on results
4. Automatic CPU/GPU adaptation is supported; GPU environments require installing the corresponding version of PyTorch (CUDA≥11.3)

## Frequently Asked Questions
- Preprocessing failure: Check if the original data format meets requirements and if ID/label column names match CONFIG settings
- Training oscillations: Reduce learning rate to 5e-5 or enable `use_numerical_stability=True`
- Poor metrics: Confirm whether the correct scenario is selected (cold start requires dedicated hyperparameters)
- Out-of-memory errors: Reduce `batch_size` to 64 or decrease point cloud sampling points (adjust in preprocessing module)
