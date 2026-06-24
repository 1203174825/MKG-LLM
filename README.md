# MKG-LLM: Multimodal Knowledge Graph-Constrained Large Language Model for Multi-Task Prediction of Flight Departure Delays 

## Project Overview

MKG-LLM (Multimodal Knowledge Graph-Constrained Large Language Model) is a novel framework for flight departure delay prediction. By constructing a multimodal knowledge graph (incorporating flight, airport, weather, and time information) and combining Graph Attention Networks (GAT) with a Large Language Model (Qwen2-1.5B + LoRA), it achieves end-to-end delay prediction.

### Core Innovations

- **Multimodal Knowledge Graph**: Fusing multi-source heterogeneous information including flight data, meteorological data, and airport network topology
- **Tri-modal Graph Attention Network**: Main GAT + Chain GAT + Network GAT fusion
- **Knowledge Graph-LLM Alignment**: Projecting KG features into LLM embedding space as soft prompts
- **Multi-Task Learning**: Simultaneously predicting delay classification (delay >= 15 minutes), delay duration (regression) and delay attribution
- **Parameter-Efficient Fine-Tuning**: Using LoRA technique, fine-tuning only ~10M parameters

## System Requirements

- **OS**: Windows 10/11
- **Python**: 3.10+
- **CUDA**: 12.1
- **GPU**: NVIDIA RTX 3090 or higher (24GB VRAM recommended)
- **RAM**: 32GB+ recommended
- **Disk Space**: 50GB+ (dataset + model weights + output)

## Installation Guide

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install DGL (CUDA 12.1)

```bash
pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/torch-2.5/cu121/repo.html
```

### 4. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download Model Weights

Ensure the following files exist:

- `LLM/qwen_weight/` - Qwen2-1.5B model weights (download from HuggingFace)

## Project Structure

```
MKG-LLM/
├── src/                              # Source code
│   ├── data/                         # Data processing
│   │   ├── aeolus_dataset.py        # Dataset loading and preprocessing
│   │   ├── kg_builder.py            # Knowledge graph construction
│   │   └── dgl_sampler.py           # DGL graph sampler
│   ├── models/                       # Model definitions
│   │   ├── stage1.py                # Stage 1: KG graph encoder
│   │   ├── stage2.py                # Stage 2: KG-LLM alignment layer
│   │   ├── gat.py                   # Graph attention network layer
│   │   ├── rgcn.py                  # Relational graph convolutional network layer
│   │   ├── feature_gate.py          # Feature gating mechanism
│   │   ├── gated_fusion.py          # Gated fusion module
│   │   ├── prediction_heads.py      # Prediction heads (classification + regression)
│   │   ├── alignment_injection.py   # Alignment injection module
│   │   ├── chain_encoder.py         # Chain encoder
│   │   └── task_attention.py        # Task attention module
│   ├── llm/                          # LLM-related
│   │   └── stage2.py                # Stage 2 LLM fine-tuning
│   ├── train/                        # Training scripts
│   │   ├── train_stage1.py          # Stage 1 training
│   │   ├── train_stage2.py          # Stage 2 training
│   │   ├── test_stage1.py           # Stage 1 testing
│   │   ├── test_stage1_cached.py    # Stage 1 cached testing
│   │   ├── test_stage2.py           # Stage 2 testing
│   │   ├── test_stage2_monthly.py   # Stage 2 monthly testing
│   │   ├── extract_kg_features.py   # KG feature extraction
│   │   └── trace_stage2_inference.py # Stage 2 inference tracing
│   └── utils/                        # Utility functions
│       ├── config.py                # Configuration parameters
│       └── metrics.py               # Evaluation metrics
├── UI/                               # User interface
│   ├── complete_server.py           # Flask server
│   └── compact_gui.html             # GUI interface
├── Output/                           # Output directory
│   └── src/                          # Model output
│       ├── stage1_best.pt           # Stage 1 best model
│       ├── stage2_best_val_auc/     # Stage 2 best model
│       ├── kg_features_cache/       # KG feature cache
│       └── normalizer.pkl           # Normalization parameters
├── Dataset/                          # Dataset directory
│   ├── Flight_Tabular/              # Tabular data
├── LLM/                              # LLM models
│   └── qwen_weight/                 # Qwen2-1.5B weights
├── requirements.txt                  # Dependency list
└── README.md                         # Documentation
```

## Usage

### Train Models

```bash
# Stage 1: Train KG graph encoder
python src/train/train_stage1.py

# Stage 2: Train KG-LLM alignment model
python src/train/train_stage2.py
```

### Test Models

```bash
# Stage 1 testing
python src/train/test_stage1.py

# Stage 2 testing (overall)
python src/train/test_stage2.py

# Stage 2 monthly testing (15th of each month in 2025)
python src/train/test_stage2_monthly.py
```

### Launch GUI

```bash
python UI/complete_server.py
```

Open in browser: http://127.0.0.1:5002

## Data Description

### Input Features (25-dim)

**20-dim Flight Features**:
1. CRS_DEP_TIME_MIN - Scheduled departure time (minutes)
2. CRS_ARR_TIME_MIN - Scheduled arrival time (minutes)
3. CRS_ELAPSED_TIME - Scheduled elapsed time
4. FLIGHTS - Flight count
5. PREV_DEP_DELAY - Predecessor flight departure delay
6. PREV_ARR_DELAY - Predecessor flight arrival delay
7. ORIGIN_CUM_DELAY_2H - Origin airport cumulative delay in past 2 hours
8. O_TEMP - Origin airport temperature
9. O_PRCP - Origin airport precipitation
10. O_WSPD - Origin airport wind speed
11. D_TEMP - Destination airport temperature
12. D_PRCP - Destination airport precipitation
13. D_WSPD - Destination airport wind speed
14. O_LATITUDE - Origin airport latitude
15. O_LONGITUDE - Origin airport longitude
16. D_LATITUDE - Destination airport latitude
17. D_LONGITUDE - Destination airport longitude
18. O_DAY_FLIGHTS - Origin airport daily flight count
19. D_DAY_FLIGHTS - Destination airport daily flight count
20. IS_PEAK_HOUR - Whether peak hour

**5-dim Airport Features**:
21. dep_cnt normalized - Normalized departure flight count
22. arr_cnt normalized - Normalized arrival flight count
23. peak_ratio - Peak hour departure ratio
24. dest_div normalized - Destination diversity
25. avg_crs_dep - Average scheduled departure time

## Model Architecture

### Stage 1: KG Graph Encoder

- **Tri-modal GAT Fusion**: Main GAT + Chain GAT + Network GAT
- **Shared RGAT**: 25-dim input → 128-dim hidden → 384-dim output (2 layers, 4 attention heads, 8 edge types)
- **Feature Gate**: 384-dim feature gating
- **Prediction Heads**: Classification head (delay classification) + Regression head (delay duration)
- **Parameters**: ~4.54M

### Stage 2: KG-LLM Alignment

- **KGAlignmentLayer**: 384-dim → 1536×5-dim (5 special tokens)
- **ModifiedEmbedding**: Inject KG features into LLM input embeddings
- **Qwen2-1.5B + LoRA**: r=8, alpha=16, target modules q_proj and v_proj
- **Trainable Parameters**: ~10.34M

**Total Parameters**: ~14.88M

## Performance Metrics

### Classification Metrics
- **AUC**: Area Under the Curve
- **Accuracy**: Accuracy
- **F1 Score**: F1 Score

### Regression Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **R²**: Coefficient of Determination

### Statistical Testing
- **t-test**: p<0.05 significance test

## FAQ

### Q: DGL installation failed?
A: Use the special installation command:
```bash
pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/torch-2.5/cu121/repo.html
```

### Q: CUDA version mismatch?
A: Ensure PyTorch and DGL versions are compatible with CUDA 12.1.

### Q: Model loading failed?
A: Check that `Output/src/stage1_best.pt` and `normalizer.pkl` exist.

### Q: GUI won't start?
A: Ensure Flask is installed and port 5002 is not in use.

### Q: Out of memory during training?
A: Reduce batch_size or enable gradient accumulation.

## References

- BTS On-Time Performance: https://transtats.bts.gov/PREZIP/
- Meteostat Weather Data: https://dev.meteostat.net/
- Qwen2 Model: https://huggingface.co/Qwen
- DGL Documentation: https://docs.dgl.ai/

## License

This project is for academic research purposes only.

## Contact

For questions, please contact the project maintainer.

---

**Last Updated**: June 2026
