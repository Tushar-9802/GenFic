
# GenFic

Production-grade parameter-efficient fine-tuning toolkit for creative text generation.

## Project Aims

GenFic provides an end-to-end pipeline for training and deploying custom language models specialized in creative fiction writing. The system focuses on:

- High-quality narrative generation with consistent style and tone
- Efficient fine-tuning on consumer hardware using LoRA/QLoRA
- Automated quality control and evaluation frameworks
- Multi-adapter management for different writing styles
- Production-ready inference with post-processing pipelines

## Target Models

Primary models for fine-tuning:

**Mistral 3.2 3B**

- Lightweight option for rapid iteration
- Lower VRAM requirements (8-12GB)
- Faster training and inference
- Suitable for style transfer tasks

**Mistral 7B Instruct**

- Primary production model
- Superior narrative coherence
- Better character consistency
- Optimal balance of quality and efficiency

Both models support 4-bit quantization for deployment on consumer GPUs (16GB VRAM).

## Project Structure

```
genfic/
├── src/genfic/              # Core package
│   ├── training/            # LoRA training pipeline
│   ├── inference/           # Generation and post-processing
│   ├── data/                # Dataset preprocessing and validation
│   ├── models/              # Model loading and adapter management
│   ├── evaluation/          # Quality metrics and scoring
│   ├── cli/                 # Command-line interface
│   └── utils/               # Shared utilities
├── configs/                 # YAML configuration files
│   ├── training/            # Training configurations
│   ├── inference/           # Generation parameters
│   ├── models/              # Model specifications
│   └── data/                # Data pipeline configs
├── data/                    # Dataset storage (gitignored)
│   ├── raw/                 # Source material
│   ├── processed/           # Cleaned and formatted data
│   ├── train/               # Training split
│   ├── val/                 # Validation split
│   └── test/                # Test split
├── models/                  # Model storage (gitignored)
│   ├── base/                # Base model weights
│   ├── adapters/            # Trained LoRA adapters
│   └── checkpoints/         # Training checkpoints
├── outputs/                 # Generation outputs (gitignored)
│   ├── generations/         # Generated text samples
│   ├── logs/                # Training and inference logs
│   └── metrics/             # Evaluation results
├── scripts/                 # Automation scripts
├── tests/                   # Test suite
├── docs/                    # Documentation
└── notebooks/               # Experimentation notebooks
```

## Planned Development Steps

### Phase 1: Data Pipeline

1. Dataset preprocessing and validation system
2. Metadata extraction and tagging framework
3. Quality filtering and consistency checks
4. Train/validation/test split generation
5. Instruction-response formatting pipeline

### Phase 2: Training Infrastructure

1. LoRA configuration and hyperparameter management
2. QLoRA training loop with gradient accumulation
3. Checkpoint management and versioning
4. Loss tracking and validation evaluation
5. Multi-adapter training support

### Phase 3: Inference Engine

1. Context-aware prompt assembly system
2. Generation parameter optimization
3. Post-processing and cleanup pipeline
4. Repetition detection and mitigation
5. Quality scoring and filtering

### Phase 4: Evaluation Framework

1. Automated quality metrics (consistency, coherence)
2. Human evaluation tooling
3. A/B testing infrastructure
4. Performance benchmarking
5. Feedback loop integration

### Phase 5: Production Deployment

1. CLI interface implementation
2. Batch generation pipeline
3. Multi-adapter workflow management
4. Performance optimization
5. Documentation and examples

## Technical Approach

**Fine-Tuning Method**: QLoRA (4-bit quantization + LoRA adapters)

- Target modules: q_proj, v_proj, k_proj, o_proj
- Rank (r): 16-32
- Alpha: 32-64
- Dropout: 0.05-0.10

**Training Strategy**:

- Effective batch size: 16-32 (via gradient accumulation)
- Learning rate: 1e-4 to 3e-4 with cosine decay
- Warmup: 5-10% of total steps
- Epochs: 3-5 with early stopping

**Inference Optimization**:

- Temperature: 0.7-0.85 for balanced creativity
- Top-p: 0.90-0.95 nucleus sampling
- Repetition penalty: 1.15-1.25
- Context management via sliding windows

**Quality Control**:

- POV/tense consistency validation
- Character name and trait verification
- Repetition pattern detection
- Lexical diversity scoring
- Human-in-loop refinement workflow

## Hardware Requirements

**Minimum**:

- GPU: 12GB VRAM
- RAM: 16GB
- Storage: 30GB

**Recommended**:

- GPU: 16GB VRAM (RTX 4080, RTX 5070 Ti)
- RAM: 32GB
- Storage: 100GB SSD

## License

MIT License - See LICENSE file for details.
