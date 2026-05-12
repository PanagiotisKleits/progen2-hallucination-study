# ProGen2 Hallucination Study

A systematic study of hallucination behaviour in [ProGen2](https://github.com/salesforce/progen), Salesforce's protein language model. This project generates
synthetic proteins across a grid of sampling parameters (temperature × top_p) and length distributions derived from a real proteome, enabling quantitative analysis
of how generation settings affect sequence quality and diversity.

## Overview

Protein language models like ProGen2 can produce sequences that are syntactically valid but biologically implausible — a phenomenon analogous to hallucination in
text models. This study investigates how two key sampling hyperparameters (temperature and nucleus sampling probability `top_p`) influence the degree of
hallucination in generated sequences, using `progen2-small` as the base model.

The generation is length-aware: protein lengths and their sampling counts are derived from a real length distribution, so the study reflects realistic protein size
profiles rather than arbitrary fixed lengths.

## Input Files

### `progen2_params.tsv`

A tab-separated file defining the sampling parameter grid. Each row is one (temperature, top_p) combination:

**Temperature** controls randomness: low values (0.2) produce repetitive, degenerate sequences; high values (1.4) produce more diverse output. **top_p** (nucleus
sampling) restricts sampling to the most probable tokens whose cumulative probability reaches `p`.

### `protein_lengths.tsv`

A tab-separated file with two columns: protein length (in amino acids) and its count in the reference proteome. The script samples `count // 10` proteins per
length per parameter combination:

Lengths greater than 1023 are automatically skipped, as ProGen2 has a hard token limit of 1024.

## Usage

### Requirements

- Python 3.8+
- [ProGen2](https://github.com/salesforce/progen) installed and accessible
- PyTorch
- HuggingFace `transformers`
- HuggingFace `tokenizers`

### Basic Usage

**CPU (multi-core VM):**
```bash
python run_progen2_v3.py --device cpu
```

**Single GPU:**
```bash
python run_progen2_v3.py --device cuda
```

**Multi-GPU (DataParallel):**
```bash
python run_progen2_v3.py --device cuda --multi-gpu
```
