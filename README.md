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
### All Arguments

| Argument | Default | Description |

| `--device` | `cpu` | `cpu` or `cuda` |

| `--multi-gpu` | `False` | Enable DataParallel across all available GPUs |

| `--params-tsv` | `progen2_params.tsv` | Path to sampling parameter grid |

| `--lengths-tsv` | `protein_lengths.tsv` | Path to length distribution file |

| `--output-dir` | `progen2_outputs2` | Directory for output FASTA file |

| `--repetition-penalty` | `1.0` | Penalise repeated tokens (1.0 = no penalty) |

| `--progen2-dir` | `/path/to/progen2` | Path to ProGen2 source directory |

| `--checkpoints` | `/path/to/progen2-small` | Path to model checkpoint directory |

## Output

All generated sequences are saved to a single FASTA file (`all_sequences.fasta`) inside the output directory.

## Implementation Details

**Compatibility patch:** ProGen2 predates newer versions of HuggingFace `transformers`. The script automatically patches `ProGenForCausalLM` to inherit from `GenerationMixin` if needed, ensuring compatibility with `transformers >= 4.50`.

**Sequence cleaning:** Generated tokens are filtered to retain only the 20 standard amino acids (`ACDEFGHIKLMNPQRSTVWY`). Non-standard characters (X, B, Z, U) and tokenizer artefacts (start token `1`, end token `2`) are removed before saving.

**Length control:** Generation uses `max_new_tokens` to control output length. ProGen2 uses `1` as a start token and `2` as an end token.

**Multi-GPU:** When `--multi-gpu` is set and multiple GPUs are available, PyTorch `DataParallel` is used to distribute the batch across all GPUs.

## Notes

- This study uses `progen2-small`. The same script can be adapted for larger ProGen2 variants by changing `--checkpoints`.
- At low temperatures (e.g. 0.2), the model tends to produce degenerate sequences dominated by a single repeated amino acid (e.g. `MSSSSSSS...`). This behaviour is itself a hallucination signal and is retained in the output for analysis.
- The `--repetition-penalty` argument can reduce repetitive outputs, but values above ~1.3 may cause the model to produce very short or empty sequences.

