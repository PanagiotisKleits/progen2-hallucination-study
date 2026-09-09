# ProGen2 Hallucination Study

A systematic study of hallucination behaviour in [ProGen2](https://github.com/enijkamp/progen2),
Salesforce's protein language model. This project generates synthetic proteins across a grid of
sampling parameters (temperature × top_p) and length distributions derived from a real proteome,
enabling quantitative analysis of how generation settings affect sequence quality and diversity.

## Overview

Protein language models like ProGen2 can produce sequences that are syntactically valid but
biologically implausible — a phenomenon analogous to hallucination in text models. This study
investigates how two key sampling hyperparameters (temperature and nucleus sampling probability
`top_p`) influence the degree of hallucination in generated sequences, using `progen2-small` as
the base model.

Generation is **unconditional**: the only input is the start token, with no prompt, no prefix and
no control tags. Temperature and top_p are the experimental axis, not guidance — they change how
the model samples from its own distribution, not what it is asked to produce.

The generation is length-aware: protein lengths and their sampling counts are derived from a real
length distribution, so the study reflects realistic protein size profiles rather than arbitrary
fixed lengths.

## Reference dataset

| Property | Value |
|---|---|
| Proteins | 82,129,039 |
| Median length | 304 aa |
| Mean length | 386.9 aa |
| Maximum length | 45,354 aa |

ProGen2 has a hard context limit of 1024 tokens, so the longest sequence it can produce is 1023
amino acids. Proteins above that limit are outside the model's reach by construction — an
architectural constraint, measured separately from hallucination.

## Input Files

### `progen2_params.tsv`

A tab-separated file defining the sampling parameter grid. Each row is one (temperature, top_p)
combination:

```
temperature	top_p
0.2	0.90
0.4	0.90
0.6	0.80
0.6	0.95
0.8	0.85
0.8	0.95
1.0	0.90
1.0	0.95
1.2	0.90
1.4	0.95
```

**Temperature** controls randomness: low values (0.2) produce repetitive, degenerate sequences;
high values (1.4) produce more diverse output. **top_p** (nucleus sampling) restricts sampling to
the most probable tokens whose cumulative probability reaches `p`.

### `protein_lengths.tsv`

A tab-separated file with two columns: protein length (in amino acids) and its count in the
reference proteome:

```
length	count
50	12043
51	11876
52	12310
```

The script samples `count // 10` proteins per length per parameter combination. With ten (t, p)
combinations in the grid, the ten runs together reconstruct a synthetic proteome of roughly the
same size and the same length distribution as the real one.

## Usage

### Requirements

- Python 3.8+
- [ProGen2](https://github.com/enijkamp/progen2) cloned locally, with `progen2-small` checkpoints
- PyTorch
- HuggingFace `transformers`
- HuggingFace `tokenizers`

ProGen2 is not a pip package — the model class lives inside the cloned repository, which sits at a
different path on each machine. `--progen2-dir` and `--checkpoints` must always be passed
explicitly.

### Basic Usage

**CPU (multi-core machine):**
```bash
python run_progen2_v3.py --device cpu \
    --progen2-dir /path/to/progen2 \
    --checkpoints /path/to/checkpoints/progen2-small
```

**Single GPU:**
```bash
python run_progen2_v3.py --device cuda \
    --batch-size 100 \
    --progen2-dir /path/to/progen2 \
    --checkpoints /path/to/checkpoints/progen2-small
```

Note that `--device cuda` uses **one** GPU (`cuda:0`), even on a multi-GPU machine. See
[Status](#status).

### All Arguments

| Argument | Default | Description |
|---|---|---|
| `--device` | `cpu` | `cpu` or `cuda` (single GPU) |
| `--params-tsv` | `progen2_params.tsv` | Path to sampling parameter grid |
| `--lengths-tsv` | `protein_lengths.tsv` | Path to length distribution file |
| `--output-dir` | `progen2_outputs` | Directory for output files |
| `--batch-size` | `None` | Sequences per `generate()` call. Defaults to all samples at once |
| `--repetition-penalty` | `1.0` | Penalise repeated tokens (1.0 = no penalty) |
| `--seed` | `42` | Random seed for reproducibility |
| `--progen2-dir` | `/path/to/progen2` | Path to ProGen2 source directory |
| `--checkpoints` | `/path/to/progen2-small` | Path to model checkpoint directory |

**Always set `--batch-size` for large runs.** The default generates every sample for a given length
in a single batch. On real data a common length may have hundreds of thousands of proteins, which
exhausts memory immediately. Memory scales as `batch_size × length × ~96 KB`.

## Output

Inside `--output-dir`:

- **`t{temperature}_p{top_p}.fasta`** — one file per parameter combination, containing sequences
  from all lengths for that combination. Sequence IDs follow the pattern
  `t{t}_p{p}_len{length}_seq{index}`.
- **`progress.txt`** — one completed `(t, p, length)` identifier per line.

**Checkpointing:** the script reads `progress.txt` on start and skips any combination already
recorded, so an interrupted run resumes instead of starting over. Use a fresh output directory
whenever the generation logic changes, otherwise results from different code versions are mixed.

## Implementation Details

**Compatibility patch:** ProGen2 predates newer versions of HuggingFace `transformers`, which
removed `GenerationMixin` from `PreTrainedModel`'s bases and left the model class without
`.generate()`. The script re-attaches the mixin at runtime, without modifying the ProGen2
repository.

**Start and end tokens:** ProGen2 was trained on every sequence in both directions, marked by a
leading `1` (N→C, the natural reading direction) or `2` (C→N, reversed). Generation starts from
`1` and stops at `2`, producing normally oriented proteins.

**Sequence cleaning:** generated tokens are filtered to retain only the 20 standard amino acids
(`ACDEFGHIKLMNPQRSTVWY`). Non-standard characters (X, B, Z, U) and the direction tokens `1` and `2`
are removed. The tokenizer does not flag `1`/`2` as special tokens, so `skip_special_tokens` alone
does not remove them.

**Length control:** generation uses `max_new_tokens`, which is an upper bound rather than a target.
See [Status](#status) for what this means in practice.

## Status

**Working:** CPU and single-GPU generation, batching, checkpointing, reproducible seeding,
per-parameter FASTA output.

**Not implemented:** comparison metrics between generated and real proteins.
