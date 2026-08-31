# dsSTICI

**Dataset-specific Split-Transformer with Integrated Convolutions for sparse scRNA-seq cell-variant imputation**

Code accompanying the manuscript:

> **De novo transformer modeling improves recovery of genetic cell types from sparse single-cell RNA sequencing**

## Overview

Single-cell RNA sequencing (scRNA-seq) can provide both expression measurements and genetic variant calls for individual cells, but the resulting cell-variant (CV) matrices are extremely sparse and contain false-positive and false-negative base calls. **dsSTICI** adapts the STICI Split-Transformer with Integrated Convolutions framework to this setting by training a separate model **de novo for each sparse CV matrix (sequence alignment)**. The trained dataset-specific model reconstructs A/T/G/C probabilities at every site and can be used to produce a substantially denser matrix for downstream phylogenetic/genetic-type analyses.

Upstream STICI resources:

- STICI paper: https://www.nature.com/articles/s41467-025-56273-3
- STICI source code: https://github.com/shilab/STICI

This repository is intended to make the dsSTICI implementation self-contained. It should not be necessary to refer to unpublished changes to the upstream STICI code in order to understand the architecture, masking procedure, loss, or training parameters used here.

---

## What changed from STICI?

The distinction between **architecture changes** and **training/application changes** is important.

### Architecture-level changes

The overall STICI design is retained: learned categorical and positional embeddings, overlapping local chunks, multi-head attention, integrated multi-kernel 1D convolution blocks, cross-attention, residual connections, and a convolutional output head.

However, dsSTICI is **not an exact copy of the public STICI architecture**. The principal architecture-level change is in the self-attention block:

| Component | Public STICI | dsSTICI |
|---|---|---|
| Input representation | genotype/imputation representation used by STICI | 5-state nucleotide input: `A`, `T`, `G`, `C`, `?` |
| Prediction head | genotype probabilities | 4-state base probabilities: `A`, `T`, `G`, `C` |
| Loss functions | Categorical cross-entropy + KLD + Mach-Rsq | Categorical cross-entropy + KLD |

The convolutional and cross-attention scaffold is otherwise closely retained from STICI. Parameter defaults such as chunk overlap and sites per saved model are also different in dsSTICI, but those are **hyperparameter changes rather than topology changes**.

### Training/application changes

The main methodological changes for the scRNA-seq application are:

1. **One model is trained de novo per CV matrix.** There is no external reference panel shared across datasets in the standard dsSTICI use case.
2. **Observed-base masking is dataset-specific training corruption.** For each training example, a fixed fraction `--mr` of positions that are currently observed is changed to `?` in the input. Positions already missing remain missing.
3. **Originally missing target positions do not contribute to the loss.** The target is the uncorrupted sequence, but loss is evaluated only where the original target has an observed A/T/G/C call.
4. **Loss is evaluated on all originally observed target positions, not only positions newly masked for that training pass.** Thus both masked and still-visible observed bases contribute to reconstruction training.
5. **The loss is categorical cross-entropy + KL divergence only.** dsSTICI does **not** use the MaCH/Minimac-style R-squared term present as an optional/default-on term in the public STICI HPC implementation. Categorical cross-entropy and KL divergence themselves are inherited from STICI; they are not new loss families introduced by dsSTICI.
6. **There is no train/validation split in this dsSTICI script.** Learning-rate reduction and early stopping monitor training `rec_loss`.
7. **The optimizer remains LAMB**, as in the STICI implementation used as the starting point.
8. **Random seeding is now operational and recorded.** The cleaned release applies `--random-seed` to Python, NumPy, TensorFlow, and `tf.data`; each outer saved model uses `random_seed + zero_based_outer_model_index` so resumed chunk training remains reproducible.

### Exact reconstruction objective

For each batch, the model computes categorical cross-entropy (CCE) and KL divergence (KLD) over the flattened set of originally observed base positions:

```text
reconstruction loss = CCE(y, p) + KLD(y || p)
```

Targets are one-hot A/T/G/C vectors and predictions are four-way softmax probabilities. With one-hot targets, the KLD term is numerically very similar to the cross-entropy term (subject to TensorFlow's clipping conventions), so the two-term objective behaves approximately like a rescaled negative log-likelihood.

The implementation uses `tf.nn.compute_average_loss(..., global_batch_size=<number of cells in the global batch>)`. Therefore, per-site losses are summed across observed positions and normalized by the global **cell** batch size; they are not divided by the number of observed sites. This behavior is preserved intentionally because changing it would change the optimization objective.

---

## Model architecture in this repository

For a training window with `L` sites and embedding dimension `D`:

1. The input is one-hot encoded over five states: `A/T/G/C/?`.
2. `CatEmbeddings` projects the categorical input through a learned `5 x D` base embedding and adds a learned positional embedding of shape `L x D`.
3. Long windows are divided into internal chunks of `--cs` central sites, with `--co` neighboring sites supplied as context when available.
4. Each internal chunk uses:
   - dsSTICI self-attention with `--na-heads` heads. The query corresponds to the central chunk while keys/values include the context window. As in the historical implementation, Keras `MultiHeadAttention` uses `key_dim=D` **per head**.
   - an integrated convolution block;
   - a second convolution block whose output forms a skip branch;
   - `Dense(D, GELU)` followed by another convolution block;
   - cross-attention against the earlier self-attention representation using `--cross-attention-heads` heads (historically hard-coded to 8, now exposed as an argument);
   - dropout (`--dropout-rate`, historically hard-coded to 0.25);
   - another convolution block;
   - concatenation with the convolutional skip branch.
5. Internal chunk outputs are concatenated along the site axis.
6. A `Conv1D(D/2, kernel_size=5, GELU)` layer is followed by `Conv1D(4, kernel_size=5, softmax)` to predict A/T/G/C probabilities.

### Integrated convolution block

Each `ConvBlock(D)` uses:

```text
Conv1D(D, k=3, GELU)
  |-- Conv1D(D, k=5, GELU) -> Conv1D(D, k=7, GELU) --|
  |-- Conv1D(D, k=7, GELU) -> Conv1D(D, k=15, GELU) -| -> add
-> Conv1D(D, k=3, GELU)
-> BatchNorm
-> Conv1D(D, k=1)
-> BatchNorm
-> GELU
```

### Outer model partitioning

Very long CV matrices are divided into independently saved outer models. `--sites-per-model` is the number of central output sites assigned to one saved model. During training, each outer model receives up to `2 * --co` extra context sites on each side. At inference time the same saved partition/layout is recovered from `run_metadata.json` rather than relying on manually re-entered model-shape arguments.

---

## Keras serialization cleanup

The original dsSTICI-derived script placed live sublayers, metrics, losses, learned arrays, and build-derived tensors inside `get_config()`. That is not the intended Keras serialization contract and can make custom-object loading brittle.

In this cleaned implementation, `get_config()` contains only values required to reconstruct each object's constructor configuration. Keras separately tracks and restores child layers and learned weights.

Examples of values that **are** serialized:

- embedding dimension;
- attention-head counts;
- FFN dimension;
- chunk size/context and output offsets;
- activation identifier;
- dropout rate;
- initializer/regularizer/constraint configuration;
- global batch size required by the custom training loss scaling.

Examples of values that are **not** placed in `get_config()`:

- `Dense`, `Conv1D`, `LayerNormalization`, `BatchNormalization`, `MultiHeadAttention`, or nested model objects;
- learned embedding/weight NumPy arrays;
- position tensors created in `build()`;
- sequence length, channel count, chunk boundaries, or other values derived from input shape;
- Keras metrics and loss objects.

New models are saved as `*.keras`. Best-effort aliases for the historical names (`SplitTransformer`, `MaskedTransformerBlock`, `GenoEmbeddings`, and the old `MyLayers>`/`MyModels>` registered names) are retained so older checkpoints can be attempted with the cleaned loader.

A small serialization check is included:

```bash
python serialization_smoke_test.py
```

It creates a small dsSTICI model, saves it to `.keras`, reloads it through the custom-object registry, and verifies that predictions are unchanged.

---

## Software environment

The supplied TensorFlow environment was reduced to packages directly required by the cleaned dsSTICI script.

| Package | Version | Note |
|---|---:|---|
| Python | 3.9.21 | from supplied environment |
| TensorFlow | 2.14.1 | target TensorFlow release |
| Keras | 2.14.0 | from supplied environment |
| NumPy | 1.26.4 | from supplied environment |
| tqdm | 4.67.1 | from supplied environment |
| TensorFlow Addons | **0.22.0** | deliberately corrected: TFA 0.22 is the release built/tested for TensorFlow 2.14; the supplied environment contained 0.23.0, which targets TensorFlow 2.15 |

Unused packages were removed. In particular, the old inactive attention-export helper depended on Polars even though Polars was not present in the supplied environment; that inactive dependency has been removed from the cleaned script. `tensorflow-io` is also not used by dsSTICI and is omitted.

### Create the environment

```bash
conda env create -f environment.yml
conda activate dsstici-tf214
```

Verify the versions:

```bash
python - <<'PY'
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
print("TensorFlow:", tf.__version__)
print("TensorFlow Addons:", tfa.__version__)
print("NumPy:", np.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY
```

TensorFlow 2.14 targets CUDA 11.8-era GPU libraries. On an HPC system, load the site's compatible CUDA/cuDNN module before running if those libraries are provided by the cluster. The included SLURM script uses `CUDA_MODULE=cuda` by default and allows this module name to be overridden.

---

## Input format

The cleaned implementation reads plain or gzip-compressed FASTA files. Each FASTA record corresponds to one cell, and every sequence must contain the same number of sites in the same order.

Example:

```text
>cell_001
A?TGC??A...
>cell_002
ACT?C??A...
>cell_003
??TGCCTA...
```

Accepted states are:

- `A`, `T`, `G`, `C`: observed base calls;
- `?`: missing/unobserved call.

Input is converted to uppercase. Other symbols are treated as missing (`?`). Multi-line FASTA records are supported by the cleaned reader.

**Site order is part of the model definition.** Training and target FASTA files must contain the same number of sites in the same order. The FASTA format itself does not carry variant coordinates, so a public data release should provide a separate site/variant manifest that maps FASTA column number to chromosome/position/reference/alternate allele or the equivalent source identifier.

---

## Training

The following command makes all model/training choices explicit and corresponds to the important settings in the active TNBC5 block of the supplied historical SLURM job, while also making previously implicit defaults explicit:

```bash
python dsSTICI.py \
  --mode train \
  --ref data/processed/TNBC5.aneuploid.cellSNP.msa.fasta.gz \
  --save-dir results/dsSTICI_TNBC5 \
  --restart-training true \
  --batch-size-per-gpu 4 \
  --embed-dim 128 \
  --na-heads 16 \
  --cross-attention-heads 8 \
  --lr 5e-4 \
  --mr 0.8 \
  --epochs 1000 \
  --cs 2048 \
  --co 256 \
  --sites-per-model 16000 \
  --dropout-rate 0.25 \
  --random-seed 2024 \
  --deterministic-ops true \
  --verbose 2
```

`--restart-training true` deletes an existing `--save-dir`; use `false` to resume completed/pending outer chunks. When resuming, dsSTICI checks the saved training-file SHA256 and the training/model arguments and refuses to mix incompatible chunks in one run directory.

### Training parameter reference

| Argument | Code default | Meaning |
|---|---:|---|
| `--embed-dim` | 128 | categorical/positional embedding width |
| `--na-heads` | 16 | self-attention heads |
| `--cross-attention-heads` | 8 | cross-attention heads; historical value was hard-coded |
| `--dropout-rate` | 0.25 | dropout after cross-attention |
| `--cs` | 2048 | internal central chunk size |
| `--co` | 256 | internal context/overlap size |
| `--sites-per-model` | 16000 | central sites handled by each independently saved outer model |
| `--mr` | 0.8 | fraction of currently observed bases masked in each training example |
| `--lr` | 0.002 | LAMB learning rate; manuscript/dataset jobs should pass the actual value explicitly |
| `--batch-size-per-gpu` | 8 | per-GPU training batch size; historical TNBC5 job used 4 |
| `--epochs` | 1000 | maximum epochs per outer model |
| `--random-seed` | 2024 | base random seed |
| `--deterministic-ops` | false | requests deterministic TensorFlow ops; reproducibility SLURM example sets true |
| `--which-chunk` | -1 | train all pending outer models; positive values select one 1-based chunk |
| `--verbose` | 2 | Keras verbosity |

Training callbacks are fixed in the source:

- `ReduceLROnPlateau`: monitor `rec_loss`, factor `0.5`, patience `15`;
- `EarlyStopping`: monitor `rec_loss`, patience `50`, restore best weights.

There is no validation dataset in this implementation, so both callbacks operate on training reconstruction loss.

---

## Imputation

```bash
python dsSTICI.py \
  --mode impute \
  --save-dir results/dsSTICI_TNBC5 \
  --ref data/processed/TNBC5.aneuploid.cellSNP.msa.fasta.gz \
  --target data/processed/TNBC5.aneuploid.cellSNP.msa.fasta.gz \
  --batch-size-per-gpu 4 \
  --confidence-threshold 0.7 \
  --verbose 2
```

For cleaned runs, the outer partition parameters (`sites-per-model`, `co`, `cs`) and training site count are recovered from `run_metadata.json`. `--ref` is therefore optional during imputation, but supplying it is useful because dsSTICI verifies that its site count matches the saved model. Legacy runs without new metadata still require `--ref`.

The confidence threshold is applied to the maximum A/T/G/C probability. If the maximum is less than or equal to the threshold, the FASTA output contains `-` at that site.

**Important:** the prediction FASTA contains the model's predicted call at **every site**, including sites that were observed in the input. It is not a patching operation that automatically copies observed input bases back into the output. If an analysis requires observed calls to remain fixed, implement that explicitly and document it as a different post-processing procedure.

---

## Outputs and run metadata

A training directory contains, for example:

```text
results/dsSTICI_TNBC5/
├── run_metadata.json
├── commandline_args.json
├── models/
│   ├── chunks_info.json
│   ├── dsstici_chunk_001.keras
│   ├── dsstici_chunk_002.keras
│   └── ...
└── out/
    ├── predictions_confidence_0.7.fasta
    ├── top_probability_per_site.tsv
    └── imputation_metadata.json
```

`run_metadata.json` records:

- model and code version;
- exact command line;
- SHA256 of the executing `dsSTICI.py` file;
- Python/TensorFlow/Keras/TensorFlow-Addons/NumPy versions;
- TensorFlow CUDA/cuDNN build identifiers when available;
- random seed and deterministic-op setting;
- training FASTA path and SHA256;
- cell count and site count;
- every parsed training argument;
- global batch size and GPU replica count;
- outer-model breakpoints and per-outer-model seed rule.

`imputation_metadata.json` similarly records the target FASTA SHA256, software/code identifiers, confidence threshold, site/cell counts, and output paths.

`top_probability_per_site.tsv` contains one row per cell and one column per site. The value is `max(P(A), P(T), P(G), P(C))`; the missing class is not part of the prediction head and is therefore inherently excluded.

---

## SLURM

`slurm_dsSTICI.job` replaces the repeated hard-coded dataset blocks in the historical job with one reusable one-dataset job. Submit one job per dataset and pass dataset-specific paths/parameters through environment variables.

Example for TNBC5:

```bash
sbatch --export=ALL,\
DATASET_NAME=TNBC5,\
REF_FASTA=/path/to/TNBC5.aneuploid.cellSNP.msa.fasta.gz,\
TARGET_FASTA=/path/to/TNBC5.aneuploid.cellSNP.msa.fasta.gz,\
LEARNING_RATE=5e-4,\
BATCH_SIZE=4,\
MASK_RATE=0.8,\
RANDOM_SEED=2024 \
slurm_dsSTICI.job
```

A second dataset uses the same job rather than a copied code block:

```bash
sbatch --export=ALL,\
DATASET_NAME=OV025,\
REF_FASTA=/path/to/OV025_0.005Mar.fasta.gz,\
TARGET_FASTA=/path/to/OV025_0.005Mar.fasta.gz,\
LEARNING_RATE=1e-3,\
BATCH_SIZE=4,\
MASK_RATE=0.8,\
RANDOM_SEED=2024 \
slurm_dsSTICI.job
```

The job defaults to deterministic TensorFlow operations and explicitly passes the model/training hyperparameters. It leaves outputs in `SAVE_DIR/out` rather than repeatedly moving/renaming directories.

### Optional FastTree step

The historical job also ran FastTree after imputation. This is retained as an optional external step without hard-coding a lab-specific binary path:

```bash
sbatch --export=ALL,\
DATASET_NAME=TNBC5,\
REF_FASTA=/path/to/TNBC5.fasta.gz,\
RUN_FASTTREE=1,\
FASTTREE_BIN=/path/to/FastTreeMP \
slurm_dsSTICI.job
```

The model environment does not install FastTree; provide the executable separately when this option is used.

---

## Reproducibility notes from the historical scripts

The cleanup identified several points that should be resolved explicitly before claiming exact reproduction of manuscript analyses:

1. **Historical `--shuffle-variants` flag.** The supplied SLURM file passes `--shuffle-variants true` in multiple blocks, but the supplied Python source for this cleanup does not define that argument or a corresponding variant-permutation implementation. The cleaned job therefore does not use it. If variant shuffling was used in a reported manuscript analysis, the exact shuffling code, fixed permutation seed, and restoration of original site order must be released with that analysis. If it was not used, remove the flag from archived run scripts.
2. **Historical “dynamic” masking labels.** The active historical job defines `MIN_MASK_RATE=0.85` and `MAX_MASK_RATE=0.95` and uses output-directory names containing `MASK_RATE_DYNAMIC`, but those variables are not passed to the attached Python program. The attached dsSTICI code implements a fixed `--mr` (default `0.8`). If a dynamic masking schedule was used for a manuscript result, that implementation and exact schedule must be added separately; it is not present in the source supplied here.
3. **Historical random seed.** The prior Python CLI exposed `--random-seed`, but the attached implementation did not actually apply it to Python/NumPy/TensorFlow random generators. The cleaned release fixes that problem. Consequently, the cleaned code provides reproducible future runs, but a bitwise reconstruction of a historical stochastic run cannot be guaranteed from the old script merely by re-entering `2024`.
4. **Validation.** The supplied dsSTICI source does not create a validation split. Any manuscript statement implying validation-based model selection should be checked against the exact scripts used for that result.

These points are included because they materially affect what can be claimed as an exact computational reproduction.

---

## What must accompany a manuscript release

To fully address reproducibility requirements, source code alone is not enough. For each analysis reported in the manuscript, the public release should include or provide a stable accession/DOI for:

- the processed CV matrix used as dsSTICI input;
- a site manifest mapping FASTA columns to variant identifiers/coordinates;
- the exact `run_metadata.json` produced by training;
- the imputation metadata and confidence threshold;
- the exact SLURM/command invocation;
- random seed(s);
- scripts and parameters used to infer trees/genetic types from the imputed matrices;
- scripts/notebooks used to generate each main and supplementary figure from those outputs;
- checksums for processed data and key result files.

A recommended repository/data layout is:

```text
.
├── dsSTICI.py
├── environment.yml
├── serialization_smoke_test.py
├── slurm_dsSTICI.job
├── README.md
├── data/
│   └── processed/
│       ├── README.md              # accession/download instructions + checksums
│       ├── <dataset>.fasta.gz
│       └── <dataset>.sites.tsv
├── runs/
│   └── <dataset>/
│       └── run_metadata.json
└── scripts/
    ├── preprocessing/
    ├── phylogeny/
    └── figures/
```

The processed manuscript matrices and figure-generation scripts were **not part of the files supplied for this code-cleanup task**, so this repository skeleton cannot truthfully claim that those materials are already included. They should be added before the manuscript repository is finalized.

---

## Notes on reproducibility and hardware

GPU kernels can still show small numerical differences across GPU models, drivers, and CUDA/cuDNN builds even with fixed seeds. For the most reproducible runs:

- use the provided environment;
- use one GPU unless the exact multi-GPU topology is part of the recorded run;
- pass `--deterministic-ops true`;
- retain `run_metadata.json` and the training FASTA checksum;
- record the GPU model/driver in the job output.

The cleaned SLURM job prints TensorFlow/TensorFlow-Addons versions and visible GPUs at job start.

---

## License and attribution

dsSTICI is adapted from STICI. The upstream STICI repository is distributed under GPL-3.0. Before public release, retain appropriate upstream attribution and ensure the dsSTICI repository uses licensing consistent with the upstream code and your institution's requirements.

## References

Mowlaei ME, Li C, Jamialahmadi O, et al. **STICI: Split-Transformer with integrated convolutions for genotype imputation.** *Nature Communications*. 2025;16:1218. https://www.nature.com/articles/s41467-025-56273-3

STICI source: https://github.com/shilab/STICI
