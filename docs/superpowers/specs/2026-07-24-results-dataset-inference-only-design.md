# Kaggle Results-Dataset Inference-Only Notebook Design

## Goal

Create a new Kaggle notebook that loads the trained clinical NER checkpoint
and runtime artifacts from an attached results dataset, processes a separate
inference-input dataset, and writes a fresh submission archive without
performing any training.

The existing training notebook
`train-ai-race-v2-32-8.ipynb` remains unchanged.

## Deliverable

- `train-ai-race-v2-32-8-inference-only.ipynb`

The notebook must be self-contained from a Kaggle user's perspective:

1. Attach the results dataset.
2. Attach the new inference-input dataset.
3. Enable a GPU accelerator.
4. Import the notebook and run all cells.
5. Download `/kaggle/working/output.zip`.

## Kaggle Inputs

### Results dataset

Kaggle may expose the uploaded results archive in either of these forms:

1. An already-extracted directory containing:
   - `training_artifacts/ner_model/`
   - `AI-Race-Viettel/v2/clinical_nlp_lab/`
   - `AI-Race-Viettel/v2/artifacts/`
2. A `results.zip` file containing the same paths.

The notebook searches attached datasets under `/kaggle/input`, prefers a
complete already-extracted result tree, and otherwise extracts only the
required paths from `results.zip` into `/kaggle/working`.

A valid checkpoint requires at least:

- `model.safetensors`
- `config.json`
- tokenizer files loadable by Hugging Face Transformers

The runtime also requires the bundled `clinical_nlp_lab` package and its
ICD-10/RxNorm artifacts.

### Inference-input dataset

The notebook accepts either:

1. An already-extracted `input/` directory containing one or more
   `<document-id>.txt` files.
2. An `input.zip` archive containing that input directory or the text files.

Automatic discovery excludes paths belonging to the results dataset,
training data, historical outputs, diagnostics, checkpoint directories, and
the notebook working directory. Explicit override variables remain available
near the top of the notebook when Kaggle contains multiple valid candidates.

No annotation or ground-truth dataset is required.

## Alternatives Considered

### Fixed Kaggle dataset paths

Hard-code the two Kaggle dataset slugs and their internal directories.
This is simple but breaks whenever either dataset is renamed or its layout
changes.

### Extracted-directories only

Assume Kaggle always expands uploaded archives. This avoids extraction code
but makes the notebook fail when a dataset contains `results.zip` or
`input.zip` as an ordinary file.

### Dual-layout automatic discovery

Support both extracted directories and ZIP files, preferring the extracted
form. This adds a small discovery layer but gives the most reliable
upload-and-run workflow. This is the selected approach.

## Runtime Flow

1. Initialize deterministic settings, logging, Kaggle paths, and optional
   source overrides.
2. Discover and validate exactly one results source.
3. Reuse its extracted tree or selectively extract the model, code, and
   knowledge-base artifacts into `/kaggle/working/inference_runtime`.
4. Add the bundled runtime package to `sys.path`.
5. Discover and validate exactly one inference-input source.
6. Reuse its `input/` directory or extract `input.zip` into a separate
   working directory.
7. Install only missing inference-time dependencies.
8. Load the saved NER checkpoint and bundled retrieval/linking artifacts.
9. Run the existing hybrid inference pipeline for every input document.
10. Validate each output JSON and write a new
    `/kaggle/working/output.zip`.
11. Write a run manifest recording the selected sources, checkpoint,
    document count, output count, and `training_skipped: true`.

The notebook must not import or invoke a Trainer, call `.train()`, launch
`train_ner_subprocess.py`, create train/validation splits, or package a new
checkpoint.

## Output Contract

The final archive contains exactly one JSON prediction for each discovered
input text:

```text
output.zip
└── output/
    ├── <document-id-1>.json
    └── <document-id-2>.json
```

The notebook does not copy the historical `output.zip` or `output/` directory
from the results dataset. It always creates new predictions for the newly
attached input dataset.

## Error Handling

The notebook stops early with an actionable message when:

- no complete results source is found;
- multiple results sources remain after applying the override;
- the model checkpoint, tokenizer, runtime package, or knowledge base is
  incomplete;
- no valid input source is found;
- multiple input sources remain after applying the override;
- the selected input is inside results, training, diagnostics, checkpoint,
  output, or working directories;
- input document identifiers collide;
- prediction JSON violates the competition schema;
- output file names or counts do not match the input documents.

ZIP extraction rejects absolute paths and parent traversal entries and
extracts only expected prefixes.

## Verification

Local verification must confirm:

- the generated notebook is valid notebook JSON;
- all Python code cells compile;
- saved cell outputs and execution counts are cleared;
- no code cell invokes training or checkpoint packaging;
- both extracted-directory and ZIP discovery paths are represented;
- model, runtime, knowledge-base, input, and output validation are present;
- output ZIP layout and one-output-per-input checks are present;
- the original training notebook is unchanged.

A lightweight smoke test should use a tiny fake directory/archive fixture for
discovery and extraction logic. Full model inference is validated on Kaggle
because the real checkpoint is approximately 1.1 GB and requires the target
GPU/runtime stack.
