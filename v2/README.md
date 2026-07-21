# Clinical NLP Kaggle Pipeline

`v2/` is the maintained Kaggle runtime for medical information extraction.

## Runtime layout

```text
v2/
├── artifacts/                         # config, mappings and ICD-10/RxNorm caches
├── clinical_nlp_lab/                  # inference, linking, assertion and NER code
├── scripts/train_ner_subprocess.py   # isolated NER training process
├── tools/                             # Kaggle builder, KB builder and CLI tools
├── medical_information_extraction_kaggle.ipynb
├── requirements-kaggle.txt
├── KAGGLE_RUNBOOK.md
└── README.md
```

## Run on Kaggle

1. Import `medical_information_extraction_kaggle.ipynb`.
2. Attach the `ai-race-clinical-data` Dataset with `input.zip` and annotated training data.
3. Select a GPU accelerator and enable Internet, unless code/model Datasets are attached.
4. Choose **Run All**.

The notebook clones the current `main` branch when a code Dataset is not attached. It trains XLM-R NER, runs the hybrid clinical pipeline, and writes diagnostics and outputs under `/kaggle/working`.

See [KAGGLE_RUNBOOK.md](KAGGLE_RUNBOOK.md) for input layouts, offline mode, troubleshooting, and output details.

## Input and annotation layouts

The loader supports:

```text
input.zip
train/001.txt
train/001.json
```

or:

```text
input.zip
synthetic_train_v1/input/001.txt
synthetic_train_v1/gt/001.json
```

Annotation offsets use an exclusive end position and must satisfy:

```python
raw_text[start:end] == text
```

## Outputs

After a successful run, download:

- `/kaggle/working/output.zip`
- `/kaggle/working/trained_ner_artifacts.zip`
- `/kaggle/working/run_manifest.json`
- `/kaggle/working/diagnostics/run_summary.json`

The trained NER checkpoint is stored in `/kaggle/working/training_artifacts/ner_model/` during the run.

## Local checks

From the repository root:

```powershell
python -m pytest v2/tests -q -p no:cacheprovider
python v2/tools/build_kaggle_notebook.py --output v2/_generated_check.ipynb
```

The generated notebook is a validation/build artifact and should not be committed.
