# Kaggle Inference-Only Notebook Design

## Goal

Create a Kaggle notebook that uses the trained NER checkpoint stored in
`results.zip` and produces a fresh `output.zip` for an attached inference
dataset without discovering annotations, splitting data, training XLM-R, or
packaging a new checkpoint.

## Deliverables

- `v2/medical_information_extraction_inference_kaggle.ipynb`
- `v2/KAGGLE_INFERENCE_RUNBOOK.md`

The existing training notebook and training runbook remain unchanged.

## Inputs

The notebook accepts two attached Kaggle datasets:

1. A results dataset containing `results.zip`.
2. An inference dataset containing either `input.zip` or `input/*.txt`.

`results.zip` is expected to contain:

```text
training_artifacts/ner_model/model.safetensors
training_artifacts/ner_model/config.json
training_artifacts/ner_model/tokenizer.json
AI-Race-Viettel/v2/clinical_nlp_lab/
AI-Race-Viettel/v2/artifacts/
```

The notebook automatically searches `/kaggle/input` for `results.zip` and the
inference input. Explicit override variables are provided for ambiguous
layouts.

## Runtime Flow

1. Configure a Kaggle-safe runtime and logging.
2. Locate and validate `results.zip` without extracting unrelated historical
   outputs or `.git` data.
3. Extract only the bundled `AI-Race-Viettel/v2` runtime and
   `training_artifacts/ner_model` checkpoint into `/kaggle/working`.
4. Locate the inference input while excluding training, archive, diagnostic,
   and extracted-results directories.
5. Install only missing inference dependencies.
6. Validate that the checkpoint and knowledge-base artifacts are complete.
7. Load the saved NER checkpoint and run the existing hybrid pipeline.
8. Validate all JSON outputs and the ZIP layout.
9. Write `/kaggle/working/output.zip`, `run_manifest.json`, and diagnostics.

There is no call to `Trainer`, `train_ner_subprocess.py`, or any other fitting
operation.

## Model Behavior

- NER uses `training_artifacts/ner_model` from `results.zip`.
- ICD-10 and RxNorm candidate retrieval use the artifacts bundled with the
  same code revision in `results.zip`.
- Qwen/vLLM reranking and assertion inference remain enabled when their
  runtime is available, matching the original pipeline behavior.
- The notebook does not reuse the historical `output.zip` from `results.zip`;
  it creates predictions for the newly attached inference input.

## Safety and Error Handling

The notebook fails early with a clear error when:

- zero or multiple unresolved `results.zip` files are found;
- the NER checkpoint is incomplete;
- the bundled project runtime is missing;
- no inference input is found;
- the selected input accidentally resolves inside extracted results,
  training data, diagnostics, or output directories;
- output JSON violates the competition schema;
- output ZIP count or layout does not match the input documents.

The run manifest records that training was skipped and identifies the loaded
checkpoint and results archive.

## Verification

Automated local verification will check:

- the notebook is valid JSON and contains no saved execution outputs;
- no code cell invokes the training subprocess or `Trainer.train()`;
- the notebook contains checkpoint, project, input, and ZIP validation;
- all Python code cells compile;
- the generated runbook documents Kaggle upload, attachment, GPU, Internet,
  execution, output download, and common failure cases.

