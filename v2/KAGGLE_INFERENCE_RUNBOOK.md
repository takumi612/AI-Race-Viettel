# Kaggle inference-only runbook

This notebook runs inference from an existing NER checkpoint. It does **not**
train or fine-tune a model.

## 1. Prepare the two private Kaggle Datasets

1. Create a **private** Kaggle Dataset for the trained artifacts. Upload the
   archive named `results.zip` exactly as it is; do not rename it or unpack it.
   The archive must contain the checkpoint members under
   `training_artifacts/ner_model/`, including exactly `config.json`,
   `model.safetensors`, and `tokenizer.json`. It must also contain
   `AI-Race-Viettel/v2/requirements-kaggle.txt`, which the notebook validates
   before installing inference dependencies.
2. Create or select a separate private inference Dataset. It must contain
   either a top-level `input.zip` or an `input/` directory with one or more
   `*.txt` documents. This Dataset should contain only the documents to infer,
   not training data or prior outputs.

Keeping artifacts and input in separate Datasets avoids accidentally treating
an old archive as the current submission.

## 2. Configure the Kaggle notebook

1. In Kaggle Code, choose **New Notebook**, then import
   `medical_information_extraction_inference_kaggle.ipynb`.
2. In the notebook's Data panel, attach both Datasets: the Dataset that
   contains `results.zip` and the separate Dataset containing `input.zip` or
   `input/*.txt`.
3. Open Notebook Settings, select a **GPU** accelerator, and enable
   **Internet**. Internet lets the notebook install any dependencies not
   already present in the Kaggle image.
4. Start a fresh session and choose **Run All**. Do not execute only the final
   cells: earlier cells locate the archives, install requirements, and restore
   the checkpoint.

## 3. Verify and collect the new result

After all cells complete:

1. Open `/kaggle/working/run_manifest.json` and confirm it contains
   `"training_skipped": true`. If it is absent or false, stop: this run does
   not meet the inference-only contract.
2. Download the fresh `/kaggle/working/output.zip` from the Kaggle Output
   panel (use **Save Version → Save & Run All** first if needed).
3. Optionally download diagnostics and `run_manifest.json` with the ZIP for
   traceability.

`results.zip` can include a historical `output.zip` from a prior run. That
embedded historical `output.zip` is **not** the new submission. Only the
fresh `/kaggle/working/output.zip` produced by this session should be
downloaded and submitted.

## Troubleshooting

| Symptom | Concrete fix |
| --- | --- |
| More than one `results.zip` is attached, or the notebook selects the wrong one | Detach unrelated artifact Datasets. If necessary, set `RESULTS_ZIP_OVERRIDE` in the first configuration cell to the exact `/kaggle/input/<dataset>/results.zip` path, then Run All again. |
| Checkpoint members are missing | Rebuild the artifact Dataset from the complete original `results.zip`; verify it has `training_artifacts/ner_model/config.json`, `model.safetensors`, and `tokenizer.json` before upload. Do not upload an extracted partial folder or an old `output.zip` instead. |
| No input is found | Attach the separate inference Dataset and ensure it has a top-level `input.zip` or `input/*.txt` with at least one non-empty document. Remove training folders and old outputs from that Dataset. |
| A package/dependency import fails | Enable Internet, restart the Kaggle session, and Run All so the requirements cell can install dependencies. If Internet is unavailable, attach a Dataset with compatible offline wheels and install those in the setup cell. |
| GPU or vLLM error | Select a GPU in Notebook Settings, restart the session, and Run All. GPU/vLLM runtime errors stop the run; for out-of-memory or unsupported vLLM/CUDA combinations, choose a larger compatible GPU when available or close other GPU notebooks, then retry. |
| `output.zip` validation, ZIP, or schema error | Download `/kaggle/working/diagnostics` and inspect the reported document IDs. Re-run with valid UTF-8 input text, then verify the fresh `output.zip` opens cleanly and contains the required `output/<document_id>.json` members with valid submission JSON before submitting. |

If any error persists, preserve the Kaggle logs, `run_manifest.json`, and
diagnostics with the exact Dataset paths used; these identify whether the
failure is artifact discovery, checkpoint restoration, input discovery, or
output validation.
