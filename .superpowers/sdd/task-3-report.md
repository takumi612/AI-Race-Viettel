# Task 3 Report — Kaggle Inference Runbook

## Status

Completed. Added `v2/KAGGLE_INFERENCE_RUNBOOK.md` and the required runbook
contract test in `v2/tests/test_inference_notebook.py`.

## Verification

```text
$ python -m pytest v2/tests/test_inference_notebook.py -q -p no:cacheprovider
....                                                                     [100%]
4 passed in 0.02s

$ python -m pytest v2/tests -q -p no:cacheprovider
..................                                                       [100%]
18 passed in 11.28s

$ python v2/tools/build_kaggle_inference_notebook.py --output v2/_generated_inference_check.ipynb
{"output": "v2\\_generated_inference_check.ipynb", "cell_count": 14, "code_cell_count": 6, "empty_code_cells": [], "valid": true}

$ python -c "... generated notebook structural assertions ..."
Generated notebook verified: 14 cells, 13243 bytes
```

## Commit

`d72b353` — `docs: add Kaggle inference runbook`

## Concerns

- `_generated_inference_check.ipynb` was intentionally left untracked after
  its contents were verified, per the task brief.
- The runbook makes Kaggle-side execution instructions explicit, but it cannot
  verify the user's Kaggle Dataset permissions, GPU availability, or Internet
  setting locally.

## Final-review remediation (2026-07-21)

### Changes

- Restricted `results.zip` extraction to the runtime-only project paths:
  `clinical_nlp_lab/`, `artifacts/`, `requirements-kaggle.txt`, and the NER
  checkpoint. The archive now also requires `requirements-kaggle.txt` before
  it is extracted for dependency installation.
- Regenerated the committed inference notebook from its generator, restoring
  the correct `Save Version → Save & Run All` Unicode text.
- Added regression coverage for the extraction allowlist (including explicit
  exclusions for training, `.git`, and notebook members), generator/notebook
  byte-for-byte synchronization, and emitted executable guards for unsafe
  archive paths, ambiguous results/input discovery, and invalid output ZIPs.
- Aligned the runbook with the exact required NER files and documented that
  GPU/vLLM runtime errors stop the run and must be remediated before retrying.

### Verification

```text
$ python -m pytest v2/tests/test_inference_notebook.py -q -p no:cacheprovider
.......                                                                  [100%]
7 passed in 0.03s

$ python -m pytest v2/tests -q -p no:cacheprovider
.....................                                                    [100%]
21 passed in 4.43s

$ python v2/tools/build_kaggle_inference_notebook.py --output v2/_generated_inference_check.ipynb
{"output": "v2\\_generated_inference_check.ipynb", "cell_count": 14, "code_cell_count": 6, "empty_code_cells": [], "valid": true}

$ python -m py_compile v2/tools/build_kaggle_inference_notebook.py
```

## Residual docs remediation (2026-07-21)

The artifact checklist now explicitly requires
`AI-Race-Viettel/v2/requirements-kaggle.txt`, matching the notebook's archive
validation before dependency installation. Added a runbook contract assertion
for this exact required path.

### Verification

```text
$ python -m pytest v2/tests/test_inference_notebook.py -q -p no:cacheprovider
.......                                                                  [100%]
7 passed in 0.02s

$ python -m pytest v2/tests -q -p no:cacheprovider
.....................                                                    [100%]
21 passed in 4.31s
```
