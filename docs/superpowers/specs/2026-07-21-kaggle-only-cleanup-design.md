# Kaggle-Only Project Cleanup Design

## Goal

Make `v2/` the single maintained runtime for the current Kaggle workflow, remove the duplicated legacy `Code_E_Platform` tree and obsolete Colab-only assets, and leave a self-contained project without stale references.

## Findings

- `Code_E_Platform/clinical_nlp_lab` is a legacy copy of `v2/clinical_nlp_lab`.
- The useful legacy behavior is already present in `v2`; the remaining unique `rerank_subprocess.py` was replaced by the current batched vLLM path and explicit resource release.
- The legacy artifacts and notebooks duplicate the current `v2` artifacts/notebooks.
- The current Kaggle notebook and Kaggle runbook do not require `Code_E_Platform`.
- Colab-only files in `v2` are not used by the active Kaggle workflow.
- User data and untracked working folders remain out of scope for this cleanup.

## Target structure

```text
v2/
├── artifacts/                         # config, mappings, cached KBs
├── clinical_nlp_lab/                  # maintained runtime package
├── scripts/train_ner_subprocess.py   # NER training subprocess
├── tools/                             # Kaggle builder, KB builder, CLI tools
├── medical_information_extraction_kaggle.ipynb
├── requirements-kaggle.txt
├── KAGGLE_RUNBOOK.md
└── README.md
```

## Changes

Delete:

- `Code_E_Platform/` and all of its duplicate code, artifacts, notebooks, tools, and runbooks.
- `test/` historical archive; maintained tests live under `v2/tests/`.
- `v2/medical_information_extraction_lab.ipynb`.
- `v2/COLAB_RUNBOOK.md`.
- `v2/requirements-colab.txt`.
- `v2/tools/build_notebook.py`.
- `v2/tools/execute_notebook.py`.
- Generated `src/` bytecode cache and `scratch/vllm-env/` virtual environment.

Modify:

- `v2/README.md` to document only the Kaggle runtime and current output contract.
- `v2/tools/build_kaggle_notebook.py` to remove stale `Code_E_Platform` fallback references while preserving canonical notebook validation/build behavior.

Preserve:

- `ai-race-clinical-data/` and all other data directories/files.
- Existing user working files in `scratch/` (except the removed generated virtualenv/cache), `scripts/`, `.obsidian/`, `.worktrees/`, and unrelated `docs/` contents.
- The remote `origin/Pipeline_colab` branch; this cleanup changes only the current working tree/main branch.

## Safety and verification

- Before deletion, verify each target is inside the explicit legacy/Colab paths.
- Search the maintained tree for references to deleted paths.
- Parse and compile the canonical notebook.
- Run the existing `v2/tests` suite and Python compilation checks.
- Run `git diff --check` and inspect staged paths so user changes are not included.
