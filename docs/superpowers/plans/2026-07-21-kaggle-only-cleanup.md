# Kaggle-Only Project Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep only the maintained Kaggle runtime in `v2/`, remove the duplicated legacy/Colab tree, and verify that the active notebook remains runnable.

**Architecture:** `v2/` is the canonical package, artifact store, training subprocess, Kaggle notebook, runbook, and Kaggle builder. The legacy `Code_E_Platform/` tree and Colab-only files are removed because their useful runtime code is duplicated in `v2` and their active path is not used by the current Kaggle notebook.

**Tech Stack:** Python 3, Jupyter notebook JSON, pytest, PowerShell, Git.

## Global Constraints

- Preserve `v2/medical_information_extraction_kaggle.ipynb` and `/kaggle/working/output.zip` behavior.
- Do not modify or delete `ai-race-clinical-data/` or other user data.
- Do not stage existing unrelated changes or untracked user folders.
- Do not alter the remote `origin/Pipeline_colab` branch.
- Delete only the explicit legacy and Colab paths listed below.

---

### Task 1: Confirm deletion targets and references

**Files:**
- Read: `docs/superpowers/specs/2026-07-21-kaggle-only-cleanup-design.md`
- Read: `v2/medical_information_extraction_kaggle.ipynb`
- Read: `v2/tools/build_kaggle_notebook.py`

- [ ] Verify the deletion targets are exactly `Code_E_Platform/`, `v2/medical_information_extraction_lab.ipynb`, `v2/COLAB_RUNBOOK.md`, `v2/requirements-colab.txt`, `v2/tools/build_notebook.py`, and `v2/tools/execute_notebook.py`.
- [ ] Search maintained files for `Code_E_Platform`, `Pipeline_colab`, `medical_information_extraction_lab`, and `requirements-colab` references.
- [ ] Record any reference that must be removed or rewritten before deletion.

Run:

```powershell
rg -n -g '!ai-race-clinical-data/**' -g '!scratch/**' -g '!test/archive/**' 'Code_E_Platform|Pipeline_colab|medical_information_extraction_lab|requirements-colab' v2
```

Expected: only intentionally retained historical references are found before the rewrite.

### Task 2: Remove legacy and Colab-only runtime files

**Files:**
- Delete: `Code_E_Platform/`
- Delete: `test/` historical archive.
- Delete: `v2/medical_information_extraction_lab.ipynb`
- Delete: `v2/COLAB_RUNBOOK.md`
- Delete: `v2/requirements-colab.txt`
- Delete: `v2/tools/build_notebook.py`
- Delete: `v2/tools/execute_notebook.py`
- Delete: generated `src/` bytecode cache and `scratch/vllm-env/`.

- [ ] Delete only the explicit paths after resolving their absolute paths inside the workspace.
- [ ] Confirm the canonical Kaggle files remain present: `v2/medical_information_extraction_kaggle.ipynb`, `v2/KAGGLE_RUNBOOK.md`, `v2/requirements-kaggle.txt`, `v2/clinical_nlp_lab/`, `v2/artifacts/`, and `v2/tools/build_kaggle_notebook.py`.

### Task 3: Remove stale Kaggle-builder references and rewrite README

**Files:**
- Modify: `v2/tools/build_kaggle_notebook.py`
- Modify: `v2/README.md`

- [ ] Remove fallback candidate strings containing `Code_E_Platform` from the generated Kaggle notebook template while preserving the canonical-source build path.
- [ ] Remove the obsolete local fixture and `test/runtime_evidence` output path from the canonical Kaggle notebook and builder; local non-Kaggle output uses `v2/runtime/`.
- [ ] Rewrite `v2/README.md` as a Kaggle-only guide with the current files, input layout, run command, output ZIPs, and diagnostics paths.
- [ ] Do not reintroduce references to deleted Colab files.

### Task 4: Verify the cleaned project

**Files:**
- Verify: all remaining `v2/` Python modules and the canonical Kaggle notebook.

- [ ] Assert deleted paths no longer exist and preserved Kaggle paths do exist.
- [ ] Parse the notebook JSON and compile every code cell.
- [ ] Run `python -m pytest v2/tests -q -p no:cacheprovider`.
- [ ] Run `python -m py_compile` on all maintained Python files under `v2/`.
- [ ] Run `python v2/tools/build_kaggle_notebook.py --output v2/_cleanup_builder_check.ipynb`, validate the generated notebook, then remove only `v2/_cleanup_builder_check.ipynb`.
- [ ] Run `git diff --check` and inspect staged names before any commit.
