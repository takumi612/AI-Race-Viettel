import ast
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "medical_information_extraction_inference_kaggle.ipynb"
GENERATOR = ROOT / "tools" / "build_kaggle_inference_notebook.py"


def _load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_source() -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in _load_notebook()["cells"]
        if cell["cell_type"] == "code"
    )

def test_inference_notebook_is_clean_and_compiles():
    notebook = _load_notebook()
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]))


def test_inference_notebook_loads_results_without_training():
    source = _code_source()
    assert "RESULTS_ZIP_OVERRIDE" in source
    assert "training_artifacts/ner_model" in source
    assert "model.safetensors" in source
    assert "run_inference(" in source
    assert "validate_submission_payload" in source
    assert '"training_skipped": True' in source
    assert "train_ner_subprocess.py" not in source
    assert "Trainer(" not in source
    assert ".train()" not in source


def test_inference_notebook_accepts_kaggle_auto_extracted_results_directory():
    source = _code_source()
    assert "RESULTS_DIRS" in source
    assert "_copy_directory_bundle" in source
    assert "Expected exactly one results.zip or extracted results directory" in source


def test_inference_notebook_normalizes_kaggle_decompressed_jsonl_artifacts():
    source = _code_source()
    assert "_normalize_plain_knowledge_bases" in source
    assert "gzip.open" in source
    assert "rxnorm_dictionary" in source


def test_inference_notebook_validates_normalized_knowledge_base_files():
    source = _code_source()
    assert "missing_kb" in source
    assert "icd10_dictionary.jsonl.gz" in source
    assert "rxnorm_dictionary.jsonl.gz" in source


def test_inference_notebook_exposes_qwen_toggle():
    source = _code_source()
    assert "ENABLE_QWEN_RERANKER" in source
    assert "enable_qwen_reranker=ENABLE_QWEN_RERANKER" in source


def test_inference_notebook_clones_code_and_uses_data_only_results_bundle():
    source = _code_source()
    assert 'GITHUB_REPO_URL = "https://github.com/takumi612/AI-Race-Viettel.git"' in source
    assert '"git", "clone", "--depth", "1"' in source
    assert '"artifacts/config.json"' in source
    assert '"artifacts/"' in source
    assert '"AI-Race-Viettel/v2/clinical_nlp_lab/"' not in source


def test_results_archive_extracts_only_inference_runtime_members():
    source = _code_source()
    tree = ast.parse(source)
    extract_prefixes = next(
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "EXTRACT_PREFIXES" for target in node.targets)
    )
    assert extract_prefixes == (
        "artifacts/",
        "training_artifacts/ner_model/",
    )
    assert all("train_ner_subprocess.py" not in prefix for prefix in extract_prefixes)
    assert all(".git" not in prefix and ".ipynb" not in prefix for prefix in extract_prefixes)


def test_generated_notebook_is_exactly_current_generator_output():
    spec = importlib.util.spec_from_file_location("inference_notebook_builder", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = json.dumps(module.build_notebook(), ensure_ascii=False, indent=1) + "\n"
    assert NOTEBOOK.read_text(encoding="utf-8") == expected


def test_notebook_contains_executable_guards_for_unsafe_or_ambiguous_inputs():
    source = _code_source()
    assert 'if member.is_absolute() or ".." in member.parts:' in source
    assert 'raise ValueError(f"Unsafe archive member: {name!r}")' in source
    assert "if len(RESULTS_ZIPS) + len(RESULTS_DIRS) != 1:" in source
    assert "if len(valid_inputs) != 1:" in source
    assert 'raise RuntimeError(f"Expected exactly one inference input source' in source
    assert "if len(zip_names) != len(INPUT_DOCUMENTS) or zip_names != expected_names:" in source
    assert "bad_member = archive.testzip()" in source


def test_inference_notebook_checks_all_direct_bundled_requirements():
    source = _code_source()
    assert '"sentencepiece": "sentencepiece"' in source
    assert '"safetensors": "safetensors"' in source


def test_inference_runbook_covers_complete_kaggle_workflow():
    text = (ROOT / "KAGGLE_INFERENCE_RUNBOOK.md").read_text(encoding="utf-8")
    for phrase in (
        "results.zip",
        "input.zip",
        "medical_information_extraction_inference_kaggle.ipynb",
        "GPU",
        "Internet",
        "Run All",
        "output.zip",
        "training_skipped",
    ):
        assert phrase in text
    assert "artifacts/" in text
    assert "GitHub" in text
