import ast
import importlib.util
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).parents[1]
WORKSPACE_ROOT = ROOT.parent
NOTEBOOK = ROOT / "medical_information_extraction_inference_kaggle.ipynb"
GENERATOR = ROOT / "tools" / "build_kaggle_inference_notebook.py"
ROOT_NOTEBOOK = WORKSPACE_ROOT / "train-ai-race-v2-32-8-inference-only.ipynb"
TRAINING_NOTEBOOK = WORKSPACE_ROOT / "train-ai-race-v2-32-8.ipynb"


def _load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_source() -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in _load_notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def _write_fake_results_tree(root: Path) -> None:
    artifact_dir = root / "artifacts"
    model_dir = root / "training_artifacts" / "ner_model"
    (artifact_dir / "icd10").mkdir(parents=True)
    (artifact_dir / "rxnorm").mkdir(parents=True)
    model_dir.mkdir(parents=True)
    for name in (
        "config.json",
        "entity_type_mapping.json",
        "assertion_mapping.json",
        "relation_mapping.json",
    ):
        (artifact_dir / name).write_text("{}\n", encoding="utf-8")
    (artifact_dir / "icd10" / "icd10_dictionary.jsonl.gz").write_bytes(b"fixture")
    (artifact_dir / "rxnorm" / "rxnorm_dictionary.jsonl.gz").write_bytes(b"fixture")
    labels = {
        "O": 0,
        "B-DISEASE": 1,
        "I-DISEASE": 2,
        "B-DRUG": 3,
        "I-DRUG": 4,
        "B-LAB_NAME": 5,
        "I-LAB_NAME": 6,
        "B-LAB_RESULT": 7,
        "I-LAB_RESULT": 8,
        "B-SYMPTOM": 9,
        "I-SYMPTOM": 10,
    }
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "xlm-roberta", "label2id": labels}),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"fixture")


def _execute_bootstrap_and_bundle(tmp_path: Path, archive_layout: bool) -> dict:
    kaggle_input = tmp_path / "kaggle_input"
    kaggle_working = tmp_path / "kaggle_working"
    results_tree = tmp_path / "fixture_results"
    kaggle_input.mkdir()
    _write_fake_results_tree(results_tree)
    dataset_root = kaggle_input / "results-dataset"
    dataset_root.mkdir()
    if archive_layout:
        with zipfile.ZipFile(dataset_root / "results.zip", "w") as archive:
            for path in results_tree.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(results_tree).as_posix())
    else:
        target = dataset_root / "results"
        for path in results_tree.rglob("*"):
            relative = path.relative_to(results_tree)
            if path.is_dir():
                (target / relative).mkdir(parents=True, exist_ok=True)
            else:
                (target / relative).parent.mkdir(parents=True, exist_ok=True)
                (target / relative).write_bytes(path.read_bytes())

    code_cells = [
        "".join(cell["source"])
        for cell in _load_notebook()["cells"]
        if cell["cell_type"] == "code"
    ]
    bootstrap = code_cells[0]
    bootstrap = bootstrap.replace(
        'KAGGLE_INPUT_ROOT = Path("/kaggle/input")',
        f"KAGGLE_INPUT_ROOT = Path({str(kaggle_input)!r})",
    ).replace(
        'KAGGLE_WORKING_ROOT = Path("/kaggle/working")',
        f"KAGGLE_WORKING_ROOT = Path({str(kaggle_working)!r})",
    ).replace(
        'PROJECT_ROOT_OVERRIDE = ""',
        f"PROJECT_ROOT_OVERRIDE = {str(ROOT)!r}",
    )
    namespace: dict = {}
    exec(compile(bootstrap, "<bootstrap>", "exec"), namespace)
    exec(compile(code_cells[1], "<bundle>", "exec"), namespace)
    return namespace


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
    assert 'KAGGLE_INPUT_ROOT.rglob("training_artifacts")' in source
    assert "Expected exactly one results.zip or extracted results directory" in source


def test_inference_notebook_accepts_the_actual_nested_artifact_layout():
    source = _code_source()
    assert 'ARTIFACT_PREFIX_CANDIDATES = ("artifacts/", "AI-Race-Viettel/v2/artifacts/")' in source
    assert "RESULTS_ARTIFACT_PREFIX" in source
    assert "relative_to(Path(RESULTS_ARTIFACT_PREFIX))" in source
    assert 'KAGGLE_WORKING_ROOT / "artifacts" / artifact_relative' in source


def test_inference_notebook_executes_with_kaggle_extracted_results_layout(tmp_path):
    namespace = _execute_bootstrap_and_bundle(tmp_path, archive_layout=False)
    assert namespace["RESULTS_SOURCE"].name == "results"
    assert (namespace["ARTIFACT_DIR"] / "config.json").is_file()
    assert (namespace["NER_MODEL_DIR"] / "model.safetensors").is_file()


def test_inference_notebook_executes_with_results_zip_layout(tmp_path):
    namespace = _execute_bootstrap_and_bundle(tmp_path, archive_layout=True)
    assert namespace["RESULTS_SOURCE"].name == "results.zip"
    assert (namespace["ARTIFACT_DIR"] / "config.json").is_file()
    assert (namespace["NER_MODEL_DIR"] / "model.safetensors").is_file()


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
    assert "QWEN_GPU_MEMORY_UTILIZATION = 0.50" in source
    assert "qwen_gpu_memory_utilization=QWEN_GPU_MEMORY_UTILIZATION" in source


def test_inference_notebook_clones_code_and_uses_data_only_results_bundle():
    source = _code_source()
    assert 'GITHUB_REPO_URL = "https://github.com/takumi612/AI-Race-Viettel.git"' in source
    assert 'GITHUB_COMMIT = "f2a699ee138f35311994da30b055739153e6dd2d"' in source
    assert '"git", "clone", GITHUB_REPO_URL, str(clone_dir)' in source
    assert '"git", "-C", str(clone_dir), "checkout", "--detach", GITHUB_COMMIT' in source
    assert '"config.json"' in source
    assert '"artifacts/"' in source
    assert '"AI-Race-Viettel/v2/clinical_nlp_lab/"' not in source


def test_inference_notebook_preflights_saved_config_and_model_metadata():
    source = _code_source()
    assert "RUNTIME_CONFIG = load_config(ARTIFACT_DIR / \"config.json\")" in source
    assert "REQUIRED_CONFIG_KEYS" in source
    assert "MODEL_CONFIG" in source
    assert 'MODEL_CONFIG.get("model_type") != "xlm-roberta"' in source
    assert "MODEL_LABELS" in source
    assert '\"config_compatibility\": \"validated\"' in source
    assert '\"source_commit\": GITHUB_COMMIT' in source


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
        "training_artifacts/ner_model/",
        "artifacts/",
        "AI-Race-Viettel/v2/artifacts/",
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


def test_root_inference_notebook_is_generated_from_the_canonical_builder():
    notebook = json.loads(ROOT_NOTEBOOK.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("inference_builder_root", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert notebook == module.build_notebook()


def test_root_inference_notebook_is_vietnamese_and_never_trains():
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(ROOT_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    )
    for phrase in ("suy luận", "dataset kết quả", "Run All", "output.zip"):
        assert phrase in source
    for forbidden in ("train_ner_subprocess.py", "Trainer(", ".train()"):
        assert forbidden not in source


def test_training_notebook_remains_present():
    assert TRAINING_NOTEBOOK.is_file()


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
