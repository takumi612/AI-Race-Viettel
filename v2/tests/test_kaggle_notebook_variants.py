from __future__ import annotations

import importlib.util
import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def _load_builder():
    path = ROOT / "tools" / "build_kaggle_notebook.py"
    spec = importlib.util.spec_from_file_location("build_kaggle_notebook", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_enabled_variant_bootstraps_cuda_129_vllm_before_runtime_imports():
    source = _source(_load_builder().build_notebook(enable_qwen_reranker=True))

    assert source.count("ENABLE_QWEN_RERANKER = True") == 1
    assert "vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" in source
    assert "https://download.pytorch.org/whl/cu129" in source
    assert "from vllm import LLM as _VLLM_IMPORT_CHECK" in source
    assert source.index("from vllm import LLM as _VLLM_IMPORT_CHECK") < source.index(
        "from clinical_nlp_lab.kaggle_phases import build_kaggle_phase_runners"
    )
    assert "--no-deps" not in source


def test_enabled_variant_single_toggle_guards_the_vllm_bootstrap():
    source = _source(_load_builder().build_notebook(enable_qwen_reranker=True))
    tree = ast.parse(source)
    guarded_bootstraps = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "ENABLE_QWEN_RERANKER"
        and "VLLM_CUDA129_WHEEL" in ast.unparse(node)
    ]

    assert len(guarded_bootstraps) == 1


def test_variant_markdown_names_the_selected_qwen_runtime():
    builder = _load_builder()
    enabled_markdown = "\n".join(
        "".join(cell["source"])
        for cell in builder.build_notebook(enable_qwen_reranker=True)["cells"]
        if cell["cell_type"] == "markdown"
    )
    unable_markdown = "\n".join(
        "".join(cell["source"])
        for cell in builder.build_notebook(enable_qwen_reranker=False)["cells"]
        if cell["cell_type"] == "markdown"
    )

    assert "Qwen enabled: vLLM 0.25.1 on CUDA 12.9" in enabled_markdown
    assert "Qwen disabled: vLLM is not installed or imported" in unable_markdown


def test_enabled_notebook_documents_entity_validation_smoke_fields():
    notebook = _load_builder().build_notebook(enable_qwen_reranker=True)
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )

    assert "qwen_entity_validation" in markdown
    assert "keep / drop / trim" in markdown
    assert "Nếu lỗi, lưu cell output và stack trace" in markdown


def test_unable_variant_has_no_vllm_install_or_import_path():
    source = _source(_load_builder().build_notebook(enable_qwen_reranker=False))

    assert source.count("ENABLE_QWEN_RERANKER = False") == 1
    assert "vllm" not in source.lower()


def test_enabled_and_unable_variants_keep_the_same_pipeline_phases():
    builder = _load_builder()
    enabled = builder.build_notebook(enable_qwen_reranker=True)
    unable = builder.build_notebook(enable_qwen_reranker=False)

    assert builder.validate_notebook(enabled)["phase_count"] == 13
    assert builder.validate_notebook(unable)["phase_count"] == 13
    assert [
        "".join(cell["source"])
        for cell in enabled["cells"]
        if cell["cell_type"] == "markdown" and "## Phase " in "".join(cell["source"])
    ] == [
        "".join(cell["source"])
        for cell in unable["cells"]
        if cell["cell_type"] == "markdown" and "## Phase " in "".join(cell["source"])
    ]


@pytest.mark.parametrize(
    ("notebook_path", "enable_qwen_reranker"),
    (
        (ROOT / "medical_information_extraction_kaggle.ipynb", True),
        (ROOT.parent / "ai-race-training-v2-enableQwen.ipynb", True),
        (ROOT.parent / "ai-race-training-v2-unableQwen.ipynb", False),
    ),
)
def test_checked_in_notebook_artifacts_match_builder_and_parse(
    notebook_path: Path,
    enable_qwen_reranker: bool,
):
    saved_notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    expected_notebook = _load_builder().build_notebook(
        enable_qwen_reranker=enable_qwen_reranker
    )

    assert saved_notebook == expected_notebook
    for index, cell in enumerate(saved_notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{notebook_path}:cell-{index}")
