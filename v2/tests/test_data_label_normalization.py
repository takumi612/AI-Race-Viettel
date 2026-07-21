import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _load_data_module():
    package = types.ModuleType("clinical_nlp_lab")
    package.__path__ = [str(ROOT / "clinical_nlp_lab")]
    sys.modules["clinical_nlp_lab"] = package

    schema_spec = importlib.util.spec_from_file_location(
        "clinical_nlp_lab.schema", ROOT / "clinical_nlp_lab" / "schema.py"
    )
    schema_module = importlib.util.module_from_spec(schema_spec)
    sys.modules["clinical_nlp_lab.schema"] = schema_module
    assert schema_spec.loader is not None
    schema_spec.loader.exec_module(schema_module)

    data_spec = importlib.util.spec_from_file_location(
        "clinical_nlp_lab.data", ROOT / "clinical_nlp_lab" / "data.py"
    )
    data_module = importlib.util.module_from_spec(data_spec)
    sys.modules["clinical_nlp_lab.data"] = data_module
    assert data_spec.loader is not None
    data_spec.loader.exec_module(data_module)
    return data_module


def test_load_annotated_documents_normalizes_official_vietnamese_types(tmp_path):
    data = _load_data_module()
    input_dir = tmp_path / "input"
    gt_dir = tmp_path / "gt"
    input_dir.mkdir()
    gt_dir.mkdir()
    raw_text = "Đau ngực, xét nghiệm glucose."
    (input_dir / "1.txt").write_text(raw_text, encoding="utf-8")
    payload = [
        {"text": "Đau ngực", "type": "TRIỆU_CHỨNG", "position": [0, 8]},
        {"text": "xét nghiệm", "type": "TÊN_XÉT_NGHIỆM", "position": [10, 20]},
    ]
    (gt_dir / "1.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    documents = data.load_annotated_documents(tmp_path)

    assert [entity.type for entity in documents[0].entities] == ["SYMPTOM", "LAB_NAME"]
