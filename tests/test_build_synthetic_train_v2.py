import json
from pathlib import Path

from scripts.build_synthetic_train_v2 import build_v2


ROOT = Path(__file__).resolve().parents[1]


def test_build_v2_keeps_200_source_pairs_and_adds_2000(tmp_path):
    result = build_v2(
        ROOT / "data_v2" / "Training_data" / "synthetic_train_v1",
        tmp_path,
        ROOT / "data" / "kb" / "metadata.db",
        synthetic_count=12,
    )
    assert result["total"] == 212
    assert result["copied"] == 200
    assert result["synthetic_generated"] == 12
    assert (tmp_path / "input" / "1.txt").exists()
    assert (tmp_path / "input" / "212.txt").exists()
    manifest = [
        json.loads(line)
        for line in (tmp_path / "reports" / "dataset_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(manifest) == 212
    assert manifest[0]["source_bucket"] == "reconstructed"
    assert manifest[0]["train_eligible"] is False
    assert manifest[0]["linking_train_eligible"] is False
    assert manifest[100]["source_bucket"] == "organizer_gt"
    assert manifest[100]["train_eligible"] is True
    assert manifest[100]["linking_train_eligible"] is False
    assert manifest[200]["source_bucket"] == "synthetic"
    assert manifest[200]["linking_train_eligible"] is True


def test_build_v2_removes_lab_assertions_and_affirmative_negation(tmp_path):
    build_v2(
        ROOT / "data_v2" / "Training_data" / "synthetic_train_v1",
        tmp_path,
        ROOT / "data" / "kb" / "metadata.db",
        synthetic_count=12,
    )
    for case_id in range(1, 201):
        text = (tmp_path / "input" / f"{case_id}.txt").read_text(encoding="utf-8")
        for entity in json.loads((tmp_path / "gt" / f"{case_id}.json").read_text(encoding="utf-8")):
            if entity["type"] in {"TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}:
                assert "assertions" not in entity
            assert text[entity["position"][0] : entity["position"][1]] == entity["text"]
