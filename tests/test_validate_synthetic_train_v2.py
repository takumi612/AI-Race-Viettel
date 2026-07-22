from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

from scripts.validate_synthetic_train_v2 import validate


def _write_jsonl_gz(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_validator_uses_canonical_inference_artifact_candidates(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "input").mkdir(parents=True)
    (dataset / "gt").mkdir()
    (dataset / "reports").mkdir()
    text = "bệnh lý thần kinh"
    (dataset / "input" / "1.txt").write_text(text, encoding="utf-8")
    (dataset / "gt" / "1.json").write_text(
        json.dumps(
            [
                {
                    "text": text,
                    "type": "CHẨN_ĐOÁN",
                    "position": [0, len(text)],
                    "assertions": [],
                    "candidates": ["G94*"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = {
        "document_id": "1",
        "source_bucket": "reconstructed",
        "genre": "test",
        "template_group": "test",
        "long_tail": False,
        "train_eligible": False,
        "linking_train_eligible": False,
        "train_exclusion_reason": "fixture",
        "primary_candidates": ["G94"],
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    (dataset / "reports" / "dataset_manifest.jsonl").write_text(
        json.dumps(manifest, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    kb = tmp_path / "metadata.db"
    connection = sqlite3.connect(kb)
    connection.execute("create table icd10 (code text primary key, name_vi text, name_en text)")
    connection.execute("create table rxnorm (rxcui text, name text, tty text)")
    connection.execute("insert into icd10 values ('G94*', ?, 'neurologic disease')", (text,))
    connection.commit()
    connection.close()
    artifacts = tmp_path / "artifacts"
    _write_jsonl_gz(
        artifacts / "icd10" / "icd10_dictionary.jsonl.gz",
        [{"candidate_id": "G94", "canonical_name": text}],
    )
    _write_jsonl_gz(artifacts / "rxnorm" / "rxnorm_dictionary.jsonl.gz", [])

    report = validate(dataset, kb, artifact_root=artifacts, expected_documents=1)

    assert report["errors"] == []
    assert report["train_eligible_documents"] == 0
    assert report["warnings"][0]["kind"] == "noncanonical_icd_display_candidate"
