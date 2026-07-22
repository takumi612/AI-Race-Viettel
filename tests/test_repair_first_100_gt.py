from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.repair_first_100_gt import repair_first_100_ground_truth


def _build_kb(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute("create table icd10 (code text primary key, name_vi text, name_en text)")
    connection.execute("create table rxnorm (rxcui text, name text, tty text)")
    connection.execute("insert into icd10 values ('I10', 'tăng huyết áp', 'hypertension')")
    connection.execute("insert into rxnorm values ('6809', 'metformin', 'IN')")
    connection.commit()
    connection.close()


def test_repair_fills_exact_candidates_and_removes_false_historical(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "input").mkdir(parents=True)
    (dataset / "gt").mkdir()
    text = "Chẩn đoán tăng huyết áp. Bệnh nhân đang sử dụng thuốc metformin tại nhà."
    disease_start = text.index("tăng huyết áp")
    drug_start = text.index("metformin")
    annotations = [
        {
            "text": "tăng huyết áp",
            "type": "CHẨN_ĐOÁN",
            "position": [disease_start, disease_start + len("tăng huyết áp")],
            "assertions": [],
            "candidates": [],
        },
        {
            "text": "metformin",
            "type": "THUỐC",
            "position": [drug_start, drug_start + len("metformin")],
            "assertions": ["isHistorical"],
            "candidates": [],
        },
    ]
    (dataset / "input" / "1.txt").write_text(text, encoding="utf-8")
    (dataset / "gt" / "1.json").write_text(json.dumps(annotations, ensure_ascii=False), encoding="utf-8")
    kb = tmp_path / "metadata.db"
    _build_kb(kb)

    report = repair_first_100_ground_truth(dataset, kb, document_ids=[1])
    repaired = json.loads((dataset / "gt" / "1.json").read_text(encoding="utf-8"))

    assert repaired[0]["candidates"] == ["I10"]
    assert repaired[1]["candidates"] == ["6809"]
    assert repaired[1]["assertions"] == []
    assert report["candidate_entities_repaired"] == 2
    assert report["historical_assertions_removed"] == 1
    assert report["train_excluded_ids"] == ["1"]


def test_repair_does_not_touch_organizer_ground_truth(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "input").mkdir(parents=True)
    (dataset / "gt").mkdir()
    for document_id in (100, 101):
        (dataset / "input" / f"{document_id}.txt").write_text("", encoding="utf-8")
        (dataset / "gt" / f"{document_id}.json").write_text("[]\n", encoding="utf-8")
    kb = tmp_path / "metadata.db"
    _build_kb(kb)
    organizer_before = (dataset / "gt" / "101.json").read_bytes()

    repair_first_100_ground_truth(dataset, kb, document_ids=[100])

    assert (dataset / "gt" / "101.json").read_bytes() == organizer_before


def test_repair_canonicalizes_existing_icd_display_markers(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "input").mkdir(parents=True)
    (dataset / "gt").mkdir()
    text = "tăng huyết áp"
    (dataset / "input" / "1.txt").write_text(text, encoding="utf-8")
    (dataset / "gt" / "1.json").write_text(
        json.dumps(
            [
                {
                    "text": text,
                    "type": "CHẨN_ĐOÁN",
                    "position": [0, len(text)],
                    "assertions": [],
                    "candidates": ["I10†"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    kb = tmp_path / "metadata.db"
    _build_kb(kb)

    report = repair_first_100_ground_truth(dataset, kb, document_ids=[1])
    repaired = json.loads((dataset / "gt" / "1.json").read_text(encoding="utf-8"))

    assert repaired[0]["candidates"] == ["I10"]
    assert report["candidate_entities_canonicalized"] == 1
