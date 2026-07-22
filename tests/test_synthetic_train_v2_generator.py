import json
import gzip
import re
import sqlite3
from pathlib import Path

from scripts.generate_synthetic_train_v2 import GENRES, build_document, generate_dataset


ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "kb" / "metadata.db"


def test_build_document_emits_exact_offsets_and_official_keys():
    text, entities = build_document(1, "cardiology", KB)
    assert entities
    assert all(text[e["position"][0] : e["position"][1]] == e["text"] for e in entities)
    assert all(
        set(e) == (
            {"text", "type", "position", "assertions", "candidates"}
            if e["type"] in {"CHẨN_ĐOÁN", "THUỐC"}
            else {"text", "type", "position", "assertions"}
            if e["type"] == "TRIỆU_CHỨNG"
            else {"text", "type", "position"}
        )
        for e in entities
    )


def test_build_document_has_coherent_cardiology_context():
    text, entities = build_document(2, "cardiology", KB)
    symptoms = [e["text"] for e in entities if e["type"] == "TRIỆU_CHỨNG" and not e.get("assertions")]
    diagnosis_codes = {
        candidate
        for entity in entities
        if entity["type"] == "CHẨN_ĐOÁN" and not entity.get("assertions")
        for candidate in entity["candidates"]
    }
    assert len(set(symptoms)) >= 2
    assert diagnosis_codes & {"I25.1", "I50", "I10"}
    assert not any(
        "isNegated" in e.get("assertions", [])
        for e in entities
        if e["type"] == "CHẨN_ĐOÁN"
    )


def test_document_genres_change_the_document_context():
    texts = {build_document(3, "cardiology", KB, genre=genre)[0].splitlines()[0] for genre in GENRES}
    assert len(texts) == len(GENRES)


def test_generate_dataset_creates_requested_pairs(tmp_path):
    result = generate_dataset(KB, tmp_path, count=12, seed=20260722)
    assert result["count"] == 12
    assert len(list((tmp_path / "input").glob("*.txt"))) == 12
    assert len(list((tmp_path / "gt").glob("*.json"))) == 12
    for path in (tmp_path / "gt").glob("*.json"):
        assert isinstance(json.loads(path.read_text(encoding="utf-8")), list)
    manifest = [
        json.loads(line)
        for line in (tmp_path / "reports" / "dataset_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(manifest) == 12
    assert manifest[0]["document_id"] == "1"
    assert manifest[0]["source_bucket"] == "synthetic"
    assert manifest[0]["genre"] in GENRES
    assert manifest[0]["template_group"]
    assert manifest[0]["train_eligible"] is True
    assert manifest[0]["linking_train_eligible"] is True
    assert isinstance(manifest[0]["primary_candidates"], list)
    assert len(manifest[0]["sha256"]) == 64


def test_generated_document_targets_realistic_length_and_not_fixed_negations():
    text, entities = build_document(41, "longtail", KB)
    assert 250 <= len(text.split()) <= 650
    symptom_texts = [e["text"] for e in entities if e["type"] == "TRIỆU_CHỨNG"]
    assert not ("buồn nôn" in symptom_texts and "nôn mửa" in symptom_texts)


def test_reason_symptoms_are_sampled_without_replacement():
    for case_id in range(201, 241):
        text, entities = build_document(case_id, "cardiology", KB)
        positive_symptoms = [
            entity["text"]
            for entity in entities
            if entity["type"] == "TRIỆU_CHỨNG" and not entity.get("assertions")
        ]
        assert len(positive_symptoms[:2]) == 2
        assert len(set(positive_symptoms[:2])) == 2


def test_longtail_mentions_use_neutral_non_indication_context():
    text, entities = build_document(205, "longtail", KB, genre="cap_cuu")

    assert "đối chiếu mã hóa" in text.casefold()
    assert "không dùng để suy diễn chỉ định" in text.casefold()
    assert any(
        entity["type"] == "CHẨN_ĐOÁN" and "isHistorical" in entity.get("assertions", [])
        for entity in entities
    )


def test_generated_candidates_are_compatible_with_inference_artifacts(tmp_path):
    artifact_root = ROOT / "v2" / "artifacts"
    with gzip.open(artifact_root / "icd10" / "icd10_dictionary.jsonl.gz", "rt", encoding="utf-8") as stream:
        icd_ids = {json.loads(line)["candidate_id"] for line in stream}
    with gzip.open(artifact_root / "rxnorm" / "rxnorm_dictionary.jsonl.gz", "rt", encoding="utf-8") as stream:
        rx_ids = {json.loads(line)["candidate_id"] for line in stream}
    generate_dataset(KB, tmp_path, count=60, seed=20260722, start_id=201)

    for path in (tmp_path / "gt").glob("*.json"):
        for entity in json.loads(path.read_text(encoding="utf-8")):
            if entity["type"] == "CHẨN_ĐOÁN":
                assert set(entity["candidates"]) <= icd_ids
            if entity["type"] == "THUỐC":
                assert set(entity["candidates"]) <= rx_ids


def test_generated_corpus_has_no_large_boilerplate_shared_by_every_document(tmp_path):
    generate_dataset(KB, tmp_path, count=48, seed=20260722, start_id=201)
    documents = [path.read_text(encoding="utf-8") for path in sorted((tmp_path / "input").glob("*.txt"))]
    common_lines = set(documents[0].splitlines())
    for document in documents[1:]:
        common_lines &= set(document.splitlines())
    fixed_characters = sum(len(line) for line in common_lines if line.strip())
    mean_characters = sum(map(len, documents)) / len(documents)

    assert fixed_characters / mean_characters < 0.25


def test_maternity_pediatric_genre_never_assigns_adult_male_demographics(tmp_path):
    generate_dataset(KB, tmp_path, count=120, seed=20260722, start_id=201)
    for path in (tmp_path / "input").glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        if not text.startswith("HỒ SƠ THEO DÕI THAI SẢN/NHI KHOA"):
            continue
        match = re.search(r"Bệnh nhân (nam|nữ) (\d+) tuổi", text)
        assert match
        sex, age = match.group(1), int(match.group(2))
        assert sex == "nữ" or age <= 15
