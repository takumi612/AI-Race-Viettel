import pytest
from transformers import AutoTokenizer
from typing import Any, Mapping

from src.training.ner.data import tokenize_ner_records
from src.training.ner.bio import decode_bio_entities, ID2LABEL

@pytest.fixture(scope="module")
def tokenizer():
    # Use xlm-roberta-base as it matches the training config
    return AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)

def _mock_record(
    record_id: str,
    text: str = "",
    entities: list[dict[str, Any]] = None,
    tokens: list[str] = None,
    ner_tags: list[str] = None
) -> dict[str, Any]:
    res = {"record_id": record_id}
    if text:
        res["text"] = text
    if entities is not None:
        res["entities"] = entities
    if tokens is not None:
        res["tokens"] = tokens
    if ner_tags is not None:
        res["ner_tags"] = ner_tags
    return res

def test_1_normal_sentence(tokenizer):
    # Câu bình thường
    records = [_mock_record("1", text="Bác sĩ kê đơn thuốc", entities=[
        {"text": "thuốc", "type": "THUỐC", "position": [14, 19]}
    ])]
    features = tokenize_ner_records(records, tokenizer, max_length=128, stride=16)
    assert len(features) == 1
    assert len(features[0]["input_ids"]) == len(features[0]["absolute_offsets"]) == len(features[0]["labels"])
    # check valid extraction
    preds = decode_bio_entities(
        features[0]["text"], features[0]["absolute_offsets"], features[0]["labels"],
        attention_mask=features[0]["attention_mask"], record_id="1"
    )
    assert len(preds) == 1
    assert preds[0]["text"] == "thuốc"

def test_2_subword_split(tokenizer):
    # Từ bị tách thành nhiều subword: ví dụ xlm-roberta có thể tách "Paracetamol"
    records = [_mock_record("2", text="Uống Paracetamol 500mg", entities=[
        {"text": "Paracetamol 500mg", "type": "THUỐC", "position": [5, 22]}
    ])]
    features = tokenize_ner_records(records, tokenizer, max_length=128, stride=16)
    assert len(features[0]["labels"]) == len(features[0]["absolute_offsets"])
    preds = decode_bio_entities(
        features[0]["text"], features[0]["absolute_offsets"], features[0]["labels"],
        attention_mask=features[0]["attention_mask"], record_id="2"
    )
    assert preds[0]["text"] == "Paracetamol 500mg"

def test_3_padding_and_label_minus_100(tokenizer):
    # Câu có padding và label -100 (case 3 và 10)
    records = [_mock_record("3", tokens=["A", "B"], ner_tags=["O", "B-THUỐC"])]
    # Mặc dù tokenizer không padding mặc định, ta cố tình kiểm tra label -100
    features = tokenize_ner_records(records, tokenizer, max_length=128, stride=16)
    labels = features[0]["labels"]
    # xlm-roberta thêm <s> và </s> có word_ids = None => label = -100
    assert -100 in labels
    assert len(labels) == len(features[0]["input_ids"])

def test_4_truncation(tokenizer):
    # Câu bị truncation
    records = [_mock_record("4", text="A " * 100, entities=[])]
    features = tokenize_ner_records(records, tokenizer, max_length=10, stride=5)
    assert len(features) > 1 # bị cắt thành nhiều windows
    for f in features:
        assert len(f["labels"]) == len(f["input_ids"])
        assert len(f["input_ids"]) <= 10

def test_5_empty_annotation(tokenizer):
    # Annotation rỗng
    records = [_mock_record("5", text="Bệnh nhân khỏe mạnh", entities=[])]
    features = tokenize_ner_records(records, tokenizer, max_length=128, stride=16)
    labels = features[0]["labels"]
    assert all(l in (-100, 0) for l in labels) # Chỉ có -100 hoặc O (0)
    assert len(labels) == len(features[0]["input_ids"])

def test_6_annotation_at_end(tokenizer):
    # Annotation ở cuối câu
    records = [_mock_record("6", text="Tôi bị ho", entities=[
        {"text": "ho", "type": "TRIỆU_CHỨNG", "position": [7, 9]}
    ])]
    features = tokenize_ner_records(records, tokenizer, max_length=128, stride=16)
    preds = decode_bio_entities(
        features[0]["text"], features[0]["absolute_offsets"], features[0]["labels"],
        attention_mask=features[0]["attention_mask"], record_id="6"
    )
    assert preds[0]["text"] == "ho"

def test_7_unicode_vietnamese(tokenizer):
    # Unicode và tiếng Việt
    records = [_mock_record("7", text="Viêm phổi nặng cấp tính", entities=[
        {"text": "Viêm phổi nặng", "type": "CHẨN_ĐOÁN", "position": [0, 14]}
    ])]
    features = tokenize_ner_records(records, tokenizer, max_length=128, stride=16)
    preds = decode_bio_entities(
        features[0]["text"], features[0]["absolute_offsets"], features[0]["labels"],
        attention_mask=features[0]["attention_mask"], record_id="7"
    )
    assert preds[0]["text"] == "Viêm phổi nặng"

def test_8_special_characters(tokenizer):
    # Ký tự đặc biệt
    records = [_mock_record("8", text="COVID-19 (SARS-CoV-2)", entities=[
        {"text": "COVID-19", "type": "CHẨN_ĐOÁN", "position": [0, 8]}
    ])]
    features = tokenize_ner_records(records, tokenizer, max_length=128, stride=16)
    preds = decode_bio_entities(
        features[0]["text"], features[0]["absolute_offsets"], features[0]["labels"],
        attention_mask=features[0]["attention_mask"], record_id="8"
    )
    assert preds[0]["text"] == "COVID-19"

def test_9_offset_zero(tokenizer):
    # Offset (0, 0) - test 9
    # Special tokens có offset (0, 0)
    records = [_mock_record("9", text="Sốt", entities=[])]
    features = tokenize_ner_records(records, tokenizer, max_length=128, stride=16)
    offsets = features[0]["absolute_offsets"]
    assert [0, 0] in offsets
    preds = decode_bio_entities(
        features[0]["text"], features[0]["absolute_offsets"], features[0]["labels"],
        attention_mask=features[0]["attention_mask"], record_id="9"
    )
    assert len(preds) == 0

def test_10_word_level_labeling(tokenizer):
    # Từ annotation theo word-level
    records = [_mock_record("10", tokens=["Đau", "đầu"], ner_tags=["B-TRIỆU_CHỨNG", "I-TRIỆU_CHỨNG"])]
    features = tokenize_ner_records(records, tokenizer, max_length=128, stride=16)
    labels = features[0]["labels"]
    assert len(labels) == len(features[0]["input_ids"])
    # <s>, Đau, đầu, </s>
    # -100, x, y, -100
    assert labels[0] == -100
    assert labels[-1] == -100
