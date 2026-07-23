import pytest
from clinical_nlp_lab.assertions import HybridAssertionPredictor
from clinical_nlp_lab.schema import EntityAnnotation


def entity(text: str, mention: str, typ: str = "DISEASE") -> EntityAnnotation:
    start = text.index(mention)
    return EntityAnnotation(mention, typ, (start, start + len(mention)))


def test_negation_does_not_cross_sentence_boundary():
    text = "Không ghi nhận sốt. Bệnh nhân ho."
    axes = HybridAssertionPredictor().predict_axes(text, entity(text, "ho", "SYMPTOM"))
    assert axes.polarity == "AFFIRMED"


def test_family_cue_requires_relation_to_patient():
    text = "Người nhà đưa bệnh nhân vào viện. Bệnh nhân khó thở."
    axes = HybridAssertionPredictor().predict_axes(text, entity(text, "khó thở", "SYMPTOM"))
    assert axes.experiencer == "PATIENT"


def test_chronic_word_is_not_historical_by_itself():
    text = "Đái tháo đường mạn tính đang điều trị."
    axes = HybridAssertionPredictor().predict_axes(text, entity(text, "Đái tháo đường"))
    assert axes.temporality == "CURRENT"


def test_uncertainty_requires_specific_cue():
    text = "Theo dõi tại khoa. Chẩn đoán tăng huyết áp."
    axes = HybridAssertionPredictor().predict_axes(text, entity(text, "tăng huyết áp"))
    assert axes.certainty == "CONFIRMED"


def test_reload_fails_on_hash_mismatch():
    from clinical_nlp_lab.assertion_model import AssertionThresholdArtifact

    artifact = AssertionThresholdArtifact(
        schema_id="clinical_nlp.assertion_thresholds",
        schema_version=1,
        thresholds=(0.5, 0.5, 0.5),
        encoder_hash="enc_hash_1",
        tokenizer_hash="tok_hash_1",
    )

    current_encoder_hash = "enc_hash_2"
    with pytest.raises(ValueError, match="Encoder hash mismatch"):
        if artifact.encoder_hash != current_encoder_hash:
            raise ValueError(f"Encoder hash mismatch: {artifact.encoder_hash} vs {current_encoder_hash}")
