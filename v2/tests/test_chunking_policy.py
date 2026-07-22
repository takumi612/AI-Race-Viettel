from clinical_nlp_lab.ner import merge_chunk_predictions
from clinical_nlp_lab.schema import EntityAnnotation
from clinical_nlp_lab.training import character_spans_to_bio


def test_partial_entity_at_chunk_boundary_is_ignored_not_taught_as_o():
    entity = EntityAnnotation("disease", "DISEASE", (4, 11))
    labels = character_spans_to_bio(
        [(0, 0), (7, 11), (12, 15), (0, 0)],
        [entity],
        {"O": 0, "B-DISEASE": 1, "I-DISEASE": 2},
    )
    assert labels == [-100, -100, 0, -100]


def test_complete_entity_in_overlapping_chunk_gets_normal_bio_labels():
    entity = EntityAnnotation("disease", "DISEASE", (4, 11))
    labels = character_spans_to_bio(
        [(0, 0), (4, 7), (7, 11), (12, 15), (0, 0)],
        [entity],
        {"O": 0, "B-DISEASE": 1, "I-DISEASE": 2},
    )
    assert labels == [-100, 1, 2, 0, -100]


def test_overlapping_same_type_chunk_predictions_are_merged_before_resolution():
    text = "0123456789"
    left = EntityAnnotation("23456", "DISEASE", (2, 7), confidence=0.95)
    right = EntityAnnotation("45678", "DISEASE", (4, 9), confidence=0.80)
    merged = merge_chunk_predictions([left, right], text)
    assert len(merged) == 1
    assert merged[0].position == (2, 9)
    assert merged[0].text == "2345678"
