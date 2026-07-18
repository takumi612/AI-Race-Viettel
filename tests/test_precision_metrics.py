from src.evaluation.precision_metrics import (
    fbeta,
    score_assertions,
    score_candidate_sets,
    score_entities_by_type,
    score_exact_entities,
)
from src.evaluation.trusted_split import development_ids, holdout_ids


def test_f05_weights_precision_more_than_recall():
    high_precision = fbeta(precision=0.9, recall=0.5, beta=0.5)
    high_recall = fbeta(precision=0.5, recall=0.9, beta=0.5)
    assert high_precision > high_recall


def test_exact_entity_metric_penalizes_wrong_type_twice():
    gt = [{"text": "ho", "type": "TRIỆU_CHỨNG", "position": [0, 2]}]
    pred = [{"text": "ho", "type": "CHẨN_ĐOÁN", "position": [0, 2]}]
    result = score_exact_entities(gt, pred)
    assert (result.tp, result.fp, result.fn) == (0, 1, 1)
    assert result.fbeta == 0.0


def test_exact_entity_metric_penalizes_duplicate_predictions():
    entity = {"text": "ho", "type": "TRIỆU_CHỨNG", "position": [0, 2]}
    result = score_exact_entities([entity], [entity, dict(entity)])
    assert (result.tp, result.fp, result.fn) == (1, 1, 0)
    assert result.precision == 0.5


def test_trusted_split_is_disjoint_and_immutable():
    assert development_ids() == tuple(range(101, 181))
    assert holdout_ids() == tuple(range(181, 201))
    assert set(development_ids()).isdisjoint(holdout_ids())


def test_per_type_and_assertion_metrics_expose_precision_errors():
    gt = [{
        "text": "ho",
        "type": "TRIỆU_CHỨNG",
        "position": [0, 2],
        "assertions": ["isNegated"],
    }]
    pred = [{
        "text": "ho",
        "type": "TRIỆU_CHỨNG",
        "position": [0, 2],
        "assertions": ["isHistorical"],
    }]
    assert score_entities_by_type(gt, pred)["TRIỆU_CHỨNG"].precision == 1.0
    assertions = score_assertions(gt, pred)
    assert assertions.by_label["isNegated"].fn == 1
    assert assertions.by_label["isHistorical"].fp == 1


def test_candidate_metrics_separate_selector_precision_from_retrieval_recall():
    gt = [{"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [0, 13], "candidates": ["I10"]}]
    pred = [{"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [0, 13], "candidates": ["I10", "I11"]}]
    retrieved = {(0, 13, "CHẨN_ĐOÁN"): ["I10", "I11", "I12"]}
    metrics = score_candidate_sets(gt, pred, retrieved=retrieved)
    assert metrics.jaccard == 0.5
    assert metrics.precision == 0.5
    assert metrics.top1_hit_rate == 1.0
    assert metrics.recall_at_20 == 1.0
