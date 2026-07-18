_DEVELOPMENT_IDS = tuple(range(101, 181))
_HOLDOUT_IDS = tuple(range(181, 201))


def development_ids() -> tuple[int, ...]:
    return _DEVELOPMENT_IDS


def holdout_ids() -> tuple[int, ...]:
    return _HOLDOUT_IDS
