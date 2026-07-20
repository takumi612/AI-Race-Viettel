from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def metadata_db(project_root: Path) -> Path:
    path = project_root / "data" / "kb" / "metadata.db"
    if not path.exists():
        pytest.skip(f"metadata DB is unavailable: {path}")
    return path
