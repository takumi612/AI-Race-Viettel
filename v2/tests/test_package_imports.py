from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_package_import_does_not_eagerly_import_pipeline_or_retrieval():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import clinical_nlp_lab; "
                "print('clinical_nlp_lab.pipeline' in sys.modules); "
                "print('clinical_nlp_lab.retrieval' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        env=environment,
    )

    assert completed.stdout.splitlines() == ["False", "False"]
