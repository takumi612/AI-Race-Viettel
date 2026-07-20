from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path


def execute(path: Path) -> dict[str, int | bool]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__notebook__", "__file__": str(path)}
    executed = 0
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            exec(compile(source, f"{path}:cell-{index}", "exec"), namespace, namespace)
        except Exception as exc:
            print(f"Notebook cell {index} failed: {exc}")
            traceback.print_exc()
            raise
        executed += 1
    return {"valid": True, "executed_code_cells": executed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a project notebook sequentially without a Jupyter server")
    parser.add_argument("path", type=Path, default=Path("medical_information_extraction_lab.ipynb"), nargs="?")
    args = parser.parse_args()
    print(json.dumps(execute(args.path.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
