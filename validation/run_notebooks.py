#!/usr/bin/env python3
"""Execute the validation notebooks without requiring nbconvert/nbclient.

Only ordinary Python code cells are used.  The runner deliberately does not
rewrite notebook outputs; compact CSV, JSON, and PNG results are written to
``validation/results`` instead.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


VALIDATION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = VALIDATION_ROOT.parent
NOTEBOOKS = {
    "lif": VALIDATION_ROOT / "01_lif.ipynb",
    "scnn": VALIDATION_ROOT / "02_scnn.ipynb",
    "srnn": VALIDATION_ROOT / "03_srnn.ipynb",
}


def execute_notebook(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    namespace = {
        "__name__": "__main__",
        "__file__": str(path),
    }
    print(f"\n{'=' * 72}\nExecuting {path.name}\n{'=' * 72}")
    for index, cell in enumerate(payload.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        print(f"\n--- code cell {index} ---")
        exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", choices=["all", *NOTEBOOKS], default="all",
        help="Notebook(s) to execute",
    )
    parser.add_argument(
        "--no-csim", action="store_true",
        help="Do not invoke Vitis; reuse the latest successful CSim runs",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate projects and disable run reuse")
    parser.add_argument("--timeout-minutes", type=int, default=420)
    parser.add_argument("--arithmetic", choices=["float", "fixed"], default="float")
    parser.add_argument("--scnn-mode", choices=["numbers", "local"], default="numbers")
    parser.add_argument("--scnn-samples", type=int, default=10)
    parser.add_argument("--srnn-samples", type=int, default=140)
    parser.add_argument(
        "--download-large-reference-activity", action="store_true",
        help="Allow downloading the roughly 500 MB official SCNN activity arrays",
    )
    arguments = parser.parse_args()

    os.environ["NEUROHLS_RUN_CSIM"] = "0" if arguments.no_csim else "1"
    os.environ["NEUROHLS_FORCE"] = "1" if arguments.force else "0"
    os.environ["NEUROHLS_TIMEOUT_MINUTES"] = str(arguments.timeout_minutes)
    os.environ["NEUROHLS_ARITHMETIC"] = arguments.arithmetic
    os.environ["NEUROHLS_SCNN_MODE"] = arguments.scnn_mode
    os.environ["NEUROHLS_SCNN_SAMPLES"] = str(arguments.scnn_samples)
    os.environ["NEUROHLS_SRNN_SAMPLES"] = str(arguments.srnn_samples)
    os.environ["NEUROHLS_DOWNLOAD_LARGE_REFERENCE_ACTIVITY"] = (
        "1" if arguments.download_large_reference_activity else "0"
    )

    os.chdir(REPO_ROOT)
    selected = NOTEBOOKS if arguments.experiment == "all" else {
        arguments.experiment: NOTEBOOKS[arguments.experiment]
    }
    for path in selected.values():
        execute_notebook(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

