"""
Master Execution Script: Runs the complete Molecular Potential Energy Surface (PES)
VQE simulation pipeline for H2 and LiH, executes all optimizer/mapping/ansatz
benchmarks, extracts spectroscopic constants, and generates all 5 publication figures.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_h2_pes import main as run_h2
from experiments.run_lih_pes import main as run_lih
from experiments.run_benchmarks import main as run_bench


def main() -> None:
    t_start = time.perf_counter()
    print("#" * 76)
    print("# MOLECULAR POTENTIAL ENERGY SURFACE MAPPING VIA VQE — FULL PIPELINE")
    print("#" * 76)

    run_h2()
    print()
    run_lih()
    print()
    run_bench()

    total_time = time.perf_counter() - t_start
    print("#" * 76)
    print(f"# FULL PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS")
    print("#" * 76)


if __name__ == "__main__":
    main()
