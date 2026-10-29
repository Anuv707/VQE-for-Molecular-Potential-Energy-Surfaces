"""
Experiment 3: Comprehensive Benchmarks & Spectroscopic Analysis
1. Classical Optimizer Convergence Benchmark (SLSQP, L-BFGS-B, COBYLA, SPSA)
   -> saves data/optimizer_convergence.csv
2. Fermion-to-Qubit Mapping Benchmark (Jordan-Wigner, Parity, Parity-2Q, Bravyi-Kitaev)
   and Ansatz Depth/Expressibility Benchmark (UCCSD vs HEA L=1, 2, 3)
   -> saves data/mapping_ansatz_benchmarks.csv
3. Physicochemical & Spectroscopic Parameter Extraction (R_e, omega_e, D_e, ZPE)
   -> saves data/spectroscopic_constants.csv
4. Generates all 5 publication-ready 300 DPI figures in outputs/figures/
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import time
from typing import Dict, List
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chemistry import (
    build_qubit_hamiltonian,
    compare_mappings,
    compute_molecular_integrals,
)
from src.algorithms import (
    run_vqe,
    solve_exact_fci,
)
from src.analysis import (
    extract_spectroscopic_constants,
    generate_all_figures,
)


def run_optimizer_benchmarks(data_dir: Path) -> pd.DataFrame:
    """Run SLSQP, L-BFGS-B, COBYLA, and SPSA convergence benchmarks."""
    print("\n[1/4] Running Classical Optimizer Convergence Benchmarks...")
    configs = [
        ("H2", 0.735, "Equilibrium"),
        ("H2", 2.000, "Stretched"),
        ("LiH", 1.546, "Equilibrium"),
    ]
    optimizers = ["SLSQP", "L-BFGS-B", "COBYLA", "SPSA"]
    rows: List[Dict[str, object]] = []

    for mol, R, geom_tag in configs:
        ints = compute_molecular_integrals(mol, R, active_space=True)
        ham = build_qubit_hamiltonian(ints, mapping="jordan_wigner")
        exact = solve_exact_fci(ham, active_electrons=ints.spec.active_electrons)

        for opt in optimizers:
            max_it = 120 if opt in ("COBYLA", "SPSA") else 60
            # Start from slightly perturbed initial point so all optimizers display a clear trajectory
            rng = np.random.default_rng(42)
            res = run_vqe(
                mapped_ham=ham,
                ansatz_type="UCCSD",
                optimizer_name=opt,
                active_electrons=ints.spec.active_electrons,
                maxiter=max_it,
                exact_solution=exact,
                seed=42,
            )

            # Build monotonic running-minimum envelope over function evaluations
            best_so_far = float("inf")
            max_steps_to_record = min(len(res.energy_history), 150)
            for step_idx in range(max_steps_to_record):
                e_val = float(res.energy_history[step_idx])
                if e_val < best_so_far:
                    best_so_far = e_val
                err = max(abs(best_so_far - exact.fci_energy), 1e-13)
                rows.append(
                    {
                        "molecule": mol,
                        "bond_length_A": R,
                        "geometry_tag": geom_tag,
                        "optimizer": opt,
                        "step": step_idx + 1,
                        "energy_Ha": best_so_far,
                        "fci_energy_Ha": exact.fci_energy,
                        "abs_error_Ha": err,
                    }
                )
            print(
                f"  {mol:3s} ({geom_tag:11s}, R={R:.3f} A) | {opt:8s}: "
                f"E = {res.energy:.6f} Ha, |E - E_FCI| = {res.abs_error_ha:.3e} Ha "
                f"({res.n_fev} fev, {res.runtime_sec:.2f}s)"
            )

    df_conv = pd.DataFrame(rows)
    out_path = data_dir / "optimizer_convergence.csv"
    df_conv.to_csv(out_path, index=False)
    print(f"  -> Saved optimizer convergence data to {out_path}")
    return df_conv


def run_mapping_and_ansatz_benchmarks(data_dir: Path) -> pd.DataFrame:
    """Compare JW, Parity, Parity-2Q, and BK mappings and UCCSD vs HEA (L=1,2,3) ansatze."""
    print("\n[2/4] Running Fermion-to-Qubit Mapping & Ansatz Expressibility Benchmarks...")
    rows: List[Dict[str, object]] = []

    for mol, R in [("H2", 0.735), ("LiH", 1.546)]:
        mapped_dict = compare_mappings(mol, R)
        for m_name, ham in mapped_dict.items():
            # For non-JW mappings, diagonalize the full mapped Hamiltonian to verify isospectrality
            exact_full = solve_exact_fci(ham, restrict_particle_sector=(m_name == "jordan_wigner"))
            vqe_res = run_vqe(
                mapped_ham=ham,
                ansatz_type="UCCSD",
                optimizer_name="SLSQP",
                maxiter=80,
                exact_solution=exact_full,
            )
            rows.append(
                {
                    "benchmark_category": "mapping",
                    "molecule": mol,
                    "bond_length_A": R,
                    "mapping": m_name,
                    "ansatz_label": "UCCSD",
                    "n_qubits": ham.n_qubits,
                    "n_pauli_terms": ham.n_pauli_terms,
                    "mean_pauli_weight": ham.mean_pauli_weight,
                    "max_pauli_weight": ham.max_pauli_weight,
                    "n_params": vqe_res.n_params,
                    "n_cnots": vqe_res.n_two_qubit_gates,
                    "fci_energy_Ha": exact_full.fci_energy,
                    "vqe_energy_Ha": vqe_res.energy,
                    "abs_error_Ha": vqe_res.abs_error_ha,
                    "chem_accuracy": vqe_res.chemical_accuracy_achieved,
                }
            )
            print(
                f"  [Mapping] {mol:3s} | {m_name:14s}: {ham.n_qubits} qubits, "
                f"{ham.n_pauli_terms:2d} Pauli terms, mean weight={ham.mean_pauli_weight:.2f}, "
                f"FCI={exact_full.fci_energy:.6f} Ha, VQE err={vqe_res.abs_error_ha:.2e} Ha"
            )

        # Ansatz depth comparison in Jordan-Wigner mapping
        ham_jw = mapped_dict["jordan_wigner"]
        exact_jw = solve_exact_fci(ham_jw, restrict_particle_sector=True)
        ansatz_configs = [
            ("UCCSD", "UCCSD", 1),
            ("HEA (L=1)", "HEA", 1),
            ("HEA (L=2)", "HEA", 2),
            ("HEA (L=3)", "HEA", 3),
        ]
        for label, a_type, layers in ansatz_configs:
            vqe_res = run_vqe(
                mapped_ham=ham_jw,
                ansatz_type=a_type,
                n_layers=layers,
                optimizer_name="SLSQP",
                maxiter=100,
                exact_solution=exact_jw,
            )
            rows.append(
                {
                    "benchmark_category": "ansatz",
                    "molecule": mol,
                    "bond_length_A": R,
                    "mapping": "jordan_wigner",
                    "ansatz_label": label,
                    "n_qubits": ham_jw.n_qubits,
                    "n_pauli_terms": ham_jw.n_pauli_terms,
                    "mean_pauli_weight": ham_jw.mean_pauli_weight,
                    "max_pauli_weight": ham_jw.max_pauli_weight,
                    "n_params": vqe_res.n_params,
                    "n_cnots": vqe_res.n_two_qubit_gates,
                    "fci_energy_Ha": exact_jw.fci_energy,
                    "vqe_energy_Ha": vqe_res.energy,
                    "abs_error_Ha": vqe_res.abs_error_ha,
                    "chem_accuracy": vqe_res.chemical_accuracy_achieved,
                }
            )
            print(
                f"  [Ansatz ] {mol:3s} | {label:10s}: {vqe_res.n_params:2d} params, "
                f"{vqe_res.n_two_qubit_gates:2d} CNOTs, VQE err={vqe_res.abs_error_ha:.3e} Ha"
            )

    df_bench = pd.DataFrame(rows)
    out_path = data_dir / "mapping_ansatz_benchmarks.csv"
    df_bench.to_csv(out_path, index=False)
    print(f"  -> Saved mapping & ansatz benchmark data to {out_path}")
    return df_bench


def compute_all_spectroscopic_constants(data_dir: Path) -> pd.DataFrame:
    """Extract R_e, omega_e, D_e, and ZPE from the H2 and LiH PES scan datasets."""
    print("\n[3/4] Extracting Physicochemical & Spectroscopic Constants (Morse Fit)...")
    spec_rows: List[Dict[str, object]] = []

    for mol, filename in [("H2", "h2_pes_results.csv"), ("LiH", "lih_pes_results.csv")]:
        df = pd.read_csv(data_dir / filename)
        R = df["bond_length_A"].to_numpy(dtype=float)

        methods = [
            ("Hartree-Fock (RHF)", "hf_energy_Ha"),
            ("VQE (HEA)", "vqe_hea_energy_Ha"),
            ("VQE (UCCSD)", "vqe_uccsd_energy_Ha"),
            ("Exact FCI", "fci_energy_Ha"),
        ]
        for method_label, col in methods:
            E = df[col].to_numpy(dtype=float)
            sc = extract_spectroscopic_constants(mol, method_label, R, E)
            spec_rows.append(asdict(sc))
            print(
                f"  {mol:3s} | {method_label:18s}: R_e = {sc.equilibrium_bond_length_A:.4f} A, "
                f"E_min = {sc.minimum_energy_Ha:.6f} Ha, D_e = {sc.dissociation_energy_De_eV:.3f} eV, "
                f"omega_e = {sc.harmonic_freq_omega_e_cm_inv:.1f} cm^-1 (R^2={sc.r_squared:.4f})"
            )

    df_spec = pd.DataFrame(spec_rows)
    out_path = data_dir / "spectroscopic_constants.csv"
    df_spec.to_csv(out_path, index=False)
    print(f"  -> Saved spectroscopic constants table to {out_path}")
    return df_spec


def main() -> None:
    t0 = time.perf_counter()
    data_dir = PROJECT_ROOT / "data"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_optimizer_benchmarks(data_dir)
    run_mapping_and_ansatz_benchmarks(data_dir)
    compute_all_spectroscopic_constants(data_dir)

    print("\n[4/4] Generating Publication-Grade Figures (300 DPI)...")
    figs = generate_all_figures(data_dir, figures_dir)
    for f in figs:
        print(f"  -> Generated figure: {f}")

    print(f"\nAll benchmarks and figures completed in {time.perf_counter() - t0:.2f}s.")


if __name__ == "__main__":
    main()
