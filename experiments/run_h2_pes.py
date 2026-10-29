"""
Experiment 1: Simulate the H2 Molecular Potential Energy Surface (PES) across
interatomic distances R in [0.35, 2.80] Angstroms using Hartree-Fock (HF),
Exact FCI, VQE (UCCSD), and VQE (Hardware-Efficient Ansatz).
Saves results to data/h2_pes_results.csv.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chemistry import get_default_bond_grid
from src.algorithms import run_pes_scan
from src.analysis import extract_spectroscopic_constants


def main() -> pd.DataFrame:
    print("=" * 72)
    print(" Running H2 Potential Energy Surface (PES) VQE Simulation (STO-3G)")
    print("=" * 72)
    t0 = time.perf_counter()

    bond_grid = get_default_bond_grid("H2")
    rows = run_pes_scan(
        molecule_name="H2",
        bond_lengths=bond_grid,
        mapping="jordan_wigner",
        ansatz_types=("UCCSD", "HEA"),
        optimizer_name="SLSQP",
        hea_layers=2,
        maxiter=120,
        warm_start=True,
    )

    df = pd.DataFrame(rows)
    out_dir = PROJECT_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "h2_pes_results.csv"
    df.to_csv(csv_path, index=False)

    spec_fci = extract_spectroscopic_constants(
        "H2", "Exact FCI", df["bond_length_A"], df["fci_energy_Ha"]
    )
    spec_ucc = extract_spectroscopic_constants(
        "H2", "VQE (UCCSD)", df["bond_length_A"], df["vqe_uccsd_energy_Ha"]
    )

    elapsed = time.perf_counter() - t0
    print(f"Saved {len(df)} geometries to {csv_path} in {elapsed:.2f}s")
    print(
        f"  Exact FCI   : R_e = {spec_fci.equilibrium_bond_length_A:.4f} A, "
        f"E_min = {spec_fci.minimum_energy_Ha:.6f} Ha, "
        f"D_e = {spec_fci.dissociation_energy_De_eV:.3f} eV, "
        f"omega_e = {spec_fci.harmonic_freq_omega_e_cm_inv:.1f} cm^-1"
    )
    print(
        f"  VQE (UCCSD) : R_e = {spec_ucc.equilibrium_bond_length_A:.4f} A, "
        f"E_min = {spec_ucc.minimum_energy_Ha:.6f} Ha, "
        f"D_e = {spec_ucc.dissociation_energy_De_eV:.3f} eV, "
        f"omega_e = {spec_ucc.harmonic_freq_omega_e_cm_inv:.1f} cm^-1"
    )
    print(
        f"  Max |E_UCCSD - E_FCI| across PES: {df['vqe_uccsd_error_Ha'].max():.3e} Ha "
        f"(Chemical accuracy achieved on {df['vqe_uccsd_chem_acc'].sum()}/{len(df)} points)"
    )
    return df


if __name__ == "__main__":
    main()
