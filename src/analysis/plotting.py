"""
Publication-quality scientific visualization module for VQE Potential Energy Surface (PES)
mapping, optimizer convergence trajectories, fermion-to-qubit mapping benchmarks,
and Morse vibrational spectroscopy.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.algorithms.vqe_runner import CHEMICAL_ACCURACY_HA
from .spectroscopy import (
    SpectroscopicConstants,
    compute_morse_vibrational_levels,
    extract_spectroscopic_constants,
    morse_potential,
)


# Clean scientific publication styling
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
    }
)


def plot_pes_comparison(
    df_pes: pd.DataFrame,
    molecule_Label: str,
    output_path: Path,
) -> None:
    """
    Generate a 2-panel publication figure for a diatomic molecule:
    - Top panel: Total energy E(R) for HF, VQE (HEA), VQE (UCCSD), Exact FCI, and Morse Fit.
    - Bottom panel: Log-scale absolute energy error |E(R) - E_FCI(R)| with shaded chemical accuracy.
    """
    R = df_pes["bond_length_A"].to_numpy(dtype=float)
    E_hf = df_pes["hf_energy_Ha"].to_numpy(dtype=float)
    E_fci = df_pes["fci_energy_Ha"].to_numpy(dtype=float)
    E_ucc = df_pes["vqe_uccsd_energy_Ha"].to_numpy(dtype=float)
    E_hea = df_pes["vqe_hea_energy_Ha"].to_numpy(dtype=float)

    err_hf = np.maximum(np.abs(E_hf - E_fci), 1e-12)
    err_hea = np.maximum(np.abs(E_hea - E_fci), 1e-12)
    err_ucc = np.maximum(np.abs(E_ucc - E_fci), 1e-12)

    spec_ucc = extract_spectroscopic_constants(
        df_pes["molecule"].iloc[0], "VQE (UCCSD)", R, E_ucc
    )
    r_dense = np.linspace(float(R.min()), float(R.max()), 300)
    e_morse_dense = morse_potential(
        r_dense,
        spec_ucc.minimum_energy_Ha,
        spec_ucc.dissociation_energy_De_Ha,
        spec_ucc.morse_width_a_inv_A,
        spec_ucc.equilibrium_bond_length_A,
    )

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.8, 7.6), sharex=True, gridspec_kw={"height_ratios": [1.55, 1.0]}
    )

    # Top panel: Potential Energy Curves
    ax_top.plot(
        R, E_hf, "--", color="#d95f02", lw=2.0, marker="^", ms=4.5, alpha=0.85,
        label="Hartree-Fock (Mean-Field RHF)"
    )
    ax_top.plot(
        R, E_fci, "-", color="#1b1b1b", lw=2.6, alpha=0.9,
        label="Exact FCI (Full Configuration Interaction)"
    )
    ax_top.plot(
        r_dense, e_morse_dense, ":", color="#66a61e", lw=1.7, alpha=0.85,
        label=f"Morse Fit ($R_e = {spec_ucc.equilibrium_bond_length_A:.3f}$ Å, $\\omega_e = {spec_ucc.harmonic_freq_omega_e_cm_inv:.0f}$ cm$^{{-1}}$)"
    )
    ax_top.plot(
        R, E_hea, "-.", color="#7570b3", lw=1.9, marker="s", ms=4.8, alpha=0.88,
        label="VQE — Hardware-Efficient Ansatz (HEA)"
    )
    ax_top.plot(
        R, E_ucc, "o", color="#e7298a", ms=6.2, markeredgecolor="white", markeredgewidth=0.7,
        label="VQE — Chemically Motivated UCCSD"
    )

    # Highlight equilibrium geometry R_e
    ax_top.axvline(
        spec_ucc.equilibrium_bond_length_A, color="#2b5c8f", ls="--", lw=1.2, alpha=0.7
    )
    ax_top.scatter(
        [spec_ucc.equilibrium_bond_length_A],
        [spec_ucc.minimum_energy_Ha],
        color="#e7298a",
        s=90,
        zorder=6,
        edgecolor="black",
        linewidth=1.1,
    )
    ax_top.annotate(
        f"Equilibrium $R_e = {spec_ucc.equilibrium_bond_length_A:.3f}$ Å\n"
        f"$E_0 = {spec_ucc.minimum_energy_Ha:.4f}$ Ha\n"
        f"$D_e = {spec_ucc.dissociation_energy_De_eV:.2f}$ eV",
        xy=(spec_ucc.equilibrium_bond_length_A, spec_ucc.minimum_energy_Ha),
        xytext=(
            spec_ucc.equilibrium_bond_length_A + 0.28 * (R.max() - R.min()),
            spec_ucc.minimum_energy_Ha + 0.15 * (E_hf.max() - spec_ucc.minimum_energy_Ha),
        ),
        arrowprops=dict(arrowstyle="->", color="#2b5c8f", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.35", fc="#f8f9fa", ec="#cbd5e1", alpha=0.95),
        fontsize=9.2,
    )

    ax_top.set_ylabel("Total Electronic Energy $E(R)$ [Hartree]")
    ax_top.set_title(
        f"{molecule_Label} Potential Energy Surface (STO-3G Basis, Jordan-Wigner Mapping)"
    )
    ax_top.legend(loc="upper right", frameon=True, framealpha=0.94)

    # Bottom panel: Log-scale Energy Error relative to Exact FCI
    ax_bot.semilogy(
        R, err_hf, "--^", color="#d95f02", lw=1.8, ms=4.5,
        label="$|E_{\\mathrm{HF}} - E_{\\mathrm{FCI}}|$ (Missing Electron Correlation)"
    )
    ax_bot.semilogy(
        R, err_hea, "-.s", color="#7570b3", lw=1.8, ms=4.8,
        label="$|E_{\\mathrm{VQE-HEA}} - E_{\\mathrm{FCI}}|$"
    )
    ax_bot.semilogy(
        R, err_ucc, "-o", color="#e7298a", lw=2.1, ms=5.2,
        label="$|E_{\\mathrm{VQE-UCCSD}} - E_{\\mathrm{FCI}}|$"
    )

    ax_bot.axhline(
        CHEMICAL_ACCURACY_HA, color="#1b9e77", ls="-", lw=1.6,
        label="Chemical Accuracy Threshold ($1.59 \\times 10^{-3}$ Ha)"
    )
    ax_bot.axhspan(1e-13, CHEMICAL_ACCURACY_HA, color="#1b9e77", alpha=0.14)

    ax_bot.set_xlabel("Interatomic Distance / Bond Length $R$ [Å]")
    ax_bot.set_ylabel("Absolute Error $|E - E_{\\mathrm{FCI}}|$ [Ha]")
    ax_bot.set_ylim(bottom=max(min(err_ucc.min() * 0.2, 1e-11), 1e-13), top=1.0)
    ax_bot.legend(loc="lower right", frameon=True, framealpha=0.94, ncol=2, fontsize=8.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_optimizer_convergence(
    df_conv: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate a 2x2 multi-panel figure comparing classical optimizers (SLSQP, L-BFGS-B, COBYLA, SPSA)
    across equilibrium and stretched dissociation geometries.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    palette = {
        "SLSQP": ("#e7298a", "-"),
        "L-BFGS-B": ("#1b9e77", "--"),
        "COBYLA": ("#d95f02", "-."),
        "SPSA": ("#7570b3", ":"),
    }

    configs = [
        ("H2", "Equilibrium", axes[0, 0], "$\\mathrm{H}_2$ at Equilibrium ($R = 0.735$ Å)"),
        ("H2", "Stretched", axes[0, 1], "$\\mathrm{H}_2$ at Stretched Bond ($R = 2.000$ Å)"),
        ("LiH", "Equilibrium", axes[1, 0], "$\\mathrm{LiH}$ at Equilibrium ($R = 1.546$ Å)"),
    ]

    for mol, geom_tag, ax, title in configs:
        sub = df_conv[(df_conv["molecule"] == mol) & (df_conv["geometry_tag"] == geom_tag)]
        for opt_name, group in sub.groupby("optimizer"):
            color, ls = palette.get(str(opt_name), ("#333333", "-"))
            steps = group["step"].to_numpy(dtype=int)
            err = np.maximum(group["abs_error_Ha"].to_numpy(dtype=float), 1e-12)
            ax.semilogy(steps, err, ls=ls, color=color, lw=2.1, label=str(opt_name))

        ax.axhline(
            CHEMICAL_ACCURACY_HA, color="#2b8cbe", ls="--", lw=1.3,
            label="Chemical Accuracy ($1.59$ mHa)"
        )
        ax.axhspan(1e-14, CHEMICAL_ACCURACY_HA, color="#2b8cbe", alpha=0.11)
        ax.set_title(title)
        ax.set_xlabel("Function Evaluation Step $k$")
        ax.set_ylabel("Energy Error $|E_k - E_{\\mathrm{FCI}}|$ [Ha]")
        ax.set_ylim(1e-13, 1.0)
        ax.legend(loc="upper right", frameon=True)

    # Panel (d): Final Error vs Function Evaluations summary bar chart
    ax_bar = axes[1, 1]
    summary_rows = []
    for (mol, geom_tag, opt_name), group in df_conv.groupby(["molecule", "geometry_tag", "optimizer"]):
        final_err = max(float(group["abs_error_Ha"].iloc[-1]), 1e-12)
        total_fev = int(group["step"].max())
        summary_rows.append(
            {
                "label": f"{mol}\n({geom_tag[:4]})",
                "optimizer": str(opt_name),
                "final_err": final_err,
                "total_fev": total_fev,
            }
        )
    df_sum = pd.DataFrame(summary_rows)
    labels = sorted(df_sum["label"].unique())
    opts = ["SLSQP", "L-BFGS-B", "COBYLA", "SPSA"]
    x = np.arange(len(labels))
    width = 0.19

    for idx, opt in enumerate(opts):
        vals = []
        for lbl in labels:
            m = df_sum[(df_sum["label"] == lbl) & (df_sum["optimizer"] == opt)]
            vals.append(float(m["final_err"].iloc[0]) if len(m) > 0 else 1e-12)
        color, _ = palette.get(opt, ("#333333", "-"))
        ax_bar.bar(x + (idx - 1.5) * width, vals, width, label=opt, color=color, alpha=0.88)

    ax_bar.set_yscale("log")
    ax_bar.axhline(CHEMICAL_ACCURACY_HA, color="#2b8cbe", ls="--", lw=1.3)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylabel("Final Converged Error $|E^* - E_{\\mathrm{FCI}}|$ [Ha]")
    ax_bar.set_title("Final Accuracy Across Classical Optimizers")
    ax_bar.set_ylim(1e-14, 1.0)
    ax_bar.legend(loc="upper left", ncol=2, frameon=True)

    fig.suptitle(
        "Classical Optimizer Convergence Benchmarks for Hybrid VQE (UCCSD Ansatz)",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_mapping_and_ansatz_benchmark(
    df_bench: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate a 2x2 visual benchmark of fermion-to-qubit mappings (JW, Parity, Parity-2Q, BK)
    and circuit ansatze (UCCSD vs HEA L=1,2,3).
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    # Filter mapping rows (where ansatz == 'UCCSD')
    df_map = df_bench[df_bench["benchmark_category"] == "mapping"].copy()
    df_ans = df_bench[df_bench["benchmark_category"] == "ansatz"].copy()

    # Panel (a): Qubit Count & Pauli Term Count by Mapping
    ax_a = axes[0, 0]
    mappings_order = ["jordan_wigner", "parity", "parity_2q", "bravyi_kitaev"]
    map_labels = ["Jordan-Wigner", "Parity", "Parity (2Q Red.)", "Bravyi-Kitaev"]
    x = np.arange(len(mappings_order))
    width = 0.35

    h2_terms = [
        int(df_map[(df_map["molecule"] == "H2") & (df_map["mapping"] == m)]["n_pauli_terms"].iloc[0])
        for m in mappings_order
    ]
    lih_terms = [
        int(df_map[(df_map["molecule"] == "LiH") & (df_map["mapping"] == m)]["n_pauli_terms"].iloc[0])
        for m in mappings_order
    ]
    h2_qubits = [
        int(df_map[(df_map["molecule"] == "H2") & (df_map["mapping"] == m)]["n_qubits"].iloc[0])
        for m in mappings_order
    ]
    lih_qubits = [
        int(df_map[(df_map["molecule"] == "LiH") & (df_map["mapping"] == m)]["n_qubits"].iloc[0])
        for m in mappings_order
    ]

    bars1 = ax_a.bar(x - width / 2, h2_terms, width, label="$\\mathrm{H}_2$ Pauli Terms", color="#2b8cbe")
    bars2 = ax_a.bar(x + width / 2, lih_terms, width, label="$\\mathrm{LiH}$ Pauli Terms", color="#e7298a")

    for bar, nq in zip(bars1, h2_qubits):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{nq}Q",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    for bar, nq in zip(bars2, lih_qubits):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{nq}Q",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(map_labels, rotation=12)
    ax_a.set_ylabel("Number of Pauli Terms in $\\hat{H}$")
    ax_a.set_title("(a) Fermion-to-Qubit Mapping Overhead (Qubits & Pauli Terms)")
    ax_a.set_ylim(0, max(lih_terms) * 1.32)
    ax_a.legend(loc="upper right", ncol=2)

    # Panel (b): Mean Pauli Weight (Locality) across Mappings
    ax_b = axes[0, 1]
    h2_weight = [
        float(df_map[(df_map["molecule"] == "H2") & (df_map["mapping"] == m)]["mean_pauli_weight"].iloc[0])
        for m in mappings_order
    ]
    lih_weight = [
        float(df_map[(df_map["molecule"] == "LiH") & (df_map["mapping"] == m)]["mean_pauli_weight"].iloc[0])
        for m in mappings_order
    ]
    ax_b.bar(x - width / 2, h2_weight, width, label="$\\mathrm{H}_2$ Mean Pauli Weight", color="#66a61e")
    ax_b.bar(x + width / 2, lih_weight, width, label="$\\mathrm{LiH}$ Mean Pauli Weight", color="#d95f02")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(map_labels, rotation=12)
    ax_b.set_ylabel("Mean Pauli String Weight (Locality)")
    ax_b.set_title("(b) Operator Locality Across Fermion-to-Qubit Mappings")
    ax_b.set_ylim(0, max(lih_weight) * 1.25)
    ax_b.legend(loc="upper right")

    # Panel (c): Ansatz Accuracy (UCCSD vs HEA L=1, 2, 3)
    ax_c = axes[1, 0]
    ans_labels = ["HEA (L=1)", "HEA (L=2)", "HEA (L=3)", "UCCSD"]
    x_ans = np.arange(len(ans_labels))
    h2_err = [
        max(float(df_ans[(df_ans["molecule"] == "H2") & (df_ans["ansatz_label"] == a)]["abs_error_Ha"].iloc[0]), 1e-12)
        for a in ans_labels
    ]
    lih_err = [
        max(float(df_ans[(df_ans["molecule"] == "LiH") & (df_ans["ansatz_label"] == a)]["abs_error_Ha"].iloc[0]), 1e-12)
        for a in ans_labels
    ]
    ax_c.bar(x_ans - width / 2, h2_err, width, label="$\\mathrm{H}_2$ $|E - E_{\\mathrm{FCI}}|$", color="#2b8cbe")
    ax_c.bar(x_ans + width / 2, lih_err, width, label="$\\mathrm{LiH}$ $|E - E_{\\mathrm{FCI}}|$", color="#e7298a")
    ax_c.set_yscale("log")
    ax_c.axhline(CHEMICAL_ACCURACY_HA, color="#1b9e77", ls="--", lw=1.4, label="Chemical Accuracy")
    ax_c.axhspan(1e-14, CHEMICAL_ACCURACY_HA, color="#1b9e77", alpha=0.11)
    ax_c.set_xticks(x_ans)
    ax_c.set_xticklabels(ans_labels, rotation=12)
    ax_c.set_ylabel("Energy Error $|E - E_{\\mathrm{FCI}}|$ [Ha]")
    ax_c.set_title("(c) Ansatz Expressibility: UCCSD vs Hardware-Efficient (HEA)")
    ax_c.set_ylim(1e-14, 1.0)
    ax_c.legend(loc="upper right")

    # Panel (d): Two-Qubit CNOT Gate Count vs Variational Parameters
    ax_d = axes[1, 1]
    h2_cnots = [
        int(df_ans[(df_ans["molecule"] == "H2") & (df_ans["ansatz_label"] == a)]["n_cnots"].iloc[0])
        for a in ans_labels
    ]
    lih_cnots = [
        int(df_ans[(df_ans["molecule"] == "LiH") & (df_ans["ansatz_label"] == a)]["n_cnots"].iloc[0])
        for a in ans_labels
    ]
    ax_d.bar(x_ans - width / 2, h2_cnots, width, label="$\\mathrm{H}_2$ Two-Qubit Gates", color="#7570b3")
    ax_d.bar(x_ans + width / 2, lih_cnots, width, label="$\\mathrm{LiH}$ Two-Qubit Gates", color="#e6ab02")
    ax_d.set_xticks(x_ans)
    ax_d.set_xticklabels(ans_labels, rotation=12)
    ax_d.set_ylabel("Two-Qubit Gate (CNOT) Count")
    ax_d.set_title("(d) Quantum Circuit Resource Cost Across Ansätze")
    ax_d.legend(loc="upper left")

    fig.suptitle(
        "Fermion-to-Qubit Mappings & Parameterized Quantum Circuit Ansatz Benchmarks",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_morse_spectroscopy_fit(
    df_h2: pd.DataFrame,
    df_lih: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate a side-by-side figure of the Morse potential oscillator fits for H2 and LiH,
    illustrating quantized vibrational levels (nu = 0, 1, 2, 3, 4), zero-point energy (ZPE),
    equilibrium bond length R_e, harmonic frequency omega_e, and dissociation energy D_e.
    """
    fig, (ax_h2, ax_lih) = plt.subplots(1, 2, figsize=(12.2, 5.4))

    datasets = [
        (df_h2, "H2", "$\\mathrm{H}_2$ Vibrational Spectroscopy & Morse Potential", ax_h2, "#e7298a"),
        (df_lih, "LiH", "$\\mathrm{LiH}$ Vibrational Spectroscopy & Morse Potential", ax_lih, "#2b8cbe"),
    ]

    for df, mol_name, title, ax, color in datasets:
        R = df["bond_length_A"].to_numpy(dtype=float)
        E_ucc = df["vqe_uccsd_energy_Ha"].to_numpy(dtype=float)
        E_fci = df["fci_energy_Ha"].to_numpy(dtype=float)

        spec = extract_spectroscopic_constants(mol_name, "VQE (UCCSD)", R, E_ucc)
        levels = compute_morse_vibrational_levels(spec, max_levels=5)

        r_grid = np.linspace(float(R.min()), float(R.max()), 350)
        v_morse = morse_potential(
            r_grid,
            spec.minimum_energy_Ha,
            spec.dissociation_energy_De_Ha,
            spec.morse_width_a_inv_A,
            spec.equilibrium_bond_length_A,
        )

        ax.plot(r_grid, v_morse, "-", color=color, lw=2.3, label="Analytical Morse Fit $V(R)$")
        ax.plot(R, E_fci, "--", color="#222222", lw=1.5, alpha=0.75, label="Exact FCI Reference")
        ax.scatter(
            R, E_ucc, color=color, edgecolor="white", s=42, zorder=5, label="VQE (UCCSD) Data"
        )

        # Draw quantized vibrational energy levels inside the Morse well
        for nu, e_lvl, r_in, r_out in levels:
            r_in_clip = max(r_in, float(R.min()))
            r_out_clip = min(r_out, float(R.max()))
            ax.hlines(
                e_lvl,
                r_in_clip,
                r_out_clip,
                colors="#1b9e77" if nu == 0 else "#555555",
                linestyles="-" if nu == 0 else "--",
                lw=2.0 if nu == 0 else 1.2,
                label="Zero-Point Level ($\\nu=0$)" if nu == 0 else None,
            )
            ax.text(
                r_out_clip + 0.03 * (R.max() - R.min()),
                e_lvl,
                f"$\\nu={nu}$",
                va="center",
                fontsize=8.2,
                color="#1b9e77" if nu == 0 else "#444444",
            )

        # Asymptotic dissociation limit line
        e_dissoc = spec.minimum_energy_Ha + spec.dissociation_energy_De_Ha
        ax.axhline(
            e_dissoc, color="#d95f02", ls=":", lw=1.4,
            label=f"Dissociation Limit ($E_{{\\min}} + D_e$)"
        )

        info_box = (
            f"$R_e = {spec.equilibrium_bond_length_A:.3f}$ Å\n"
            f"$E(R_e) = {spec.minimum_energy_Ha:.4f}$ Ha\n"
            f"$D_e = {spec.dissociation_energy_De_Ha:.4f}$ Ha ({spec.dissociation_energy_De_eV:.2f} eV)\n"
            f"$D_0 = {spec.dissociation_energy_D0_eV:.2f}$ eV\n"
            f"$\\omega_e = {spec.harmonic_freq_omega_e_cm_inv:.1f}$ cm$^{{-1}}$\n"
            f"$\\mathrm{{ZPE}} = {spec.zero_point_energy_cm_inv:.1f}$ cm$^{{-1}}$\n"
            f"Fit $R^2 = {spec.r_squared:.4f}$"
        )
        ax.text(
            0.97,
            0.27,
            info_box,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.38", fc="#f8f9fa", ec="#cbd5e1", alpha=0.94),
        )

        ax.set_title(title)
        ax.set_xlabel("Interatomic Distance $R$ [Å]")
        ax.set_ylabel("Potential Energy $V(R)$ [Hartree]")
        y_span = e_dissoc - spec.minimum_energy_Ha
        ax.set_ylim(spec.minimum_energy_Ha - 0.08 * y_span, e_dissoc + 0.25 * y_span)
        ax.legend(loc="upper right", frameon=True, fontsize=8.6)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(
    data_dir: Path,
    figures_dir: Path,
) -> List[Path]:
    """
    Generate all 5 publication-ready figures from the CSV datasets in data_dir.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    df_h2 = pd.read_csv(data_dir / "h2_pes_results.csv")
    p_h2 = figures_dir / "h2_pes_comparison.png"
    plot_pes_comparison(df_h2, "$\\mathrm{H}_2$", p_h2)
    generated.append(p_h2)

    df_lih = pd.read_csv(data_dir / "lih_pes_results.csv")
    p_lih = figures_dir / "lih_pes_comparison.png"
    plot_pes_comparison(df_lih, "$\\mathrm{LiH}$", p_lih)
    generated.append(p_lih)

    df_conv = pd.read_csv(data_dir / "optimizer_convergence.csv")
    p_conv = figures_dir / "optimizer_convergence.png"
    plot_optimizer_convergence(df_conv, p_conv)
    generated.append(p_conv)

    df_bench = pd.read_csv(data_dir / "mapping_ansatz_benchmarks.csv")
    p_bench = figures_dir / "mapping_and_ansatz_benchmark.png"
    plot_mapping_and_ansatz_benchmark(df_bench, p_bench)
    generated.append(p_bench)

    p_morse = figures_dir / "morse_spectroscopy_fit.png"
    plot_morse_spectroscopy_fit(df_h2, df_lih, p_morse)
    generated.append(p_morse)

    return generated
