"""
Physicochemical and spectroscopic parameter extraction from Potential Energy Surfaces (PES):
- Analytical Morse Potential nonlinear least-squares fitting:
    V(R) = E_min + D_e * (1 - exp(-a * (R - R_e)))^2
- Extraction of equilibrium bond length (R_e), minimum electronic energy (E_min),
  spectroscopic dissociation energy (D_e in Ha, eV, kcal/mol), harmonic force constant (k_e),
  harmonic vibrational frequency (omega_e in cm^-1), anharmonicity (omega_e * x_e),
  and zero-point vibrational energy (ZPE).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np
import scipy.optimize

from src.chemistry.molecules import MASS_H1_AMU, MASS_LI7_AMU


# CODATA 2018 / 2022 physical constants
HARTREE_TO_JOULE = 4.3597447222071e-18
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL_MOL = 627.5094740631
HARTREE_TO_CM_INV = 219474.63136320
AMU_TO_KG = 1.66053906660e-27
ANGSTROM_TO_METER = 1.0e-10
SPEED_OF_LIGHT_CM_S = 2.99792458e10


@dataclass
class SpectroscopicConstants:
    """Extracted physicochemical and spectroscopic parameters from a diatomic PES curve."""

    molecule_name: str
    method_name: str
    equilibrium_bond_length_A: float
    minimum_energy_Ha: float
    dissociation_energy_De_Ha: float
    dissociation_energy_De_eV: float
    dissociation_energy_De_kcal_mol: float
    asymptotic_dissociation_Ha: float
    morse_width_a_inv_A: float
    force_constant_Ha_per_A2: float
    force_constant_N_per_m: float
    harmonic_freq_omega_e_cm_inv: float
    anharmonicity_omega_e_xe_cm_inv: float
    zero_point_energy_Ha: float
    zero_point_energy_cm_inv: float
    dissociation_energy_D0_eV: float
    fit_rmse_Ha: float
    r_squared: float


def get_reduced_mass_amu(molecule_name: str) -> float:
    """Return the isotopic nuclear reduced mass mu = m1*m2/(m1+m2) in unified atomic mass units (u)."""
    mol_upper = molecule_name.strip().upper()
    if mol_upper == "H2":
        return (MASS_H1_AMU * MASS_H1_AMU) / (MASS_H1_AMU + MASS_H1_AMU)
    elif mol_upper == "LIH":
        return (MASS_LI7_AMU * MASS_H1_AMU) / (MASS_LI7_AMU + MASS_H1_AMU)
    else:
        raise ValueError(f"Unsupported molecule '{molecule_name}'. Supported: 'H2', 'LiH'.")


def morse_potential(
    R: np.ndarray, E_min: float, D_e: float, a: float, R_e: float
) -> np.ndarray:
    """
    Evaluate the Morse potential energy function:
        V(R) = E_min + D_e * (1 - exp(-a * (R - R_e)))^2
    """
    r_arr = np.asarray(R, dtype=float)
    exponent = np.clip(-a * (r_arr - R_e), -50.0, 50.0)
    return E_min + D_e * ((1.0 - np.exp(exponent)) ** 2)


def extract_spectroscopic_constants(
    molecule_name: str,
    method_name: str,
    bond_lengths_A: Sequence[float],
    energies_Ha: Sequence[float],
) -> SpectroscopicConstants:
    """
    Fit a computed potential energy curve E(R) to a Morse oscillator potential
    combined with local quartic polynomial refinement around the equilibrium well
    to extract R_e, E_min, D_e, k_e, and omega_e.
    """
    R = np.asarray(bond_lengths_A, dtype=float)
    E = np.asarray(energies_Ha, dtype=float)

    if len(R) != len(E) or len(R) < 5:
        raise ValueError("Require at least 5 matching (R, E) points to extract spectroscopy.")

    sort_idx = np.argsort(R)
    R = R[sort_idx]
    E = E[sort_idx]

    min_idx = int(np.argmin(E))
    r_e_guess = float(R[min_idx])
    e_min_guess = float(E[min_idx])
    e_asymp = float(E[-1])
    d_e_guess = max(e_asymp - e_min_guess, 0.05)
    a_guess = 1.8 if molecule_name.strip().upper() == "H2" else 1.15

    # Focus Morse fit on points from slightly left of R_e out to the dissociation tail
    # so the steep R^-12 repulsive wall at very small R does not skew the harmonic well
    fit_mask = R >= max(R[0], 0.60 * r_e_guess)
    R_fit = R[fit_mask] if np.sum(fit_mask) >= 5 else R
    E_fit = E[fit_mask] if np.sum(fit_mask) >= 5 else E

    # Weight points near the minimum higher for accurate R_e and curvature k_e
    sigma_weights = 1.0 + 3.0 * ((R_fit - r_e_guess) ** 2)

    bounds_lower = [e_min_guess - 0.5, 0.01, 0.2, float(R[0])]
    bounds_upper = [e_min_guess + 0.1, 2.00, 5.0, float(R[-1])]

    popt, _ = scipy.optimize.curve_fit(
        morse_potential,
        R_fit,
        E_fit,
        p0=[e_min_guess, d_e_guess, a_guess, r_e_guess],
        sigma=sigma_weights,
        bounds=(bounds_lower, bounds_upper),
        maxfev=10000,
    )
    e_min_morse, d_e_morse, a_morse, r_e_morse = [float(x) for x in popt]

    # Local polynomial refinement around the minimum (nearest 7 points) for high-precision R_e & k_e
    win_start = max(0, min_idx - 3)
    win_end = min(len(R), win_start + 7)
    win_start = max(0, win_end - 7)
    R_loc = R[win_start:win_end]
    E_loc = E[win_start:win_end]

    if len(R_loc) >= 5:
        # Fit degree-3 or degree-4 polynomial in (R - r_e_morse)
        deg = 4 if len(R_loc) >= 6 else 3
        poly_coeffs = np.polyfit(R_loc, E_loc, deg=deg)
        poly_fn = np.poly1d(poly_coeffs)
        poly_d1 = np.polyder(poly_fn, 1)
        poly_d2 = np.polyder(poly_fn, 2)

        res_min = scipy.optimize.minimize_scalar(
            poly_fn, bounds=(float(R_loc[0]), float(R_loc[-1])), method="bounded"
        )
        if res_min.success and float(poly_d2(res_min.x)) > 0:
            r_e_final = float(res_min.x)
            e_min_final = float(poly_fn(r_e_final))
            k_e_poly = float(poly_d2(r_e_final))
        else:
            r_e_final = r_e_morse
            e_min_final = e_min_morse
            k_e_poly = 2.0 * (a_morse ** 2) * d_e_morse
    else:
        r_e_final = r_e_morse
        e_min_final = e_min_morse
        k_e_poly = 2.0 * (a_morse ** 2) * d_e_morse

    # Combine Morse global fit with local curvature
    k_e_morse = 2.0 * (a_morse ** 2) * d_e_morse
    k_e_ha_A2 = 0.5 * (k_e_morse + k_e_poly) if k_e_poly > 0 else k_e_morse
    a_eff = float(np.sqrt(k_e_ha_A2 / (2.0 * max(d_e_morse, 1e-6))))

    # Convert force constant to SI (N/m = J/m^2)
    k_e_N_m = k_e_ha_A2 * (HARTREE_TO_JOULE / (ANGSTROM_TO_METER ** 2))
    mu_kg = get_reduced_mass_amu(molecule_name) * AMU_TO_KG

    # Harmonic vibrational wavenumber omega_e = (1 / (2 * pi * c)) * sqrt(k_e / mu) in cm^-1
    omega_e_cm_inv = float(
        (1.0 / (2.0 * np.pi * SPEED_OF_LIGHT_CM_S)) * np.sqrt(max(k_e_N_m, 0.0) / mu_kg)
    )

    # Morse anharmonicity constant omega_e * x_e = omega_e^2 / (4 * D_e_cm_inv)
    d_e_cm_inv = d_e_morse * HARTREE_TO_CM_INV
    omega_e_xe_cm_inv = float((omega_e_cm_inv ** 2) / (4.0 * max(d_e_cm_inv, 1.0)))

    # Zero-point vibrational energy ZPE (nu = 0): E(0) = 0.5 * omega_e - 0.25 * omega_e * x_e
    zpe_cm_inv = max(0.5 * omega_e_cm_inv - 0.25 * omega_e_xe_cm_inv, 0.0)
    zpe_ha = zpe_cm_inv / HARTREE_TO_CM_INV

    d_e_ev = d_e_morse * HARTREE_TO_EV
    d_e_kcal = d_e_morse * HARTREE_TO_KCAL_MOL
    d_0_ev = max(d_e_morse - zpe_ha, 0.0) * HARTREE_TO_EV
    asymp_de_ha = max(e_asymp - e_min_final, 0.0)

    # Fit quality metrics over R_fit
    e_pred = morse_potential(R_fit, e_min_morse, d_e_morse, a_morse, r_e_morse)
    residuals = E_fit - e_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((E_fit - np.mean(E_fit)) ** 2))
    r2 = 1.0 - (ss_res / max(ss_tot, 1e-15))

    return SpectroscopicConstants(
        molecule_name=molecule_name.strip(),
        method_name=method_name.strip(),
        equilibrium_bond_length_A=r_e_final,
        minimum_energy_Ha=e_min_final,
        dissociation_energy_De_Ha=d_e_morse,
        dissociation_energy_De_eV=d_e_ev,
        dissociation_energy_De_kcal_mol=d_e_kcal,
        asymptotic_dissociation_Ha=asymp_de_ha,
        morse_width_a_inv_A=a_eff,
        force_constant_Ha_per_A2=k_e_ha_A2,
        force_constant_N_per_m=k_e_N_m,
        harmonic_freq_omega_e_cm_inv=omega_e_cm_inv,
        anharmonicity_omega_e_xe_cm_inv=omega_e_xe_cm_inv,
        zero_point_energy_Ha=zpe_ha,
        zero_point_energy_cm_inv=zpe_cm_inv,
        dissociation_energy_D0_eV=d_0_ev,
        fit_rmse_Ha=rmse,
        r_squared=r2,
    )


def compute_morse_vibrational_levels(
    spec_const: SpectroscopicConstants, max_levels: int = 5
) -> List[Tuple[int, float, float, float]]:
    """
    Compute quantized vibrational energy levels E_nu (in Ha) and classical turning points
    (R_inner, R_outer in Angstroms) for a fitted Morse oscillator:
        G(nu) = omega_e * (nu + 1/2) - omega_e * x_e * (nu + 1/2)^2  (in cm^-1)
    Returns a list of (nu, E_total_Ha, R_inner_A, R_outer_A).
    """
    levels: List[Tuple[int, float, float, float]] = []
    E_min = spec_const.minimum_energy_Ha
    D_e = spec_const.dissociation_energy_De_Ha
    a = spec_const.morse_width_a_inv_A
    R_e = spec_const.equilibrium_bond_length_A
    we = spec_const.harmonic_freq_omega_e_cm_inv
    wexe = spec_const.anharmonicity_omega_e_xe_cm_inv

    for nu in range(max_levels):
        g_cm = we * (nu + 0.5) - wexe * ((nu + 0.5) ** 2)
        g_ha = g_cm / HARTREE_TO_CM_INV
        if g_ha >= 0.95 * D_e or g_ha <= 0.0:
            break
        frac = np.sqrt(g_ha / D_e)
        r_in = R_e - (1.0 / a) * np.log(1.0 + frac)
        r_out = R_e - (1.0 / a) * np.log(max(1.0 - frac, 1e-6))
        levels.append((nu, E_min + g_ha, float(r_in), float(r_out)))

    return levels
