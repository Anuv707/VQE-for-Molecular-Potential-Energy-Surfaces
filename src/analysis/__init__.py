"""
Spectroscopic analysis (Morse potential fitting, equilibrium bond length R_e,
harmonic vibrational frequency omega_e, dissociation energy D_e) and publication plotting.
"""

from .spectroscopy import (
    SpectroscopicConstants,
    morse_potential,
    extract_spectroscopic_constants,
    compute_morse_vibrational_levels,
)
from .plotting import (
    generate_all_figures,
)

__all__ = [
    "SpectroscopicConstants",
    "morse_potential",
    "extract_spectroscopic_constants",
    "compute_morse_vibrational_levels",
    "generate_all_figures",
]
