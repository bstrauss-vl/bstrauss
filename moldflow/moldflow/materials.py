"""Default injection-molding material profiles.

Each profile carries:
  * Cross-WLF viscosity parameters  (n, tau_star [Pa], D1 [Pa.s], D2 [K], A1, A2 [K])
  * Recommended processing temperatures  (melt, mold, ejection) in °C
  * Thermal diffusivity alpha [m^2/s] of the solidifying melt
  * Linear mold shrinkage (nominal, in flow direction) and a thickness sensitivity
  * Elastic constants used by the warpage solver (E [MPa], nu, CTE [1/K])

IMPORTANT: these are *generic, representative* grade values compiled from public
handbook ranges - good for comparing gate locations and design variants, NOT a
substitute for supplier datasheets or a measured rheology curve. Edit or add
profiles freely; the solver only reads the fields below.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class Material:
    key: str
    name: str
    family: str                 # amorphous | semi-crystalline
    filled: bool
    # Cross-WLF
    n: float
    tau_star: float
    D1: float
    D2: float
    A1: float
    A2: float
    # processing
    melt_temp_c: float
    mold_temp_c: float
    eject_temp_c: float
    # thermal
    alpha_m2_s: float
    # shrinkage
    shrinkage: float            # nominal linear shrinkage (fraction) at nominal wall, well-packed
    shrink_thickness_coeff: float  # extra shrinkage per mm of wall above nominal (fraction/mm)
    shrink_pressure_coeff: float   # extra shrinkage at far-from-gate (fraction) from packing loss
    # mechanical
    E_mpa: float
    nu: float
    cte_1_k: float
    notes: str = ""

    # ---- rheology --------------------------------------------------------- #
    def viscosity(self, shear_rate: float, temp_c: float) -> float:
        """Cross-WLF viscosity [Pa.s] at a given shear rate [1/s] and temperature [°C]."""
        T = temp_c + 273.15
        T_star = self.D2
        denom = self.A2 + (T - T_star)
        if denom <= 1.0:
            return float("inf")
        eta0 = self.D1 * math.exp(-self.A1 * (T - T_star) / denom)
        return eta0 / (1.0 + (eta0 * max(shear_rate, 1e-6) / self.tau_star) ** (1.0 - self.n))

    def zero_shear_viscosity(self, temp_c: float) -> float:
        return self.viscosity(1e-6, temp_c)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["viscosity_1000_at_melt"] = round(self.viscosity(1000.0, self.melt_temp_c), 1)
        d["viscosity_100_at_melt"] = round(self.viscosity(100.0, self.melt_temp_c), 1)
        return d


# Representative handbook-style profiles --------------------------------------
# (viscosity targets at 1000 1/s and the recommended melt temperature are noted
#  in each profile's `notes`; D1 was tuned so the curve lands in that range)
MATERIALS: dict[str, Material] = {}


def _add(m: Material):
    MATERIALS[m.key] = m


_add(Material(
    key="abs", name="ABS (general purpose)", family="amorphous", filled=False,
    n=0.27, tau_star=42_000.0, D1=4.0e12, D2=373.15, A1=27.6, A2=51.6,
    melt_temp_c=230, mold_temp_c=60, eject_temp_c=90,
    alpha_m2_s=9.0e-8,
    shrinkage=0.0055, shrink_thickness_coeff=0.0006, shrink_pressure_coeff=0.0015,
    E_mpa=2200, nu=0.38, cte_1_k=85e-6,
    notes="Easy-flowing amorphous resin; ~250-350 Pa.s at 1000 1/s, 230 °C.",
))
_add(Material(
    key="pc_abs", name="PC/ABS blend", family="amorphous", filled=False,
    n=0.25, tau_star=150_000.0, D1=5.0e12, D2=383.15, A1=28.0, A2=51.6,
    melt_temp_c=260, mold_temp_c=75, eject_temp_c=110,
    alpha_m2_s=1.0e-7,
    shrinkage=0.0055, shrink_thickness_coeff=0.0005, shrink_pressure_coeff=0.0012,
    E_mpa=2400, nu=0.38, cte_1_k=75e-6,
    notes="Housings and enclosures; stiffer flow than ABS.",
))
_add(Material(
    key="pc", name="Polycarbonate (PC)", family="amorphous", filled=False,
    n=0.16, tau_star=850_000.0, D1=7.0e11, D2=417.15, A1=27.5, A2=51.6,
    melt_temp_c=300, mold_temp_c=90, eject_temp_c=130,
    alpha_m2_s=1.1e-7,
    shrinkage=0.006, shrink_thickness_coeff=0.0004, shrink_pressure_coeff=0.0010,
    E_mpa=2350, nu=0.37, cte_1_k=68e-6,
    notes="High viscosity, needs high melt temperature and pressure; low shrink.",
))
_add(Material(
    key="pmma", name="Acrylic (PMMA)", family="amorphous", filled=False,
    n=0.25, tau_star=200_000.0, D1=2.0e14, D2=373.15, A1=35.0, A2=51.6,
    melt_temp_c=240, mold_temp_c=70, eject_temp_c=95,
    alpha_m2_s=1.1e-7,
    shrinkage=0.005, shrink_thickness_coeff=0.0004, shrink_pressure_coeff=0.0010,
    E_mpa=3000, nu=0.37, cte_1_k=70e-6,
    notes="Optical parts; sensitive to weld lines (visible).",
))
_add(Material(
    key="ps", name="Polystyrene (GPPS)", family="amorphous", filled=False,
    n=0.25, tau_star=30_000.0, D1=1.0e12, D2=373.15, A1=25.0, A2=51.6,
    melt_temp_c=220, mold_temp_c=45, eject_temp_c=85,
    alpha_m2_s=9.0e-8,
    shrinkage=0.005, shrink_thickness_coeff=0.0004, shrink_pressure_coeff=0.0010,
    E_mpa=3200, nu=0.34, cte_1_k=80e-6,
    notes="Very easy flow, brittle.",
))
_add(Material(
    key="pp", name="Polypropylene (PP homopolymer)", family="semi-crystalline", filled=False,
    n=0.25, tau_star=30_000.0, D1=5.0e14, D2=263.15, A1=30.2, A2=51.6,
    melt_temp_c=230, mold_temp_c=40, eject_temp_c=95,
    alpha_m2_s=7.0e-8,
    shrinkage=0.015, shrink_thickness_coeff=0.0020, shrink_pressure_coeff=0.0040,
    E_mpa=1400, nu=0.42, cte_1_k=100e-6,
    notes="High shrinkage, strong thickness sensitivity -> warpage prone.",
))
_add(Material(
    key="hdpe", name="Polyethylene (HDPE)", family="semi-crystalline", filled=False,
    n=0.30, tau_star=40_000.0, D1=2.0e13, D2=233.15, A1=28.0, A2=51.6,
    melt_temp_c=220, mold_temp_c=30, eject_temp_c=80,
    alpha_m2_s=1.3e-7,
    shrinkage=0.020, shrink_thickness_coeff=0.0025, shrink_pressure_coeff=0.0050,
    E_mpa=1000, nu=0.42, cte_1_k=150e-6,
    notes="Highest shrinkage of the common resins.",
))
_add(Material(
    key="pa6", name="Nylon 6 (PA6, unfilled)", family="semi-crystalline", filled=False,
    n=0.60, tau_star=120_000.0, D1=2.0e12, D2=323.15, A1=28.0, A2=51.6,
    melt_temp_c=260, mold_temp_c=80, eject_temp_c=160,
    alpha_m2_s=1.0e-7,
    shrinkage=0.012, shrink_thickness_coeff=0.0015, shrink_pressure_coeff=0.0030,
    E_mpa=2800, nu=0.39, cte_1_k=90e-6,
    notes="Low viscosity, near-Newtonian; flashes easily.",
))
_add(Material(
    key="pa66_gf30", name="Nylon 66 + 30% glass fibre (PA66-GF30)", family="semi-crystalline", filled=True,
    n=0.42, tau_star=180_000.0, D1=3.0e12, D2=323.15, A1=27.0, A2=51.6,
    melt_temp_c=290, mold_temp_c=90, eject_temp_c=200,
    alpha_m2_s=1.5e-7,
    shrinkage=0.005, shrink_thickness_coeff=0.0008, shrink_pressure_coeff=0.0020,
    E_mpa=9000, nu=0.38, cte_1_k=30e-6,
    notes="Fibre orientation makes real shrinkage anisotropic (0.3-0.5% flow / 0.8-1.2% cross); "
          "this isotropic model uses the mean. Weld lines lose ~50% strength.",
))
_add(Material(
    key="pom", name="Acetal (POM homopolymer)", family="semi-crystalline", filled=False,
    n=0.35, tau_star=150_000.0, D1=5.3e14, D2=223.15, A1=32.0, A2=51.6,
    melt_temp_c=205, mold_temp_c=90, eject_temp_c=130,
    alpha_m2_s=1.2e-7,
    shrinkage=0.020, shrink_thickness_coeff=0.0020, shrink_pressure_coeff=0.0045,
    E_mpa=2800, nu=0.35, cte_1_k=110e-6,
    notes="Narrow processing window; high shrink.",
))
_add(Material(
    key="pbt", name="PBT (unfilled)", family="semi-crystalline", filled=False,
    n=0.50, tau_star=150_000.0, D1=2.8e11, D2=323.15, A1=25.0, A2=51.6,
    melt_temp_c=255, mold_temp_c=70, eject_temp_c=150,
    alpha_m2_s=1.0e-7,
    shrinkage=0.016, shrink_thickness_coeff=0.0015, shrink_pressure_coeff=0.0035,
    E_mpa=2500, nu=0.39, cte_1_k=95e-6,
    notes="Electrical connectors; fast crystallising.",
))


_add(Material(
    key="pps_gf40", name="PPS + 40% glass fibre (PPS-GF40)", family="semi-crystalline", filled=True,
    n=0.45, tau_star=150_000.0, D1=5.0e10, D2=363.15, A1=22.0, A2=51.6,
    melt_temp_c=320, mold_temp_c=140, eject_temp_c=200,
    alpha_m2_s=1.8e-7,
    shrinkage=0.004, shrink_thickness_coeff=0.0005, shrink_pressure_coeff=0.0012,
    E_mpa=14000, nu=0.38, cte_1_k=20e-6,
    notes="High-temperature, chemically resistant valve/pump grade. Hot mold (>=135 °C) needed for "
          "crystallinity. Real shrinkage is anisotropic (~0.2-0.3% flow / 0.5-0.7% cross); mean used here. "
          "Weld lines retain only ~30-50% of strength - keep them away from pressure-bearing walls.",
))


def get_material(key: str) -> Material:
    try:
        return MATERIALS[key]
    except KeyError:
        raise KeyError(f"Unknown material '{key}'. Available: {', '.join(MATERIALS)}")


def list_materials() -> list[dict]:
    return [m.to_dict() for m in MATERIALS.values()]
