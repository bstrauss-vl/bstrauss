"""Export simulation results as CAD-importable point clouds in the original STEP coordinate frame.

The voxel domain and the display mesh are built directly from the STEP geometry without
any transform, so every coordinate below is in the part's native frame - the exported
points land exactly on the model when imported next to the original STEP.

Files written per run (in `<out_dir>/`, each prefixed with the original CAD file name):
  <part>_weld_lines.step         all weld/knit-line cells as STEP vertex points (one compound)
  <part>_weld_lines_headon.step  only head-on welds (meeting angle < `headon_deg`)
  <part>_weld_lines_surface.step weld cells projected onto the part skin (for visual overlay in CAD)
  <part>_weld_lines.csv          x, y, z, meeting_angle_deg, fill_time_s, gate_label, on_surface
  <part>_weld_lines.xyz          plain "x y z" (for point-cloud tools / NX "Import Points")
  <part>_air_traps.csv           last-to-fill pocket centroids with fill time
  <part>_gates.csv               gate locations used
  <part>_summary.json            headline numbers
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .solver import SimulationResult, air_trap_points


def weld_point_cloud(res: SimulationResult, surface_only: bool = False) -> dict:
    """All weld-line cells with attributes, in STEP coordinates.

    surface_only=True pushes each cell onto the part skin (nice for viewing);
    the default keeps the true volumetric cell centres so the seam's through-thickness
    extent is preserved for analysis.
    """
    dom = res.domain
    idx = np.argwhere(res.weld_mask)
    if len(idx) == 0:
        return {"xyz": np.zeros((0, 3)), "angle": np.zeros(0), "fill_time": np.zeros(0),
                "gate": np.zeros(0, int), "on_surface": np.zeros(0, bool)}
    xyz = dom.index_to_world(idx)
    edt = dom.edt[idx[:, 0], idx[:, 1], idx[:, 2]]
    on_surface = edt <= 1.6 * dom.pitch
    if surface_only:
        n = dom.normal_dir[idx[:, 0], idx[:, 1], idx[:, 2]]
        xyz = xyz - n * edt[:, None]
    return {
        "xyz": xyz,
        "angle": res.weld_angle[idx[:, 0], idx[:, 1], idx[:, 2]].astype(float),
        "fill_time": res.fill_time[idx[:, 0], idx[:, 1], idx[:, 2]].astype(float),
        "gate": res.gate_label[idx[:, 0], idx[:, 1], idx[:, 2]].astype(int) + 1,
        "on_surface": on_surface,
    }


def safe_prefix(name: str) -> str:
    """File-name-safe version of the original CAD file name (extension stripped)."""
    import re
    stem = Path(name).stem if name else ""
    stem = re.sub(r"[\\/:*?\"<>|]+", "_", stem).strip(" .")
    return stem[:80]


def write_step_points(path: str | Path, xyz: np.ndarray) -> Path:
    """Write points as a STEP file of vertex points (CARTESIAN_POINT / VERTEX_POINT entities).

    NX, SolidWorks, Creo, Fusion and FreeCAD import these as point entities.
    """
    import cadquery as cq

    path = Path(path)
    if len(xyz) == 0:
        # still write a valid (empty) file so downstream tooling does not break
        comp = cq.Compound.makeCompound([cq.Vertex.makeVertex(0.0, 0.0, 0.0)])
    else:
        comp = cq.Compound.makeCompound([cq.Vertex.makeVertex(float(x), float(y), float(z)) for x, y, z in xyz])
    cq.exporters.export(comp, str(path))
    return path


def write_csv(path: str | Path, header: list[str], rows) -> Path:
    path = Path(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return path


def export_results(res: SimulationResult, out_dir: str | Path, gates: list, prefix: str = "",
                   headon_deg: float = 45.0, max_step_points: int = 200_000) -> dict:
    """Write the full export set. Every file is named `<prefix>_<kind>.<ext>`, where `prefix`
    is the original CAD file name (without extension). Returns {name: path} plus counts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    prefix = safe_prefix(prefix)
    P = (lambda kind: out / f"{prefix}_{kind}") if prefix else (lambda kind: out / kind)

    W = weld_point_cloud(res, surface_only=False)
    xyz, ang, ft, g, surf = W["xyz"], W["angle"], W["fill_time"], W["gate"], W["on_surface"]
    if len(xyz) > max_step_points:
        sel = np.random.default_rng(0).choice(len(xyz), max_step_points, replace=False)
        xyz_s, ang_s = xyz[sel], ang[sel]
    else:
        xyz_s, ang_s = xyz, ang

    files["weld_lines.step"] = str(write_step_points(P("weld_lines.step"), xyz_s))
    headon = xyz_s[np.nan_to_num(ang_s, nan=180.0) < headon_deg]
    files["weld_lines_headon.step"] = str(write_step_points(P("weld_lines_headon.step"), headon))
    Ws = weld_point_cloud(res, surface_only=True)
    skin = Ws["xyz"][Ws["on_surface"]]
    if len(skin) > max_step_points:
        skin = skin[np.random.default_rng(0).choice(len(skin), max_step_points, replace=False)]
    files["weld_lines_surface.step"] = str(write_step_points(P("weld_lines_surface.step"), skin))
    files["weld_lines.csv"] = str(write_csv(
        P("weld_lines.csv"), ["x_mm", "y_mm", "z_mm", "meeting_angle_deg", "fill_time_s", "gate", "on_surface"],
        ((f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}", f"{a:.1f}", f"{t:.4f}", int(gg), int(s))
         for p, a, t, gg, s in zip(xyz, ang, ft, g, surf))))
    with open(P("weld_lines.xyz"), "w") as f:
        for p in xyz:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
    files["weld_lines.xyz"] = str(P("weld_lines.xyz"))

    ap, at = air_trap_points(res, max_points=200)
    files["air_traps.csv"] = str(write_csv(P("air_traps.csv"), ["x_mm", "y_mm", "z_mm", "fill_time_s"],
                                           ((f"{p[0]:.3f}", f"{p[1]:.3f}", f"{p[2]:.3f}", f"{t:.4f}") for p, t in zip(ap, at))))
    files["gates.csv"] = str(write_csv(P("gates.csv"), ["gate", "x_mm", "y_mm", "z_mm"],
                                       ((i + 1, f"{gq[0]:.3f}", f"{gq[1]:.3f}", f"{gq[2]:.3f}") for i, gq in enumerate(gates))))
    summary = dict(res.summary)
    summary["exported_at"] = datetime.now().isoformat(timespec="seconds")
    summary["coordinate_frame"] = "native STEP frame of the uploaded part (mm)"
    with open(P("summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    files["summary.json"] = str(P("summary.json"))

    files = {Path(v).name: v for v in files.values()}   # key = actual file name on disk
    return {"dir": str(out.resolve()), "files": files, "prefix": prefix, "weld_points": int(len(xyz)),
            "headon_points": int(len(headon)), "air_traps": int(len(ap))}
