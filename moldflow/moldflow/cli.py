"""Headless command-line run: `moldflow-cli part.step --material abs --gate -49,0,0 --out result.json`

Writes a JSON summary (and optionally per-vertex fields) without starting the web UI -
handy for batch gate-location studies.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .geometry import load_mesh_any, voxelize
from .materials import MATERIALS, get_material
from .server import RESOLUTIONS, _result_payload
from .solver import ProcessParams, simulate


def main(argv=None):
    ap = argparse.ArgumentParser(prog="moldflow-cli", description="Run a moldflow simulation headlessly")
    ap.add_argument("step", help="STEP / STL file")
    ap.add_argument("--material", default="abs", choices=sorted(MATERIALS))
    ap.add_argument("--gate", action="append", required=True, help="x,y,z in mm (repeat for multiple gates)")
    ap.add_argument("--resolution", default="medium", choices=sorted(RESOLUTIONS))
    ap.add_argument("--voxels", type=int, help="custom target cell count (overrides --resolution; ~1.6 GB RAM and ~2 min per million)")
    ap.add_argument("--injection-time", type=float)
    ap.add_argument("--melt", type=float); ap.add_argument("--mold", type=float); ap.add_argument("--eject", type=float)
    ap.add_argument("--dT", type=float, default=0.0, help="core/cavity temperature difference, °C")
    ap.add_argument("--no-warpage", action="store_true")
    ap.add_argument("--fields", action="store_true", help="include per-vertex fields in the JSON")
    ap.add_argument("--out", help="write JSON here (default: print summary)")
    ap.add_argument("--export-dir", help="folder for CAD point-cloud exports (default: <step folder>/moldflow_results/<part>_<time>)")
    ap.add_argument("--no-export", action="store_true", help="skip STEP/CSV point-cloud export")
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    mesh = load_mesh_any(a.step)
    dom = voxelize(mesh, target_voxels=a.voxels if a.voxels else RESOLUTIONS[a.resolution])
    gates = [np.array([float(v) for v in g.split(",")]) for g in a.gate]
    params = ProcessParams(injection_time_s=a.injection_time, melt_temp_c=a.melt, mold_temp_c=a.mold,
                           eject_temp_c=a.eject, core_cavity_dT_c=a.dT)
    res = simulate(dom, get_material(a.material), gates, params, run_warpage=not a.no_warpage)
    payload = _result_payload(res, time.perf_counter() - t0)
    if not a.fields:
        payload.pop("fields", None); payload.pop("warpage", None); payload.pop("weld", None); payload.pop("air_traps", None)
    if not a.no_export:
        from .export import export_results
        src = Path(a.step)
        out_dir = Path(a.export_dir) if a.export_dir else src.parent / "moldflow_results" / f"{src.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
        info = export_results(res, out_dir, [list(map(float, g)) for g in gates], prefix=src.stem)
        print(f"exported {info['weld_points']} weld points ({info['headon_points']} head-on) to {info['dir']}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"wrote {a.out}")
    for k, v in payload["summary"].items():
        print(f"{k:28s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
