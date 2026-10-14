"""FastAPI backend: upload STEP -> mesh, run simulation -> per-vertex result fields for the viewer."""
from __future__ import annotations

import io
import json
import math
import os
import re
import tempfile
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .geometry import VoxelDomain, load_mesh_any, mesh_to_payload, refine_for_display, voxelize
from .export import export_results
from .materials import get_material, list_materials
from .solver import (ProcessParams, SimulationResult, air_trap_points,
                     interpolate_node_field_to_vertices, simulate, weld_points)

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="moldflow", version=__version__)

_MODELS: dict[str, dict] = {}        # model_id -> {"mesh":..., "dom":..., "name":...}
_EXPORTS: dict[str, Path] = {}       # run_id -> export directory
_RESULTS: "OrderedDict[str, SimulationResult]" = OrderedDict()   # run_id -> result (last few, for /api/volume)
MAX_CACHED_RESULTS = 3
EXPORT_ROOT = Path(os.environ.get("MOLDFLOW_EXPORT_DIR", Path.cwd() / "moldflow_results"))
RESOLUTIONS = {"coarse": 50_000, "medium": 120_000, "fine": 250_000, "ultra": 500_000}


def _r(a, nd=4):
    """Round a float array to a JSON-friendly list (nan -> None)."""
    a = np.asarray(a, dtype=np.float64)
    out = np.round(a, nd)
    lst = out.tolist()
    if np.isnan(out).any():
        lst = [None if (isinstance(x, float) and math.isnan(x)) else x for x in lst]
    return lst


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})   # always revalidate the viewer


@app.get("/api/materials")
def materials():
    return list_materials()


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), resolution: str = "medium", target_voxels: Optional[int] = None):
    """Mesh a CAD file. `resolution` picks a preset; `target_voxels` (10k .. 20M) overrides it with a custom cell count."""
    suffix = Path(file.filename or "part.step").suffix.lower()
    if suffix not in (".step", ".stp", ".stl", ".obj", ".ply"):
        raise HTTPException(400, "Upload a .step/.stp file (STL/OBJ/PLY also accepted).")
    data = await file.read()
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        mesh = load_mesh_any(tmp_path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not read CAD file: {e}")
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    if target_voxels:
        target = int(np.clip(target_voxels, 10_000, 20_000_000))
        resolution = f"custom:{target}"
    else:
        target = RESOLUTIONS.get(resolution, RESOLUTIONS["medium"])
    try:
        dom = voxelize(mesh, target_voxels=target)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Voxelization failed: {e}")
    t_vox = time.perf_counter() - t0
    # finer surface mesh for display / field sampling (voxelization used the coarse CAD tessellation)
    dom.mesh = refine_for_display(mesh, dom.pitch)
    mesh = dom.mesh

    model_id = uuid.uuid4().hex[:12]
    _MODELS[model_id] = {"mesh": mesh, "dom": dom, "name": file.filename, "resolution": resolution}
    # keep memory bounded
    while len(_MODELS) > 6:
        _MODELS.pop(next(iter(_MODELS)))

    thick_v = dom.sample_field_at_vertices(dom.thickness)
    T = dom.thickness[dom.solid]
    payload = mesh_to_payload(mesh)
    payload.update({
        "model_id": model_id,
        "name": file.filename,
        "thickness": _r(thick_v, 3),
        "domain": {
            "pitch_mm": round(float(dom.pitch), 3),
            "n_voxels": int(dom.solid.sum()),
            "shape": list(dom.shape),
            "nominal_wall_mm": round(float(np.median(T)), 2),
            "min_wall_mm": round(float(np.percentile(T, 2)), 2),
            "max_wall_mm": round(float(np.percentile(T, 98)), 2),
            "volume_cm3": round(dom.volume_mm3 / 1000, 2),
        },
        "timings": {"load_s": round(t_load, 2), "voxelize_s": round(t_vox, 2)},
    })
    return JSONResponse(payload)


class SimRequest(BaseModel):
    model_id: str
    material: str = "abs"
    gates: list[list[float]] = Field(default_factory=list)
    injection_time_s: Optional[float] = None
    melt_temp_c: Optional[float] = None
    mold_temp_c: Optional[float] = None
    eject_temp_c: Optional[float] = None
    gate_diameter_mm: float = 1.5
    core_cavity_dT_c: float = 0.0
    mold_axis: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    packing_quality: float = 1.0
    run_warpage: bool = True
    warpage_max_elements: int = 40_000
    export: bool = True
    export_dir: Optional[str] = None     # override output folder (default: ./moldflow_results/<part>_<time>)


@app.post("/api/simulate")
def run_simulation(req: SimRequest):
    entry = _MODELS.get(req.model_id)
    if entry is None:
        raise HTTPException(404, "Model not found - upload the STEP file again.")
    if not req.gates:
        raise HTTPException(400, "Place at least one gate on the model.")
    try:
        mat = get_material(req.material)
    except KeyError as e:
        raise HTTPException(400, str(e))
    dom: VoxelDomain = entry["dom"]
    params = ProcessParams(
        injection_time_s=req.injection_time_s, melt_temp_c=req.melt_temp_c, mold_temp_c=req.mold_temp_c,
        eject_temp_c=req.eject_temp_c, gate_diameter_mm=req.gate_diameter_mm,
        core_cavity_dT_c=req.core_cavity_dT_c, mold_axis=tuple(req.mold_axis),
        packing_quality=float(np.clip(req.packing_quality, 0, 1)),
        warpage_max_elements=int(np.clip(req.warpage_max_elements, 5000, 120_000)),
    )
    t0 = time.perf_counter()
    try:
        res = simulate(dom, mat, [np.array(g, float) for g in req.gates], params, run_warpage=req.run_warpage)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Simulation failed: {e}")
    total = time.perf_counter() - t0
    payload = _result_payload(res, total)
    stem = Path(entry["name"] or "part").stem
    run_id = f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', stem)[:60]}_{time.strftime('%Y%m%d-%H%M%S')}"
    _remember_result(run_id, res)
    payload["run_id"] = run_id
    payload["volume"] = {"url": f"/api/volume/{run_id}", "n_voxels": int(dom.solid.sum()),
                         "shape": [int(x) for x in dom.shape], "pitch_mm": round(float(dom.pitch), 4)}
    if req.export:
        out_dir = Path(req.export_dir) if req.export_dir else EXPORT_ROOT / run_id
        try:
            info = export_results(res, out_dir, [list(map(float, g)) for g in req.gates], prefix=stem)
            _EXPORTS[run_id] = Path(info["dir"])
            payload["export"] = dict(info, run_id=run_id)
        except Exception as e:  # noqa: BLE001
            payload["export"] = {"error": f"export failed: {e}"}
    return JSONResponse(payload)


def _remember_result(run_id: str, res: SimulationResult) -> None:
    _RESULTS[run_id] = res
    while len(_RESULTS) > MAX_CACHED_RESULTS:
        _RESULTS.popitem(last=False)


def encode_volume(res: SimulationResult) -> bytes:
    """Binary 3-D voxel fields for the viewer (little-endian):

        b"MFV1" | uint32 header_len | header JSON (padded to 4 bytes)
        | uint32[n]  flat C-order index of every solid voxel
        | float32[n] fill time (s; +inf = never filled)
        | uint8[n]   weld meeting angle in degrees (255 = not a weld voxel)

    The header carries shape, pitch, origin (world position of voxel (0,0,0) centre), n and t_max.
    """
    dom = res.domain
    solid = dom.solid
    idx = np.flatnonzero(solid.ravel(order="C")).astype("<u4")
    ft = res.fill_time[solid]
    ft = np.where(np.isfinite(ft), ft, np.inf).astype("<f4")
    wa = res.weld_angle[solid]
    w8 = np.where(np.isfinite(wa), np.clip(np.rint(wa), 0, 180), 255).astype(np.uint8)
    hdr = {"shape": [int(x) for x in dom.shape], "pitch": float(dom.pitch), "origin": [float(x) for x in dom.origin],
           "n": int(len(idx)), "t_max": float(np.nanmax(res.fill_time)) if np.isfinite(res.fill_time).any() else 0.0,
           "n_weld": int((w8 != 255).sum())}
    hb = json.dumps(hdr).encode("utf-8")
    hb += b" " * (-len(hb) % 4)
    return b"".join([b"MFV1", np.uint32(len(hb)).tobytes(), hb, idx.tobytes(), ft.tobytes(), w8.tobytes()])


@app.get("/api/volume/{run_id}")
def volume(run_id: str):
    res = _RESULTS.get(run_id)
    if res is None:
        raise HTTPException(404, "Unknown run (results are kept in memory for the last few runs only - rerun the simulation).")
    return Response(content=encode_volume(res), media_type="application/octet-stream",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/export/{run_id}/{filename}")
def download_export(run_id: str, filename: str):
    d = _EXPORTS.get(run_id)
    if d is None:
        raise HTTPException(404, "Unknown run.")
    p = (d / filename).resolve()
    if d.resolve() not in p.parents or not p.exists():
        raise HTTPException(404, "No such file.")
    return FileResponse(p, filename=filename)


def _result_payload(res: SimulationResult, total_s: float) -> dict:
    dom = res.domain
    P_inj = float(np.nanmax(res.pressure_drop))
    fields = {
        "fill_time": dom.sample_field_at_vertices(res.fill_time),
        "thickness": dom.sample_field_at_vertices(dom.thickness),
        "pressure_eof": P_inj - dom.sample_field_at_vertices(res.pressure_drop),
        "pressure_drop": dom.sample_field_at_vertices(res.pressure_drop),
        "front_temp": dom.sample_field_at_vertices(res.front_temp),
        "frozen_fraction": dom.sample_field_at_vertices(res.frozen_fraction),
        "cooling_time": dom.sample_field_at_vertices(res.cooling_time),
        "shrinkage_pct": 100.0 * dom.sample_field_at_vertices(res.shrink_strain),
        "gate_label": dom.sample_field_at_vertices(res.gate_label.astype(np.float64)),
    }
    out = {"fields": {k: _r(v, 4) for k, v in fields.items()}}

    wp, wa = weld_points(res)
    out["weld"] = {"points": _r(wp.ravel(), 3), "angle": _r(wa, 1)}
    ap, at = air_trap_points(res)
    out["air_traps"] = {"points": _r(ap.ravel(), 3), "fill_time": _r(at, 3)}

    if res.warp_grid is not None:
        dw = interpolate_node_field_to_vertices(dom, res.warp_grid, res.warp_disp)
        dt = interpolate_node_field_to_vertices(dom, res.warp_grid, res.warp_disp_total)
        out["warpage"] = {
            "disp_warp": _r(dw.ravel(), 5),
            "disp_total": _r(dt.ravel(), 5),
            "mag_warp": _r(np.linalg.norm(dw, axis=1), 5),
            "mag_total": _r(np.linalg.norm(dt, axis=1), 5),
            "fea_elements": int(res.warp_grid["n_elements"]),
            "fea_pitch_mm": round(float(res.warp_grid["h"]), 3),
        }
    out["summary"] = res.summary
    out["timings"] = dict(res.timings, total=round(total_s, 2))
    out["material"] = res.material.to_dict()
    return out


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
