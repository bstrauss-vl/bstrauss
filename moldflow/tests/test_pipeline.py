"""End-to-end checks on the generated sample parts (run: pytest -q)."""
from pathlib import Path

import numpy as np
import pytest

from moldflow.geometry import load_step, voxelize
from moldflow.materials import MATERIALS, get_material
from moldflow.solver import ProcessParams, simulate, solve_warpage, weld_points

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture(scope="module")
def plate():
    path = SAMPLES / "plate_with_hole.step"
    if not path.exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(SAMPLES / "make_samples.py")], check=True)
    mesh = load_step(path)
    return voxelize(mesh, target_voxels=60_000)


def test_thickness_is_exact_on_flat_plate(plate):
    T = plate.thickness[plate.solid]
    assert abs(np.median(T) - 2.5) < 0.05
    assert np.percentile(T, 5) > 2.3


def test_volume_matches_brep(plate):
    assert abs(plate.volume_mm3 / 1000 - 14.5) / 14.5 < 0.08


def test_materials_have_sane_viscosity():
    for m in MATERIALS.values():
        eta = m.viscosity(1000.0, m.melt_temp_c)
        assert 30 < eta < 2000, f"{m.key}: {eta}"
        assert m.viscosity(100.0, m.melt_temp_c) > eta  # shear thinning


def test_weld_line_forms_behind_hole(plate):
    res = simulate(plate, get_material("abs"), [np.array([-49.0, 0.0, 0.0])], ProcessParams(), run_warpage=False)
    pts, ang = weld_points(res)
    assert len(pts) > 20
    # weld must be downstream of the hole (hole spans x in [-8, 8]) and on the centreline
    assert pts[:, 0].min() > 5.0
    assert np.abs(pts[:, 1]).max() < 6.0
    # near-head-on meeting right behind the hole, melding further downstream
    near = ang[pts[:, 0] < 12]
    far = ang[pts[:, 0] > 25]
    assert np.nanmean(near) < np.nanmean(far)
    assert res.summary["injection_pressure_mpa"] > 1.0
    assert 5 < res.summary["cooling_time_s"] < 40


def test_two_gates_meet_in_the_middle(plate):
    res = simulate(plate, get_material("pp"), [np.array([-49.0, 0.0, 0.0]), np.array([49.0, 0.0, 0.0])],
                   ProcessParams(), run_warpage=False)
    pts, ang = weld_points(res)
    head_on = pts[ang < 30]
    assert len(head_on) > 10
    assert np.abs(head_on[:, 0]).mean() < 4.0  # fronts meet near x = 0


def test_uniform_shrink_gives_no_warpage(plate):
    eps = -0.01 * plate.solid.astype(float)
    w = solve_warpage(plate, eps, 0 * eps, 0.38, max_elements=15_000)
    warp = np.linalg.norm(w["disp_warp"], axis=1).max()
    total = np.linalg.norm(w["disp_total"], axis=1).max()
    assert warp < 0.02
    assert 0.4 < total < 0.7  # ~1% of the half-diagonal (58 mm)


def test_asymmetric_cooling_bows_plate(plate):
    res = simulate(plate, get_material("abs"), [np.array([-49.0, 0.0, 0.0])],
                   ProcessParams(core_cavity_dT_c=15, warpage_max_elements=15_000))
    assert res.summary["max_warpage_mm"] > 0.2


def test_server_roundtrip():
    from fastapi.testclient import TestClient
    from moldflow.server import app
    c = TestClient(app)
    assert len(c.get("/api/materials").json()) >= 10
    with open(SAMPLES / "plate_with_hole.step", "rb") as f:
        r = c.post("/api/upload?resolution=coarse", files={"file": ("plate.step", f, "application/step")})
    assert r.status_code == 200, r.text
    up = r.json()
    r = c.post("/api/simulate", json={"model_id": up["model_id"], "material": "abs", "gates": [[-49, 0, 0]],
                                      "run_warpage": True, "warpage_max_elements": 10_000})
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["fields"]["fill_time"]) == up["n_vertices"]
    assert d["warpage"]["fea_elements"] > 1000
    assert d["summary"]["weld_voxels"] > 0


def test_export_point_cloud_roundtrip(plate, tmp_path):
    import cadquery as cq
    from moldflow.export import export_results
    res = simulate(plate, get_material("abs"), [np.array([-49.0, 0.0, 0.0])], ProcessParams(), run_warpage=False)
    info = export_results(res, tmp_path, [[-49.0, 0.0, 0.0]], prefix="plate_with_hole.step")
    assert info["weld_points"] > 20
    assert set(info["files"]) == {f"plate_with_hole_{k}" for k in
        ["weld_lines.step", "weld_lines_headon.step", "weld_lines_surface.step", "weld_lines.csv", "weld_lines.xyz", "air_traps.csv", "gates.csv", "summary.json"]}
    P = np.array([[v.X, v.Y, v.Z] for v in cq.importers.importStep(str(tmp_path / "plate_with_hole_weld_lines.step")).vertices().vals()])
    assert len(P) == info["weld_points"]
    # same frame as the STEP: weld sits behind the hole on the centreline, inside the 2.5 mm plate
    assert P[:, 0].min() > 5.0 and np.abs(P[:, 1]).max() < 6.0 and np.abs(P[:, 2]).max() < 1.3
    assert (tmp_path / "plate_with_hole_weld_lines.csv").read_text().startswith("x_mm,y_mm,z_mm,meeting_angle_deg")
