# moldflow — quick injection-molding simulation on a STEP file

A fast engineering-approximation tool: upload a `.step`, pick a resin, click the part
to place one or more gates, and get an interactive 3D picture of

* **fill pattern** (animated melt front, fill time),
* **weld / knit lines** with front meeting angle (head-on vs. meld),
* **air-trap / last-to-fill** candidates,
* **pressure at end of fill**, melt-front temperature and frozen-layer fraction,
* **cooling time** to ejection and an estimated cycle time,
* **warpage** (linear-elastic FEA with shrinkage eigenstrain) and total deflection.

It runs locally: a Python backend (FastAPI + OpenCascade via cadquery + numpy/scipy)
and a three.js viewer in your browser. Typical run time is 5–30 s per part.

> **What it is for:** comparing gate locations, spotting weld lines in bad places,
> finding thick/thin sections and hesitation, getting a first feel for warpage direction.
> **What it is not:** a replacement for Moldflow/Moldex3D. Pressures, clamp tonnage and
> absolute warpage magnitudes are order-of-magnitude estimates (see *Assumptions*).

---

## Install (Windows, Python 3.10–3.12; also works on 3.13/Linux/macOS)

```bat
cd C:\Users\BenjaminStrauss\git\bstrauss\moldflow
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e .
```

The `cadquery` wheel pulls in OpenCascade (~150 MB); the first install takes a few minutes.

## Run

```bat
moldflow
```

…opens `http://127.0.0.1:8765` in your browser. (`python -m moldflow` does the same;
`--port`, `--no-browser` and `--reload` are available.)

Then in the UI:

1. **Part** – drop a `.step` (STL/OBJ also accepted). Units are read as mm. Choose the
   mesh resolution (coarse ≈ 50k cells … ultra ≈ 500k) or type a custom cell count (up to
   20 M; budget about 1.6 GB RAM and 2 min per million cells, so 2 M ≈ 3–4 min). The
   thickness map appears at once.
2. **Material** – pick a profile; its Cross-WLF viscosity, temperatures, shrinkage and
   modulus are shown. Edit `moldflow/materials.py` to add your own grades.
3. **Gates** – click the model to add a gate (shift-click removes the nearest). Gate
   diameter sets the size of the injection region.
4. **Process** – leave injection time on *auto* (estimated from part volume) or set it;
   override melt/mold/eject temperatures; set a **core/cavity ΔT** to drive bending warpage;
   choose the mold-opening axis; toggle the warpage FEA and its element budget.
5. **Run** – results appear in the view dropdown. **Fill time** and **weld lines** are 3-D
   voxel views: the filled plastic (or the weld cells) are drawn as voxels inside a transparent
   ghost of the CAD, so the melt front and welds *inside* thick walls are visible, not just
   their trace on the skin. The *CAD opacity* slider sets the ghost; fill time has a play/scrub
   bar with a *weld lines* checkbox and a category panel (meeting-angle ranges, each with its
   own colour or not highlighted) that colours weld cells as the fronts meet;
   the same ranges drive the weld view, where cells outside a shown range are hidden; warpage has a
   deformation-scale slider.
   Hover for values (in the voxel views: the voxel under the cursor), *Screenshot* saves a PNG.

### CAD export (automatic)

Every run writes a point-cloud export in the **native STEP coordinate frame** of the part
to `./moldflow_results/<part>_<timestamp>/` (override with `MOLDFLOW_EXPORT_DIR` or the
CLI `--export-dir`); the UI shows the folder and download links:

| file | contents |
|---|---|
| `<part>_weld_lines.step` | every weld/knit-line cell as STEP vertex points (mid-wall positions) |
| `<part>_weld_lines_headon.step` | only head-on welds (meeting angle < 45°) |
| `<part>_weld_lines_surface.step` | weld cells projected onto the part skin, for visual overlay |
| `<part>_weld_lines.csv` / `.xyz` | x, y, z, meeting angle, fill time, gate id, on-surface flag |
| `<part>_air_traps.csv`, `<part>_gates.csv`, `<part>_summary.json` | vent candidates, gate positions, headline numbers |

`<part>` is the uploaded file's name without extension (e.g. `ribbed_housing`).

Import the STEP next to the original part (NX: File → Import → STEP; SolidWorks/Creo/Fusion
also read vertex points) — no transform is needed, the points land on the model.

### Headless / batch

```bat
moldflow-cli samples\plate_with_hole.step --material abs --gate -49,0,0 --gate 49,0,0 --out result.json
moldflow-cli part.step --material abs --gate 10,0,5 --voxels 1500000 --no-warpage   # custom cell count
```

### Sample parts

`python samples\make_samples.py` regenerates three test parts (plate with hole, ribbed
housing with boss, L-bracket). `pytest -q` runs the regression tests against them.

---

## How it works

```
STEP ─cadquery/OCC─▶ B-rep ─tessellate─▶ triangle mesh ─ray parity─▶ solid voxel grid
                                                          │
          thickness: inscribed-sphere EDT  +  exact axis-aligned chords
                                                          │
 fill:  fast marching, speed ∝ (h/h_nom)²  →  arrival order → constant-Q time mapping
 weld:  converging flow-direction field / gate-label boundaries → meeting angle
 pressure: ∫ 12 η(γ̇) v / h_eff² ds along flow paths (Cross-WLF, frozen-layer-reduced h)
 cooling: 1-D slab  t = h²/(π²α) · ln(4/π · (Tmelt−Tmold)/(Teject−Tmold))
 warpage: voxel-hex FEA, isotropic eigenstrain from local shrinkage (+ through-thickness
          asymmetry from core/cavity ΔT), AMG solve; rigid body & uniform shrink removed
```

Everything runs on the same voxel grid, so resolution is one knob. Surface fields (pressure,
temperature, cooling, warpage…) are sampled from the nearest solid voxel onto the refined display
mesh. Fill time and weld lines are streamed to the viewer as the full 3-D voxel field
(`/api/volume/<run_id>`, a compact binary of index / fill time / weld angle per solid voxel; the
server keeps the last three results in memory for this) and drawn as instanced cubes.

## Assumptions and limitations (read before trusting numbers)

* **Fill** is kinematic Hele-Shaw: the front advances with speed ∝ thickness² (constant
  pressure-gradient-per-viscosity assumption), so thick sections lead and thin sections
  hesitate. There is no coupled pressure/temperature solve, no fountain flow, no
  compressibility, no runner system (the gate is a point on the part).
* **Time scale**: arrival order from fast marching is mapped to real time assuming a
  constant volumetric flow rate; the injection time is user-set or estimated from volume.
* **Weld lines** are detected geometrically from the arrival-time field on the full 3-D grid
  (converging flow directions / different-gate boundaries), so welds buried in thick sections
  are found and shown, not only those that reach the skin. Meeting angle 0° = head-on
  (weakest), >135° is usually considered a meld line. Weld *strength* is not computed;
  melt-front temperature at the weld is shown in the *front temperature* view.
* **Pressure** uses mean-velocity Hele-Shaw drop with Cross-WLF viscosity at the local
  wall shear rate (6v/h) and melt temperature; a simple frozen-layer thickness
  (δ ≈ 0.6·√(αt)) reduces the effective gap. Expect under-prediction for long flow lengths
  and cold molds; use for ranking gate options, not for sizing a machine.
* **Clamp force** = projected area × ½ injection pressure.
* **Cooling time** is the classical slab formula on the local thickness – thick bosses
  and pads dominate; it ignores insert/corner effects and cooling-line layout.
* **Warpage**: linear-elastic, isotropic, homogeneous material with an eigenstrain
  ε* = −[s₀ + c_h·(h−h_nom) + c_p·(fill fraction)·(1−0.6·packing)] plus a through-thickness
  term ∝ core/cavity ΔT. It captures differential-shrinkage distortion from thickness
  variation, packing loss away from the gate and mold-half temperature difference. It does
  **not** model fibre orientation (GF grades are treated isotropically with the mean
  shrinkage), crystallinity gradients or in-mold constraints/ejection. Displacements are
  independent of E for free shrinkage, so only Poisson's ratio matters. Results are reported
  both as *warpage* (rigid-body motion and uniform scaling removed – pure distortion) and
  *deflection* (rigid body removed only – includes shrinkage).
* **Material data** are generic handbook-style values, not supplier datasheets.
* **Thin features** below ~2 voxels are under-resolved; increase the mesh resolution when
  ribs matter (the thickness view tells you quickly whether a rib reads correctly).

## Project layout

```
moldflow/
  geometry.py   STEP import, tessellation, ray-parity voxelizer, thickness
  materials.py  default resin profiles (Cross-WLF, thermal, shrinkage, elastic)
  solver.py     fill / weld / pressure / cooling / warpage
  server.py     FastAPI endpoints  (/api/upload, /api/simulate, /api/volume, /api/materials)
  cli.py        headless runner
  static/       index.html viewer (three.js vendored in static/vendor)
samples/        sample STEP generator
tests/          pytest regression tests
```

## Roadmap ideas

* Runner/sprue elements and multi-cavity layouts
* True Hele-Shaw pressure solve with temperature coupling (slower but quantitative)
* Fibre-orientation-aware anisotropic shrinkage for GF grades
* Sink-mark index from rib/wall ratios; vent recommendations from air-trap clusters
* Export results (VTK/CSV) for downstream FEA
