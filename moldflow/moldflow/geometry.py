"""STEP import, tessellation, voxelization and local wall-thickness estimation.

Pipeline
--------
STEP file --(cadquery / OpenCascade)--> B-rep solid
          --(BRepMesh)--> watertight surface triangle mesh (mm)
          --(trimesh voxelize + flood fill)--> solid voxel grid
          --(Euclidean distance transform + maximal inscribed spheres)--> local thickness

The voxel grid is the computational domain for every solver in `solver.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from scipy import ndimage


# --------------------------------------------------------------------------- #
# STEP import / tessellation
# --------------------------------------------------------------------------- #
def load_step(path: str | Path, linear_deflection: float = 0.15, angular_deflection: float = 0.25) -> trimesh.Trimesh:
    """Load a STEP file and return a single (possibly multi-body) triangle mesh in mm.

    `linear_deflection` is the chordal tolerance in mm; smaller = finer mesh.
    """
    import cadquery as cq  # heavy import, keep local

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    shape = cq.importers.importStep(str(path))
    solids = shape.solids().vals()
    if not solids:
        raise ValueError("STEP file contains no solid bodies (surfaces/wires only).")

    meshes = []
    for solid in solids:
        verts, tris = solid.tessellate(linear_deflection, angular_deflection)
        v = np.array([[p.x, p.y, p.z] for p in verts], dtype=np.float64)
        f = np.array(tris, dtype=np.int64)
        if len(f):
            meshes.append(trimesh.Trimesh(vertices=v, faces=f, process=False))
    if not meshes:
        raise ValueError("Tessellation produced no triangles.")

    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.fix_normals()
    return mesh


def load_mesh_any(path: str | Path) -> trimesh.Trimesh:
    """Load STEP (via OpenCascade) or any mesh format trimesh understands (STL/OBJ/PLY)."""
    p = Path(path)
    if p.suffix.lower() in (".step", ".stp"):
        return load_step(p)
    m = trimesh.load(str(p), force="mesh")
    m.fix_normals()
    return m


# --------------------------------------------------------------------------- #
# Voxel domain
# --------------------------------------------------------------------------- #
@dataclass
class VoxelDomain:
    """A solid voxelization of the part plus derived geometric fields."""

    pitch: float                       # voxel edge length, mm
    origin: np.ndarray                 # world coords of voxel (0,0,0) centre
    solid: np.ndarray                  # bool (nx,ny,nz) - True inside part
    thickness: np.ndarray              # float32 (nx,ny,nz) - local wall thickness, mm (0 outside)
    edt: np.ndarray                    # float32 distance to nearest surface, mm
    mesh: trimesh.Trimesh = field(repr=False)
    normal_dir: np.ndarray = field(repr=False, default=None)  # unit gradient of EDT (points inward), (nx,ny,nz,3)

    # ---- indexing helpers ------------------------------------------------- #
    @property
    def shape(self):
        return self.solid.shape

    def world_to_index(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        idx = np.rint((pts - self.origin) / self.pitch).astype(int)
        return np.clip(idx, 0, np.array(self.shape) - 1)

    def index_to_world(self, idx: np.ndarray) -> np.ndarray:
        return np.asarray(idx, dtype=float) * self.pitch + self.origin

    def nearest_solid_index(self, pt: np.ndarray) -> tuple[int, int, int]:
        """Snap a world point to the nearest solid voxel (for gate placement)."""
        i = self.world_to_index(pt)[0]
        if self.solid[tuple(i)]:
            return tuple(int(x) for x in i)
        # search outward in growing cubes
        n = np.array(self.shape)
        for r in range(1, max(n)):
            lo = np.maximum(i - r, 0)
            hi = np.minimum(i + r + 1, n)
            sub = self.solid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            if sub.any():
                cand = np.argwhere(sub) + lo
                d = np.linalg.norm(cand - i, axis=1)
                return tuple(int(x) for x in cand[np.argmin(d)])
        raise ValueError("No solid voxels in domain.")

    _vertex_idx_cache: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def nearest_solid_indices(self) -> np.ndarray:
        """Index of the nearest solid voxel for every outside voxel (3, nx, ny, nz); cached."""
        if getattr(self, "_nearest_cache", None) is None:
            _, nearest = ndimage.distance_transform_edt(~self.solid, return_indices=True)
            self._nearest_cache = nearest
        return self._nearest_cache

    def vertex_voxel_indices(self) -> np.ndarray:
        """Solid voxel index for every surface-mesh vertex (snapped into the solid); cached."""
        if self._vertex_idx_cache is None or len(self._vertex_idx_cache) != len(self.mesh.vertices):
            # sample slightly inside the part (along the inward normal) to avoid outside voxels
            V = self.mesh.vertices
            vn = self.mesh.vertex_normals
            idx = self.world_to_index(V - 0.6 * self.pitch * vn)
            inside = self.solid[idx[:, 0], idx[:, 1], idx[:, 2]]
            if not inside.all():
                nearest = self.nearest_solid_indices()
                bad = ~inside
                ii = idx[bad]
                idx[bad] = nearest[:, ii[:, 0], ii[:, 1], ii[:, 2]].T
            self._vertex_idx_cache = idx
        return self._vertex_idx_cache

    def sample_field_at_vertices(self, field_grid: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
        """Sample a voxel field at the surface-mesh vertices (nearest solid voxel)."""
        idx = self.vertex_voxel_indices()
        vals = field_grid[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float64)
        vals[~np.isfinite(vals)] = fill_value
        return vals

    @property
    def volume_mm3(self) -> float:
        return float(self.solid.sum()) * self.pitch ** 3


def _ray_parity(V: np.ndarray, F: np.ndarray, origin: np.ndarray, shape: tuple, pitch: float):
    """Rasterise triangles onto the xy grid of column centres and pair z-crossings.

    Returns (solid bool grid, chord float32 grid) where chord is the exact length of the
    solid segment along z through each solid voxel's column (0 outside).
    """
    nx, ny, nz = shape
    tri = V[F]
    col_idx_all, z_all = [], []
    for a, b, c in tri:
        d00 = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(d00) < 1e-14:
            continue  # vertical triangle: no column crossing
        xs = (a[0], b[0], c[0]); ys = (a[1], b[1], c[1])
        i0 = max(int(np.ceil((min(xs) - origin[0]) / pitch)), 0); i1 = min(int(np.floor((max(xs) - origin[0]) / pitch)), nx - 1)
        j0 = max(int(np.ceil((min(ys) - origin[1]) / pitch)), 0); j1 = min(int(np.floor((max(ys) - origin[1]) / pitch)), ny - 1)
        if i1 < i0 or j1 < j0:
            continue
        gi, gj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        px = origin[0] + gi * pitch - a[0]
        py = origin[1] + gj * pitch - a[1]
        v = (px * (c[1] - a[1]) - (c[0] - a[0]) * py) / d00
        w = ((b[0] - a[0]) * py - px * (b[1] - a[1])) / d00
        u = 1.0 - v - w
        inside = (u >= 0) & (v >= 0) & (w >= 0)
        if not inside.any():
            continue
        z = u * a[2] + v * b[2] + w * c[2]
        col_idx_all.append(gi[inside] * ny + gj[inside])
        z_all.append(z[inside])

    solid = np.zeros((nx, ny, nz), dtype=bool)
    chord = np.zeros((nx, ny, nz), dtype=np.float32)
    if not col_idx_all:
        return solid, chord
    col = np.concatenate(col_idx_all)
    z = np.concatenate(z_all)
    order = np.lexsort((z, col))
    col, z = col[order], z[order]
    starts = np.r_[0, np.flatnonzero(np.diff(col)) + 1]
    counts = np.diff(np.r_[starts, len(col)])
    rank = np.arange(len(col)) - np.repeat(starts, counts)
    valid_pair = (rank % 2 == 0) & (rank + 1 < np.repeat(counts, counts))
    z0 = z[valid_pair]
    z1 = z[np.flatnonzero(valid_pair) + 1]
    cols = col[valid_pair]
    k0 = np.ceil((z0 - origin[2]) / pitch).astype(int)
    k1 = np.floor((z1 - origin[2]) / pitch).astype(int)
    ok = k1 >= k0
    k0, k1, cols, seg = np.clip(k0[ok], 0, nz), np.clip(k1[ok], 0, nz - 1), cols[ok], (z1 - z0)[ok]
    acc = np.zeros((nx * ny, nz + 1), dtype=np.int32)
    np.add.at(acc, (cols, k0), 1)
    np.add.at(acc, (cols, k1 + 1), -1)
    solid = (np.cumsum(acc, axis=1)[:, :nz] > 0).reshape(nx, ny, nz)
    # chord lengths: paint each segment's voxels (vectorised via repeat)
    lens = k1 - k0 + 1
    kk = np.repeat(k0, lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
    cc = np.repeat(cols, lens)
    chord_flat = chord.reshape(nx * ny, nz)
    np.maximum.at(chord_flat, (cc, kk), np.repeat(seg, lens).astype(np.float32))
    return solid, chord_flat.reshape(nx, ny, nz)


def voxelize_mesh(mesh: trimesh.Trimesh, pitch: float, pad: int = 2, with_chords: bool = False):
    """Solid voxelization by ray parity (fast, handles multi-body parts and internal voids).

    Returns (solid, origin) or, with `with_chords`, (solid, origin, chord_min) where chord_min is
    the shortest axis-aligned solid chord through each voxel - an exact wall thickness for
    axis-aligned walls, used to de-bias the voxel thickness estimate.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    bmin, bmax = V.min(0), V.max(0)
    # irrational-ish offset so column centres never sit exactly on CAD edges
    origin = bmin - pad * pitch + pitch * np.array([0.1371, 0.2718, 0.3142])
    n = np.ceil((bmax - origin) / pitch).astype(int) + pad + 1
    shape = (int(n[0]), int(n[1]), int(n[2]))

    solid, chord_z = _ray_parity(V, F, origin, shape, pitch)
    if not with_chords:
        return solid, origin
    chord = np.where(solid, chord_z, np.inf).astype(np.float32)
    # x- and y-direction passes by cyclic axis permutation (x,y,z) -> (y,z,x) -> (z,x,y)
    for perm in ((1, 2, 0), (2, 0, 1)):
        Vp = V[:, perm]
        op = origin[list(perm)]
        shp = tuple(shape[i] for i in perm)
        sp, cp = _ray_parity(Vp, F, op, shp, pitch)
        inv = np.argsort(perm)
        cp = np.transpose(cp, inv)
        sp = np.transpose(sp, inv)
        chord = np.where(sp & (cp > 0), np.minimum(chord, cp), chord)
    chord[~np.isfinite(chord)] = 0.0
    return solid, origin, chord


def _choose_pitch(mesh: trimesh.Trimesh, target_voxels: int) -> float:
    ext = mesh.bounding_box.extents
    vol_bbox = float(np.prod(ext))
    # part volume fraction of its bbox is typically 0.1-0.5; aim for target solid voxels
    frac = max(min(mesh.volume / vol_bbox if mesh.is_watertight and mesh.volume > 0 else 0.3, 1.0), 0.05)
    pitch = (vol_bbox * frac / target_voxels) ** (1 / 3)
    return float(pitch)


def voxelize(mesh: trimesh.Trimesh, target_voxels: int = 120_000, pitch: Optional[float] = None,
             min_cells_across: int = 3) -> VoxelDomain:
    """Voxelize the solid and compute EDT + local thickness.

    `target_voxels` controls resolution: roughly how many solid voxels to produce.
    Thin walls thinner than `min_cells_across * pitch` are automatically refined.
    """
    if pitch is None:
        pitch = _choose_pitch(mesh, target_voxels)

    for _attempt in range(3):
        solid, origin, chord = voxelize_mesh(mesh, pitch, pad=2, with_chords=True)
        if solid.sum() < 50:
            raise ValueError("Voxelization produced almost no solid cells - is the STEP a closed solid in mm?")

        # distance from voxel centre to the true surface ~ EDT - pitch/2
        edt = ndimage.distance_transform_edt(solid, sampling=pitch).astype(np.float32)
        edt = np.where(solid, np.maximum(edt - 0.5 * pitch, 0.25 * pitch), 0.0).astype(np.float32)
        # Representative wall thickness ~ 2 * median EDT at medial voxels
        med_mask = _medial_mask(edt, solid)
        rep_thick = 2.0 * float(np.median(edt[med_mask])) if med_mask.any() else 2 * pitch
        cells_across = rep_thick / pitch
        if cells_across >= min_cells_across or solid.sum() > 3 * target_voxels:
            break
        # refine so the nominal wall spans >= min_cells_across voxels (bounded by voxel budget)
        new_pitch = rep_thick / (min_cells_across + 0.2)
        growth = (pitch / new_pitch) ** 3
        if solid.sum() * growth > 3 * target_voxels:
            new_pitch = pitch / (3 * target_voxels / solid.sum()) ** (1 / 3)
        if abs(new_pitch - pitch) / pitch < 0.05:
            break
        pitch = new_pitch

    thickness = local_thickness(edt, solid, pitch)
    # the inscribed-sphere estimate carries a +-1 voxel bias; the exact axis-aligned chord
    # length is a tight upper bound on wall thickness for flat walls, so take the minimum
    # (+-1 voxel tolerance). Chords much longer than the sphere estimate run along a feature
    # or through an inclined wall and are ignored.
    sphere_local_max = ndimage.maximum_filter(thickness, size=5, mode="nearest")
    chord_ok = solid & (chord > 0) & (chord <= sphere_local_max + 1.5 * pitch)
    thickness = np.where(chord_ok, chord, thickness).astype(np.float32)
    thickness = np.where(solid, np.maximum(thickness, 0.5 * pitch), 0.0).astype(np.float32)
    normal_dir = _inward_normal_field(edt, pitch)

    return VoxelDomain(pitch=pitch, origin=origin, solid=solid, thickness=thickness,
                       edt=edt, mesh=mesh, normal_dir=normal_dir)


def _medial_mask(edt: np.ndarray, solid: np.ndarray) -> np.ndarray:
    """Approximate medial axis: solid voxels that are local maxima of the EDT."""
    mx = ndimage.maximum_filter(edt, size=3, mode="constant")
    return solid & (edt >= mx - 1e-6) & (edt > 0)


def local_thickness(edt: np.ndarray, solid: np.ndarray, pitch: float) -> np.ndarray:
    """Local thickness (Hildebrand & Rüegsegger): diameter of the largest sphere
    that fits inside the part and contains the voxel.

    Implemented by painting spheres from medial-axis voxels in descending radius order.
    """
    thick = np.zeros(edt.shape, dtype=np.float32)
    med = _medial_mask(edt, solid)
    idx = np.argwhere(med)
    radii = edt[med]
    order = np.argsort(-radii)
    idx, radii = idx[order], radii[order]

    shape = np.array(edt.shape)
    for (i, j, k), r in zip(idx, radii):
        rv = int(np.ceil(r / pitch))
        if rv < 1:
            rv = 1
        lo = np.maximum([i - rv, j - rv, k - rv], 0)
        hi = np.minimum([i + rv + 1, j + rv + 1, k + rv + 1], shape)
        sub = thick[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        if sub.min() >= 2 * r:
            continue  # already covered by a bigger sphere
        gx, gy, gz = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        inside = ((gx - i) ** 2 + (gy - j) ** 2 + (gz - k) ** 2) * pitch ** 2 <= (r + 0.5 * pitch) ** 2
        np.maximum(sub, np.where(inside, 2 * r, 0), out=sub)

    # any solid voxel not painted (numerical corner cases): fall back to 2*EDT
    unpainted = solid & (thick == 0)
    thick[unpainted] = 2 * edt[unpainted]
    thick[~solid] = 0
    # smooth slightly to remove sphere-painting facets, then re-mask
    thick_s = ndimage.uniform_filter(thick, size=3, mode="nearest")
    cnt = ndimage.uniform_filter(solid.astype(np.float32), size=3, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        thick_s = np.where(cnt > 0, thick_s / cnt, 0)
    thick_s[~solid] = 0
    return thick_s.astype(np.float32)


def _inward_normal_field(edt: np.ndarray, pitch: float) -> np.ndarray:
    g = np.stack(np.gradient(edt, pitch), axis=-1)
    n = np.linalg.norm(g, axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(n > 1e-9, g / n, 0.0)
    return g.astype(np.float32)


# --------------------------------------------------------------------------- #
# Surface mesh export for the viewer
# --------------------------------------------------------------------------- #
def refine_for_display(mesh: trimesh.Trimesh, pitch: float, max_faces: int = 400_000) -> trimesh.Trimesh:
    """Subdivide long edges so per-vertex colouring can resolve voxel-scale detail.

    CAD tessellations put only a handful of triangles on flat faces, which makes vertex
    colours interpolate across the whole face. Splitting edges down to ~2 voxel pitches
    gives a smooth, faithful picture while keeping the exact B-rep surface.
    """
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    max_edge = max(2.0 * pitch, diag / 250.0)
    for _ in range(4):
        v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=max_edge, max_iter=12)
        if len(f) <= max_faces:
            break
        max_edge *= 1.5
    out = trimesh.Trimesh(vertices=v, faces=f, process=False)
    out.merge_vertices()
    return out



def mesh_to_payload(mesh: trimesh.Trimesh) -> dict:
    return {
        "vertices": mesh.vertices.astype(np.float32).ravel().tolist(),
        "faces": mesh.faces.astype(np.int32).ravel().tolist(),
        "bbox_min": mesh.bounds[0].tolist(),
        "bbox_max": mesh.bounds[1].tolist(),
        "volume_mm3": float(abs(mesh.volume)) if mesh.is_watertight else None,
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
    }
