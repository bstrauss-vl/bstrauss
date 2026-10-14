"""Fast engineering-approximation molding solvers on the voxel domain.

Physics summary (all deliberately simple, documented in README "Assumptions"):

Fill   - Hele-Shaw thin-cavity flow: mean front speed  v ~ h^2 * dP/ds / (12 eta).
         We propagate the melt front with the fast-marching method using a speed
         field F(x) = (h(x)/h_nom)^2 so thick sections lead and thin sections lag
         (hesitation / race-tracking), then scale the arrival-time field to the
         requested injection time.
Weld   - Knit (weld) lines are where two flow fronts meet: either fronts from
         different gates (label change) or a single front re-joining around a
         core/hole (flow-direction field converges: div(d) << 0). The front
         meeting angle is reported (0 deg = head-on butt weld, weakest).
Pressure - Integrated Hele-Shaw pressure drop along the flow path
         dP/ds = 12 eta(gamma_dot) v / h_eff^2, with a frozen-layer-reduced h_eff
         and Cross-WLF viscosity at the local shear rate 6v/h.
Cooling  - 1-D slab cooling to ejection temperature:
         t_c = h^2/(pi^2 alpha) ln(4/pi (T_melt-T_mold)/(T_eject-T_mold)).
Warpage  - Linear-elastic voxel-hex FEA with an isotropic eigenstrain field built
         from local shrinkage (thickness- and packing-dependent) plus a
         through-thickness asymmetry term for core/cavity temperature difference.
         Rigid-body motion and uniform shrink are removed to isolate distortion.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage

from .geometry import VoxelDomain
from .materials import Material

try:  # fast marching (preferred)
    import skfmm  # type: ignore
    HAVE_SKFMM = True
except Exception:  # pragma: no cover
    HAVE_SKFMM = False


# 26-connected neighbour offsets
_OFFS = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)])
_OFF_DIST = np.linalg.norm(_OFFS, axis=1)


@dataclass
class ProcessParams:
    injection_time_s: Optional[float] = None    # None -> estimated from part volume
    melt_temp_c: Optional[float] = None         # None -> material default
    mold_temp_c: Optional[float] = None
    eject_temp_c: Optional[float] = None
    gate_diameter_mm: float = 1.5
    core_cavity_dT_c: float = 0.0               # +ve: the +axis mold half is hotter
    mold_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    packing_quality: float = 1.0                # 0..1, 1 = well packed everywhere
    warpage_max_elements: int = 40_000
    weld_div_threshold: float = 0.25            # convergence needed to flag a weld (0..1); 0.25 ~ meeting angle < 150 deg


@dataclass
class SimulationResult:
    domain: VoxelDomain
    material: Material
    params: ProcessParams
    # voxel fields
    fill_time: np.ndarray            # s, nan outside
    gate_label: np.ndarray           # int, -1 outside
    flow_dir: np.ndarray             # (nx,ny,nz,3) unit vectors
    weld_mask: np.ndarray            # bool
    weld_angle: np.ndarray           # deg, front meeting angle (0 head-on)
    air_trap_mask: np.ndarray
    pressure_drop: np.ndarray        # MPa drop from gate to voxel
    front_temp: np.ndarray           # °C, core melt temperature when front arrives
    frozen_fraction: np.ndarray      # fraction of wall frozen at end of fill
    cooling_time: np.ndarray         # s
    shrink_strain: np.ndarray        # eigenstrain magnitude (fraction)
    # warpage (on FE node grid)
    warp_nodes_xyz: Optional[np.ndarray] = None
    warp_disp: Optional[np.ndarray] = None       # distortion only (rigid+uniform scale removed)
    warp_disp_total: Optional[np.ndarray] = None # rigid removed only
    warp_grid: Optional[dict] = None
    summary: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Fill
# --------------------------------------------------------------------------- #
def _gate_mask(dom: VoxelDomain, gate_pt: np.ndarray, gate_d_mm: float) -> np.ndarray:
    ijk = np.array(dom.nearest_solid_index(gate_pt))
    r_vox = max(gate_d_mm / 2.0 / dom.pitch, 1.0)
    rv = int(math.ceil(r_vox))
    shape = np.array(dom.shape)
    lo = np.maximum(ijk - rv, 0)
    hi = np.minimum(ijk + rv + 1, shape)
    gx, gy, gz = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sph = ((gx - ijk[0]) ** 2 + (gy - ijk[1]) ** 2 + (gz - ijk[2]) ** 2) <= r_vox ** 2
    m = np.zeros(dom.shape, dtype=bool)
    m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = sph
    m &= dom.solid
    if not m.any():
        m[tuple(ijk)] = True
    return m


def _travel_time(dom: VoxelDomain, speed: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Arrival time (relative units) from the source voxels through the solid."""
    if HAVE_SKFMM:
        phi = np.ones(dom.shape)
        phi[source] = -1.0
        phi_m = np.ma.MaskedArray(phi, mask=~dom.solid)
        spd = np.where(dom.solid, speed, 1e-9)
        t = skfmm.travel_time(phi_m, spd, dx=float(dom.pitch))
        t = np.ma.filled(t, np.nan).astype(np.float64)
        t[~dom.solid] = np.nan
        t[source] = 0.0
        return t
    return _dijkstra_time(dom, speed, source)


def _dijkstra_time(dom: VoxelDomain, speed: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Fallback when scikit-fmm is unavailable: Dijkstra on the 26-connected voxel graph."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    solid = dom.solid
    ids = -np.ones(dom.shape, dtype=np.int64)
    sol_idx = np.argwhere(solid)
    ids[solid] = np.arange(len(sol_idx))
    rows, cols, w = [], [], []
    shape = np.array(dom.shape)
    for off, d in zip(_OFFS, _OFF_DIST):
        nb = sol_idx + off
        ok = np.all((nb >= 0) & (nb < shape), axis=1)
        a = sol_idx[ok]
        b = nb[ok]
        ib = ids[b[:, 0], b[:, 1], b[:, 2]]
        good = ib >= 0
        a, b, ib = a[good], b[good], ib[good]
        ia = ids[a[:, 0], a[:, 1], a[:, 2]]
        sa = speed[a[:, 0], a[:, 1], a[:, 2]]
        sb = speed[b[:, 0], b[:, 1], b[:, 2]]
        rows.append(ia); cols.append(ib)
        w.append(d * dom.pitch / (0.5 * (sa + sb) + 1e-12))
    n = len(sol_idx)
    G = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    src = ids[source]
    dist = dijkstra(G, directed=False, indices=src, min_only=True)
    t = np.full(dom.shape, np.nan)
    t[solid] = dist
    return t


def _estimate_injection_time(volume_mm3: float) -> float:
    v_cm3 = volume_mm3 / 1000.0
    return float(np.clip(0.4 + 0.015 * v_cm3, 0.5, 8.0))


def _constant_flow_rate_mapping(dom: VoxelDomain, t_rel: np.ndarray, speed: np.ndarray,
                                t_inj: float, n_bins: int = 80):
    """Map fast-marching arrival order to real time assuming constant volumetric flow rate.

    With constant Q the front reaches voxel x when the cumulative filled volume equals
    Q * t, i.e. t(x) = V(t_rel <= t_rel(x)) / Q.  The real front speed is then the FMM
    speed scaled by d(t_rel)/dt, estimated from the arrival-time histogram - fronts slow
    down where the flow front area grows (radial flow) and speed up in constrictions.
    """
    solid = dom.solid
    vox_vol = dom.pitch ** 3
    Q = dom.volume_mm3 / t_inj                          # mm^3/s
    tr = t_rel[solid]
    order = np.argsort(tr, kind="stable")
    cum = np.empty_like(tr)
    cum[order] = (np.arange(1, len(tr) + 1) * vox_vol) / Q
    fill_time = np.full(dom.shape, np.nan)
    fill_time[solid] = cum

    # dt_real/dt_rel per arrival-time bin
    edges = np.linspace(0.0, tr.max() + 1e-12, n_bins + 1)
    counts, _ = np.histogram(tr, bins=edges)
    dt_rel = edges[1] - edges[0]
    dt_real_dt_rel = (counts * vox_vol / Q) / dt_rel    # per bin
    # smooth to avoid histogram noise
    k = np.array([1, 2, 3, 2, 1], float); k /= k.sum()
    dt_real_dt_rel = np.convolve(np.pad(dt_real_dt_rel, 2, mode="edge"), k, mode="valid")
    dt_real_dt_rel = np.maximum(dt_real_dt_rel, 1e-9)
    bin_idx = np.clip(((tr - edges[0]) / dt_rel).astype(int), 0, n_bins - 1)
    front_speed = np.zeros(dom.shape)
    front_speed[solid] = speed[solid] / dt_real_dt_rel[bin_idx]      # mm/s
    return fill_time, front_speed


def _flow_direction(t: np.ndarray, solid: np.ndarray, pitch: float) -> np.ndarray:
    tf = np.where(solid, t, np.nan)
    # fill nan with nearest value so gradient at the boundary is usable
    if (~solid).any():
        _, nearest = ndimage.distance_transform_edt(~solid, return_indices=True)
        tf = tf[nearest[0], nearest[1], nearest[2]]
    g = np.stack(np.gradient(tf, pitch), axis=-1)
    n = np.linalg.norm(g, axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(n > 1e-12, g / n, 0.0)   # front advances toward increasing arrival time
    d[~solid] = 0.0
    return d.astype(np.float32)


# --------------------------------------------------------------------------- #
# Weld lines
# --------------------------------------------------------------------------- #
def _weld_lines(dom: VoxelDomain, t: np.ndarray, d: np.ndarray, labels: np.ndarray,
                gate_mask_all: np.ndarray, threshold: float):
    solid = dom.solid
    shape = dom.shape
    weld = np.zeros(shape, dtype=bool)
    angle = np.full(shape, np.nan, dtype=np.float32)

    # (1) fronts from different gates: label boundary
    if labels.max() > 0:
        for ax in range(3):
            b = np.roll(labels, -1, axis=ax)
            m = solid & (b >= 0) & (labels != b)      # one-sided -> single voxel thick
            weld |= m
    # (2) converging single front: opposing flow directions across the voxel
    best_conv = np.zeros(shape, dtype=np.float32)
    best_dot = np.ones(shape, dtype=np.float32)
    for ax in range(3):
        dm = np.roll(d, 1, axis=ax)   # neighbour at x - e_ax
        dp = np.roll(d, -1, axis=ax)  # neighbour at x + e_ax
        both = np.roll(solid, 1, axis=ax) & np.roll(solid, -1, axis=ax)
        # convergence along this axis: flow at x- pointing +, flow at x+ pointing -
        conv = np.where(both, 0.5 * (dm[..., ax] - dp[..., ax]), 0.0)
        dot = np.einsum("...i,...i->...", dm, dp)
        better = conv > best_conv
        best_conv = np.where(better, conv, best_conv)
        best_dot = np.where(better, dot, best_dot)
    # gradient stencils straddling the part boundary are unreliable: demand a strong,
    # near head-on convergence there, the normal threshold in the interior
    boundary = solid & (dom.edt <= 1.0 * dom.pitch)
    conv_mask = solid & (best_conv >= np.where(boundary, max(threshold, 0.5), threshold))
    weld |= conv_mask
    # meeting angle: 180 - angle between the two incoming flow directions
    ang = 180.0 - np.degrees(np.arccos(np.clip(best_dot, -1, 1)))
    angle[weld] = ang[weld]

    # exclude the gate region itself and the first few % of fill (jetting region)
    near_gate = ndimage.binary_dilation(gate_mask_all, iterations=2)
    weld &= ~near_gate
    tmax = np.nanmax(t)
    weld &= ~(np.nan_to_num(t, nan=0) < 0.03 * tmax)

    # remove speckle
    lab, n = ndimage.label(weld, structure=np.ones((3, 3, 3)))
    if n:
        sizes = ndimage.sum(weld, lab, index=np.arange(1, n + 1))
        small = np.isin(lab, np.where(sizes < 3)[0] + 1)
        weld &= ~small
    angle[~weld] = np.nan
    return weld, angle


def _air_traps(t: np.ndarray, solid: np.ndarray) -> np.ndarray:
    """Last-to-fill pockets: voxels whose arrival time is a local maximum (vent / air-trap candidates)."""
    tf = np.where(solid, t, -np.inf)
    mx = ndimage.maximum_filter(tf, size=5, mode="constant", cval=-np.inf)
    traps = solid & (tf >= mx) & np.isfinite(tf)
    return traps


def air_trap_points(res: "SimulationResult", max_points: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Cluster air-trap voxels (connected, merged within ~2 voxels) and return centroids + fill time."""
    dom = res.domain
    m = ndimage.binary_dilation(res.air_trap_mask, iterations=2) & dom.solid
    lab, n = ndimage.label(m, structure=np.ones((3, 3, 3)))
    if n == 0:
        return np.zeros((0, 3)), np.zeros(0)
    idx = np.arange(1, n + 1)
    cms = np.array(ndimage.center_of_mass(res.air_trap_mask, lab, idx))
    tmax = np.array(ndimage.maximum(np.nan_to_num(res.fill_time, nan=0.0), lab, idx))
    ok = np.isfinite(cms).all(axis=1)
    cms, tmax = cms[ok], tmax[ok]
    if len(cms) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    order = np.argsort(-tmax)[:max_points]
    cms = cms[order]; tmax = tmax[order]
    pts = dom.index_to_world(cms)
    # push onto the skin
    ci = np.clip(np.rint(cms).astype(int), 0, np.array(dom.shape) - 1)
    nrm = dom.normal_dir[ci[:, 0], ci[:, 1], ci[:, 2]]
    pts = pts - nrm * dom.edt[ci[:, 0], ci[:, 1], ci[:, 2]][:, None]
    return pts, tmax


# --------------------------------------------------------------------------- #
# Pressure (upwind path integration)
# --------------------------------------------------------------------------- #
def _upwind_integrate(dom: VoxelDomain, t: np.ndarray, cost_per_mm: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Integrate `cost_per_mm` along fast-marching characteristics (upwind neighbour chain)."""
    solid = dom.solid
    shape = np.array(dom.shape)
    sol_idx = np.argwhere(solid)
    tv = t[solid]
    order = np.argsort(tv, kind="stable")
    sol_idx = sol_idx[order]
    n = len(sol_idx)
    pos = -np.ones(dom.shape, dtype=np.int64)
    pos[sol_idx[:, 0], sol_idx[:, 1], sol_idx[:, 2]] = np.arange(n)

    # upwind neighbour = neighbour with minimum arrival time
    best_t = np.full(n, np.inf)
    best_nb = -np.ones(n, dtype=np.int64)
    best_d = np.zeros(n)
    for off, dd in zip(_OFFS, _OFF_DIST):
        nb = sol_idx + off
        ok = np.all((nb >= 0) & (nb < shape), axis=1)
        tn = np.full(n, np.inf)
        p = -np.ones(n, dtype=np.int64)
        p[ok] = pos[nb[ok, 0], nb[ok, 1], nb[ok, 2]]
        valid = p >= 0
        tn[valid] = t[sol_idx[valid, 0], sol_idx[valid, 1], sol_idx[valid, 2]] * 0 + t[nb[valid, 0], nb[valid, 1], nb[valid, 2]]
        better = tn < best_t
        best_t = np.where(better, tn, best_t)
        best_nb = np.where(better, p, best_nb)
        best_d = np.where(better, dd * dom.pitch, best_d)

    c = cost_per_mm[sol_idx[:, 0], sol_idx[:, 1], sol_idx[:, 2]]
    src = source[sol_idx[:, 0], sol_idx[:, 1], sol_idx[:, 2]]
    P = np.zeros(n)
    tt = t[sol_idx[:, 0], sol_idx[:, 1], sol_idx[:, 2]]
    # sequential accumulation in arrival order (upwind neighbour always earlier -> already computed)
    for i in range(n):
        if src[i]:
            continue
        j = best_nb[i]
        if j < 0 or best_t[i] > tt[i]:
            P[i] = P[i - 1] if i else 0.0
            continue
        P[i] = P[j] + 0.5 * (c[i] + c[j]) * best_d[i]
    out = np.full(dom.shape, np.nan)
    out[sol_idx[:, 0], sol_idx[:, 1], sol_idx[:, 2]] = P
    return out


# --------------------------------------------------------------------------- #
# Warpage FEA (voxel hex, matrix-free PCG)
# --------------------------------------------------------------------------- #
def _hex8_ke_and_bt_d(nu: float, h: float):
    """Element stiffness (E=1) and the eigenstrain load operator for a cube hex8 of edge h.
    Returns ke (24x24) and L (24x6): f_elem = L @ (D @ eps*) * ... we fold D in: f = L_eps @ eps* where eps* is (6,)."""
    E = 1.0
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[3:, 3:] = np.eye(3) * mu
    g = 1 / math.sqrt(3)
    gp = np.array([[-g, -g, -g], [g, -g, -g], [g, g, -g], [-g, g, -g], [-g, -g, g], [g, -g, g], [g, g, g], [-g, g, g]])
    nodes = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    ke = np.zeros((24, 24))
    L = np.zeros((24, 6))
    J = h / 2.0
    detJ = J ** 3
    for xi, eta, zeta in gp:
        dN = np.zeros((8, 3))
        for a in range(8):
            sx, sy, sz = nodes[a]
            dN[a, 0] = sx * (1 + sy * eta) * (1 + sz * zeta) / 8
            dN[a, 1] = sy * (1 + sx * xi) * (1 + sz * zeta) / 8
            dN[a, 2] = sz * (1 + sx * xi) * (1 + sy * eta) / 8
        dN /= J  # physical derivatives
        B = np.zeros((6, 24))
        for a in range(8):
            B[0, 3 * a] = dN[a, 0]
            B[1, 3 * a + 1] = dN[a, 1]
            B[2, 3 * a + 2] = dN[a, 2]
            B[3, 3 * a] = dN[a, 1]; B[3, 3 * a + 1] = dN[a, 0]
            B[4, 3 * a + 1] = dN[a, 2]; B[4, 3 * a + 2] = dN[a, 1]
            B[5, 3 * a] = dN[a, 2]; B[5, 3 * a + 2] = dN[a, 0]
        ke += B.T @ D @ B * detJ
        L += B.T @ D * detJ
    return ke, L


def solve_warpage(dom: VoxelDomain, eps_iso: np.ndarray, eps_asym: np.ndarray, nu: float,
                  max_elements: int, tol: float = 1e-5, max_iter: int = 3000) -> dict:
    """Matrix-free PCG solve of K u = f(eps*) on the voxel hex mesh.

    eps_iso : isotropic eigenstrain (negative = shrink) per voxel
    eps_asym: additional isotropic eigenstrain per voxel (through-thickness antisymmetric term)
    """
    from .geometry import voxelize_mesh
    eps_fine = eps_iso + eps_asym
    n_fine = int(dom.solid.sum())
    if n_fine > max_elements:
        # re-voxelize the B-rep at a coarser pitch so the element budget is honoured
        h = dom.pitch * (n_fine / max_elements) ** (1 / 3)
        solid, origin = voxelize_mesh(dom.mesh, h, pad=1)
        factor = h / dom.pitch
        # sample fine eigenstrain at coarse cell centres (nearest fine voxel, snapped into the solid)
        cc = np.argwhere(solid)
        world = cc * h + origin
        fi = dom.world_to_index(world)
        inside = dom.solid[fi[:, 0], fi[:, 1], fi[:, 2]]
        if not inside.all():
            _, nearest = ndimage.distance_transform_edt(~dom.solid, return_indices=True)
            bad = ~inside
            fi[bad] = nearest[:, fi[bad, 0], fi[bad, 1], fi[bad, 2]].T
        eps = np.zeros(solid.shape)
        eps[cc[:, 0], cc[:, 1], cc[:, 2]] = eps_fine[fi[:, 0], fi[:, 1], fi[:, 2]]
    else:
        solid, origin, h, factor = dom.solid, dom.origin, dom.pitch, 1.0
        eps = eps_fine

    # nodes
    nx, ny, nz = solid.shape
    node_grid = -np.ones((nx + 1, ny + 1, nz + 1), dtype=np.int64)
    el = np.argwhere(solid)
    corner_offs = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    corners = el[:, None, :] + corner_offs[None, :, :]          # (ne, 8, 3)
    flat = corners.reshape(-1, 3)
    used = np.zeros(node_grid.shape, dtype=bool)
    used[flat[:, 0], flat[:, 1], flat[:, 2]] = True
    nn = int(used.sum())
    node_grid[used] = np.arange(nn)
    conn = node_grid[corners[..., 0], corners[..., 1], corners[..., 2]]   # (ne, 8)
    ne = len(el)
    dof = (conn[:, :, None] * 3 + np.arange(3)[None, None, :]).reshape(ne, 24)
    node_xyz = (np.argwhere(used) - 0.5) * h + origin  # corner positions (voxel centre - h/2)

    ke, L = _hex8_ke_and_bt_d(nu, h)
    e = eps[el[:, 0], el[:, 1], el[:, 2]]
    eps_vec = np.zeros((ne, 6))
    eps_vec[:, :3] = e[:, None]
    f_el = eps_vec @ L.T                                          # (ne, 24)
    ndof = 3 * nn
    f = np.bincount(dof.ravel(), weights=f_el.ravel(), minlength=ndof)

    # constraints: 3-2-1 on three well separated nodes to kill rigid body modes
    fixed = np.zeros(ndof, dtype=bool)
    xyz = node_xyz
    i0 = int(np.argmin(xyz[:, 0] + xyz[:, 1] + xyz[:, 2]))
    i1 = int(np.argmax(np.linalg.norm(xyz - xyz[i0], axis=1)))
    v = xyz[i1] - xyz[i0]
    d2 = np.linalg.norm(np.cross(xyz - xyz[i0], v), axis=1)
    i2 = int(np.argmax(d2))
    fixed[3 * i0:3 * i0 + 3] = True
    # i1: fix the two components perpendicular to the dominant axis of v
    ax = int(np.argmax(np.abs(v)))
    for c in range(3):
        if c != ax:
            fixed[3 * i1 + c] = True
    # i2: fix the component normal to the plane (i0,i1,i2)
    nrm = np.cross(v, xyz[i2] - xyz[i0])
    fixed[3 * i2 + int(np.argmax(np.abs(nrm)))] = True
    free = ~fixed

    # ---- assemble sparse K (chunked to bound memory) ---- #
    from scipy import sparse
    K = sparse.csr_matrix((ndof, ndof))
    chunk = 6000
    kef = ke.ravel()
    for s in range(0, ne, chunk):
        d_ = dof[s:s + chunk]
        rows = np.repeat(d_, 24, axis=1).ravel()
        cols = np.tile(d_, (1, 24)).ravel()
        vals = np.tile(kef, d_.shape[0])
        K = K + sparse.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    K.sum_duplicates()
    # constraints by penalty on the diagonal (keeps 3x3 node blocks intact for BSR/AMG)
    pen = 1e6 * float(K.diagonal().mean())
    K = K + sparse.diags(np.where(fixed, pen, 0.0))
    b = f.copy(); b[fixed] = 0.0
    Kb = K.tobsr(blocksize=(3, 3))

    it = 0
    solved = False
    try:
        import pyamg  # algebraic multigrid: robust on thin-walled, badly conditioned meshes
        B = _rigid_body_modes(np.repeat(xyz, 3, axis=0), np.tile(np.arange(3), nn))
        ml = pyamg.smoothed_aggregation_solver(Kb, B=B, max_coarse=300, strength=("symmetric", {"theta": 0.0}),
                                               smooth=("jacobi", {"omega": 4.0 / 3.0}))
        res_hist = []
        u = ml.solve(b, tol=tol, maxiter=400, accel="cg", residuals=res_hist)
        it = len(res_hist)
        solved = True
    except Exception:
        solved = False
    if not solved:
        if ndof < 150_000:
            u = sparse.linalg.spsolve(K.tocsc(), b)
            it = 1
        else:
            Minv = sparse.diags(1.0 / K.diagonal())
            u, info = sparse.linalg.cg(K.tocsr(), b, M=Minv, rtol=tol, maxiter=max_iter)
            it = max_iter if info else -1

    U = u.reshape(-1, 3)
    # remove best-fit rigid body (Procrustes, no scaling) -> total deflection incl. shrink
    U_total = _remove_similarity(xyz, U, allow_scale=False)
    # remove best-fit similarity (rigid + uniform scale) -> distortion / warpage only
    U_warp = _remove_similarity(xyz, U, allow_scale=True)
    return {
        "nodes": xyz.astype(np.float32),
        "disp_total": U_total.astype(np.float32),
        "disp_warp": U_warp.astype(np.float32),
        "node_grid": node_grid, "h": h, "origin": origin, "factor": factor,
        "solid": solid, "iterations": it, "n_elements": int(ne), "n_nodes": int(nn),
    }


def _rigid_body_modes(xyz_dof: np.ndarray, comp: np.ndarray) -> np.ndarray:
    """Six rigid-body modes evaluated per dof (near-nullspace for AMG)."""
    n = len(comp)
    B = np.zeros((n, 6))
    for c in range(3):
        B[comp == c, c] = 1.0
    x, y, z = xyz_dof[:, 0], xyz_dof[:, 1], xyz_dof[:, 2]
    # rotation about z: (-y, x, 0); about x: (0, -z, y); about y: (z, 0, -x)
    B[comp == 0, 3] = -y[comp == 0]; B[comp == 1, 3] = x[comp == 1]
    B[comp == 1, 4] = -z[comp == 1]; B[comp == 2, 4] = y[comp == 2]
    B[comp == 2, 5] = -x[comp == 2]; B[comp == 0, 5] = z[comp == 0]
    return B


def _remove_similarity(X: np.ndarray, U: np.ndarray, allow_scale: bool) -> np.ndarray:
    Y = X + U
    cx, cy = X.mean(0), Y.mean(0)
    X0, Y0 = X - cx, Y - cy
    H = X0.T @ Y0
    Uu, S, Vt = np.linalg.svd(H)
    dsign = np.sign(np.linalg.det(Vt.T @ Uu.T))
    Dm = np.diag([1, 1, dsign])
    R = Vt.T @ Dm @ Uu.T
    s = (S * np.diag(Dm)).sum() / (X0 ** 2).sum() if allow_scale else 1.0
    fit = (s * (R @ X0.T)).T + cy
    return Y - fit


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def simulate(dom: VoxelDomain, material: Material, gates: list[np.ndarray],
             params: ProcessParams | None = None, run_warpage: bool = True) -> SimulationResult:
    params = params or ProcessParams()
    T = dom.thickness.astype(np.float64)
    solid = dom.solid
    pitch = dom.pitch
    timings = {}

    T_melt = params.melt_temp_c if params.melt_temp_c is not None else material.melt_temp_c
    T_mold = params.mold_temp_c if params.mold_temp_c is not None else material.mold_temp_c
    T_ej = params.eject_temp_c if params.eject_temp_c is not None else material.eject_temp_c

    # nominal wall = volume-weighted median thickness
    T_nom = float(np.median(T[solid]))
    T_safe = np.where(solid, np.maximum(T, 0.3 * pitch), pitch)

    # ---------------- fill ---------------- #
    t0 = time.perf_counter()
    speed = np.where(solid, (T_safe / T_nom) ** 2, 1e-9)
    gate_masks = [_gate_mask(dom, np.asarray(g, float), params.gate_diameter_mm) for g in gates]
    if not gate_masks:
        raise ValueError("At least one gate is required.")
    times = [_travel_time(dom, speed, gm) for gm in gate_masks]
    stack = np.stack([np.nan_to_num(ti, nan=np.inf) for ti in times])
    labels = np.where(solid, np.argmin(stack, axis=0), -1).astype(np.int32)
    t_rel = np.min(stack, axis=0)
    t_rel[~solid] = np.nan
    gate_all = np.any(gate_masks, axis=0)

    t_rel_max = float(np.nanmax(t_rel))
    t_inj = params.injection_time_s or _estimate_injection_time(dom.volume_mm3)
    fill_time, front_speed = _constant_flow_rate_mapping(dom, t_rel, speed, t_inj)
    timings["fill"] = time.perf_counter() - t0

    # ---------------- flow direction / welds / traps ---------------- #
    t0 = time.perf_counter()
    d = _flow_direction(fill_time, solid, pitch)
    weld, weld_angle = _weld_lines(dom, fill_time, d, labels, gate_all, params.weld_div_threshold)
    traps = _air_traps(fill_time, solid)
    timings["weld"] = time.perf_counter() - t0

    # ---------------- thermal state during fill ---------------- #
    # Edge / fillet voxels carry a tiny inscribed-sphere thickness; for heat transfer and
    # pressure a cell behaves like the wall it belongs to, so use the local wall thickness
    # (max over a 2-voxel neighbourhood) for these terms. Fill speed keeps the raw field so
    # thin features still hesitate.
    T_wall = ndimage.maximum_filter(np.where(solid, T_safe, 0.0), size=5, mode="nearest")
    T_wall = np.where(solid, np.maximum(T_wall, T_safe), pitch)
    alpha = material.alpha_m2_s * 1e6                           # mm^2/s
    with np.errstate(invalid="ignore", divide="ignore"):
        fo = alpha * np.nan_to_num(fill_time, nan=0.0) / T_wall ** 2    # Fourier number at arrival
        front_temp = T_mold + (T_melt - T_mold) * np.minimum(1.0, (4 / math.pi) * np.exp(-math.pi ** 2 * fo))
        frozen = 2.0 * 0.6 * np.sqrt(alpha * np.nan_to_num(fill_time, nan=0.0)) / T_wall   # fraction of wall frozen
        frozen = np.clip(frozen, 0.0, 0.7)
    front_temp[~solid] = np.nan
    frozen[~solid] = np.nan

    # ---------------- pressure ---------------- #
    t0 = time.perf_counter()
    h_eff = T_wall * (1.0 - frozen)                            # mm
    h_eff = np.where(solid, np.maximum(h_eff, 0.3 * T_wall), 1.0)
    shear = 6.0 * front_speed / h_eff                           # 1/s
    eta = np.vectorize(lambda g: material.viscosity(float(g), T_melt))(np.clip(shear[solid], 1.0, 1e5))
    eta_grid = np.ones(dom.shape)
    eta_grid[solid] = eta
    cost = 12.0 * eta_grid * front_speed / (h_eff ** 2)         # Pa/mm  (Pa.s * mm/s / mm^2)
    cost = cost * 1e-6                                         # MPa/mm
    P_drop = _upwind_integrate(dom, fill_time, np.where(solid, cost, 0.0), gate_all)
    P_drop[~solid] = np.nan
    P_inj = float(np.nanmax(P_drop))
    timings["pressure"] = time.perf_counter() - t0

    # ---------------- cooling ---------------- #
    ratio = (T_melt - T_mold) / max(T_ej - T_mold, 1.0)
    lnterm = math.log(max((4 / math.pi) * ratio, 1.0001))
    cooling = (T_wall ** 2) / (math.pi ** 2 * alpha) * lnterm
    cooling[~solid] = np.nan

    # ---------------- shrinkage -> eigenstrain ---------------- #
    t0 = time.perf_counter()
    s0 = material.shrinkage
    frac_fill = np.nan_to_num(fill_time, nan=0.0) / t_inj
    shrink = (s0
              + material.shrink_thickness_coeff * (T_safe - T_nom)
              + material.shrink_pressure_coeff * frac_fill * (1.0 - 0.6 * params.packing_quality))
    shrink = np.where(solid, np.clip(shrink, 0.2 * s0, 3.0 * s0), 0.0)
    eps_iso = -shrink
    # through-thickness asymmetry: hotter half -> more shrink on that face
    axis = np.asarray(params.mold_axis, float)
    axis /= np.linalg.norm(axis) + 1e-12
    n_dot_a = np.einsum("...i,i->...", dom.normal_dir, axis)   # inward normal . mold axis (+1 on face looking -axis)
    k_asym = 0.012 * s0                                        # +12% shrink per 10°C ... per 1°C = 1.2%
    eps_asym = np.where(solid, -k_asym * params.core_cavity_dT_c * (-n_dot_a), 0.0)

    warp = None
    if run_warpage:
        warp = solve_warpage(dom, eps_iso, eps_asym, material.nu, params.warpage_max_elements)
    timings["warpage"] = time.perf_counter() - t0

    # ---------------- summary ---------------- #
    cycle_open = 4.0
    summary = {
        "material": material.name,
        "melt_temp_c": T_melt, "mold_temp_c": T_mold, "eject_temp_c": T_ej,
        "part_volume_cm3": round(dom.volume_mm3 / 1000.0, 2),
        "voxel_pitch_mm": round(pitch, 3),
        "n_voxels": int(solid.sum()),
        "nominal_wall_mm": round(T_nom, 2),
        "min_wall_mm": round(float(np.percentile(T[solid], 2)), 2),
        "max_wall_mm": round(float(np.percentile(T[solid], 98)), 2),
        "injection_time_s": round(t_inj, 2),
        "max_flow_length_mm": round(float(np.nanmax(t_rel) * np.nanmean(speed[solid])), 1),
        "injection_pressure_mpa": round(P_inj, 1),
        "clamp_tonnage_est_t": None,
        "min_front_temp_c": round(float(np.nanmin(front_temp)), 1),
        "max_frozen_fraction": round(float(np.nanmax(frozen)), 2),
        "cooling_time_s": round(float(np.nanpercentile(cooling, 98)), 1),
        "cycle_time_est_s": round(t_inj + float(np.nanpercentile(cooling, 98)) + cycle_open, 1),
        "weld_voxels": int(weld.sum()),
        "worst_weld_angle_deg": (round(float(np.nanmin(weld_angle)), 0) if weld.any() else None),
        "air_trap_count": int(ndimage.label(ndimage.binary_dilation(traps, iterations=2) & solid, structure=np.ones((3, 3, 3)))[1]),
        "n_gates": len(gates),
    }
    # projected area along mold axis for clamp estimate
    proj = solid.any(axis=int(np.argmax(np.abs(axis))))
    area_mm2 = float(proj.sum()) * pitch ** 2
    summary["projected_area_cm2"] = round(area_mm2 / 100.0, 1)
    summary["clamp_tonnage_est_t"] = round(area_mm2 * 1e-6 * (0.5 * P_inj * 1e6) / 9806.65, 1)  # mean cavity P ~ 0.5 Pinj
    if warp is not None:
        mag = np.linalg.norm(warp["disp_warp"], axis=1)
        summary["max_warpage_mm"] = round(float(mag.max()), 3)
        summary["max_deflection_total_mm"] = round(float(np.linalg.norm(warp["disp_total"], axis=1).max()), 3)
        summary["fea_elements"] = warp["n_elements"]
        summary["fea_iterations"] = warp["iterations"]

    res = SimulationResult(
        domain=dom, material=material, params=params,
        fill_time=fill_time, gate_label=labels, flow_dir=d, weld_mask=weld, weld_angle=weld_angle,
        air_trap_mask=traps, pressure_drop=P_drop, front_temp=front_temp, frozen_fraction=frozen,
        cooling_time=cooling, shrink_strain=shrink,
        warp_nodes_xyz=warp["nodes"] if warp else None,
        warp_disp=warp["disp_warp"] if warp else None,
        warp_disp_total=warp["disp_total"] if warp else None,
        warp_grid=warp, summary=summary, timings={k: round(v, 3) for k, v in timings.items()},
    )
    return res


# --------------------------------------------------------------------------- #
# Surface mapping for the viewer
# --------------------------------------------------------------------------- #
def interpolate_node_field_to_vertices(dom: VoxelDomain, warp: dict, values: np.ndarray) -> np.ndarray:
    """Trilinear interpolation of a per-node field (nn, k) onto the surface mesh vertices."""
    h = warp["h"]; origin = warp["origin"]; node_grid = warp["node_grid"]; solid = warp["solid"]
    V = dom.mesh.vertices
    # cell containing vertex (cells centred at origin + idx*h)
    loc = (V - origin) / h + 0.5
    cell = np.floor(loc).astype(int)
    shape = np.array(solid.shape)
    cell = np.clip(cell, 0, shape - 1)
    inside = solid[cell[:, 0], cell[:, 1], cell[:, 2]]
    if not inside.all():
        _, nearest = ndimage.distance_transform_edt(~solid, return_indices=True)
        bad = ~inside
        c = cell[bad]
        cell[bad] = nearest[:, c[:, 0], c[:, 1], c[:, 2]].T
    frac = np.clip(loc - cell, 0, 1)
    out = np.zeros((len(V), values.shape[1]))
    for dx in (0, 1):
        wx = frac[:, 0] if dx else 1 - frac[:, 0]
        for dy in (0, 1):
            wy = frac[:, 1] if dy else 1 - frac[:, 1]
            for dz in (0, 1):
                wz = frac[:, 2] if dz else 1 - frac[:, 2]
                nid = node_grid[cell[:, 0] + dx, cell[:, 1] + dy, cell[:, 2] + dz]
                nid = np.where(nid < 0, 0, nid)
                out += (wx * wy * wz)[:, None] * values[nid]
    return out


def weld_points(res: SimulationResult, max_points: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """World coordinates of weld voxels (near the surface preferred) and their meeting angle."""
    dom = res.domain
    idx = np.argwhere(res.weld_mask)
    if len(idx) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    # keep voxels within ~1 voxel of the surface so they are visible, fall back to all
    near_surf = dom.edt[idx[:, 0], idx[:, 1], idx[:, 2]] <= 1.6 * dom.pitch
    if near_surf.sum() >= 3:
        idx = idx[near_surf]
    if len(idx) > max_points:
        sel = np.random.default_rng(0).choice(len(idx), max_points, replace=False)
        idx = idx[sel]
    pts = dom.index_to_world(idx)
    # push points slightly outward along the surface normal so they sit on the skin
    n = dom.normal_dir[idx[:, 0], idx[:, 1], idx[:, 2]]
    pts = pts - n * dom.edt[idx[:, 0], idx[:, 1], idx[:, 2]][:, None]
    ang = res.weld_angle[idx[:, 0], idx[:, 1], idx[:, 2]]
    return pts, ang
