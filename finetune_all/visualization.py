"""
Surface-pressure visualization for PhysGen (SP-VAE test outputs and dataset samples).

A car surface is rendered coloured by pressure from several fixed views with a
small CPU rasterizer written in numpy, so it runs on headless cluster nodes
without OpenGL. VTK is only used (lazily) to read the DrivAerNet++ ``.vtk``
files and to write ``.vtp`` files for ParaView.

All geometry lives in the normalized frame used by the PhysGen datasets:
x along the car (front at -x), y lateral, z up, longest side scaled to [-1, 1].
Pressure values are in the dataset's units (kinematic pressure, m^2/s^2).
"""
import csv
import json
import os
import random
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Categories & sampling
# --------------------------------------------------------------------------------------

BODY_TYPES = {"E": "Estateback", "F": "Fastback", "N": "Notchback"}
UNDERBODY_TYPES = {"S": "smooth underbody", "D": "detailed underbody"}


def uid_from_path(path: str) -> str:
    return os.path.basename(str(path)).replace(".npz", "").replace(".vtk", "")


def uid_category(uid: str) -> str:
    """'F_D_WM_WW_0123' -> 'F_D_WM_WW'."""
    return re.sub(r"_\d+$", "", uid_from_path(uid))


def describe_category(uid: str) -> str:
    cat = uid_category(uid)
    parts = cat.split("_")
    desc = [BODY_TYPES.get(parts[0], parts[0])]
    if len(parts) > 1:
        desc.append(UNDERBODY_TYPES.get(parts[1], parts[1]))
    return f"{', '.join(desc)} ({cat})"


def select_vis_uids(uids: Sequence[str], num_samples: int, seed: int = 0) -> List[str]:
    """Stratified pick: round-robin over car categories (largest first), so every
    category is covered once before any category gets a second sample."""
    groups = defaultdict(list)
    for u in uids:
        groups[uid_category(u)].append(uid_from_path(u))
    rng = random.Random(seed)
    for members in groups.values():
        members.sort()
        rng.shuffle(members)
    order = sorted(groups, key=lambda c: (-len(groups[c]), c))
    picked, depth = [], 0
    while len(picked) < num_samples and any(depth < len(groups[c]) for c in order):
        for c in order:
            if depth < len(groups[c]) and len(picked) < num_samples:
                picked.append(groups[c][depth])
        depth += 1
    return picked


# --------------------------------------------------------------------------------------
# Mesh I/O
# --------------------------------------------------------------------------------------

def triangulate_polygons(offsets: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    """Fan-triangulate a VTK polygon cell array (offsets/connectivity layout)."""
    sizes = np.diff(offsets)
    starts = offsets[:-1]
    tris = []
    for k in np.unique(sizes):
        if k < 3:
            continue
        s = starts[sizes == k]
        for j in range(1, k - 1):
            tris.append(np.stack([connectivity[s], connectivity[s + j], connectivity[s + j + 1]], axis=1))
    return np.concatenate(tris, axis=0).astype(np.int64)


def read_vtk_surface(path: str, pressure_key: str = "p") -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Read a DrivAerNet++ PressureVTK file -> (raw points [N,3], triangles [M,3], pressure [N] or None)."""
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOLegacy import vtkPolyDataReader

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    reader = vtkPolyDataReader()
    reader.SetFileName(path)
    reader.ReadAllScalarsOn()
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise IOError(f"Could not read surface from {path}")
    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    cells = poly.GetPolys()
    tris = triangulate_polygons(vtk_to_numpy(cells.GetOffsetsArray()).astype(np.int64),
                                vtk_to_numpy(cells.GetConnectivityArray()).astype(np.int64))
    arr = poly.GetPointData().GetArray(pressure_key)
    pressure = vtk_to_numpy(arr).astype(np.float64).reshape(-1) if arr is not None else None
    return points, tris, pressure


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Same normalization as the pressure caches: bbox centre at 0, longest side -> [-1, 1]."""
    bbmin, bbmax = points.min(0), points.max(0)
    return (points - (bbmin + bbmax) / 2) * (2.0 / (bbmax - bbmin).max())


def write_vtp(path: str, points: np.ndarray, triangles: np.ndarray,
              point_data: Optional[Dict[str, np.ndarray]] = None) -> str:
    """Write a triangle mesh with per-vertex arrays as a compressed .vtp (opens in ParaView / PyVista)."""
    from vtkmodules.util import numpy_support as nps
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pts = vtkPoints()
    pts.SetData(nps.numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float32), deep=True))
    tris = np.ascontiguousarray(triangles).astype(nps.ID_TYPE_CODE)
    offsets = np.arange(0, 3 * len(tris) + 1, 3).astype(nps.ID_TYPE_CODE)
    cells = vtkCellArray()
    cells.SetData(nps.numpy_to_vtkIdTypeArray(offsets, deep=True),
                  nps.numpy_to_vtkIdTypeArray(tris.reshape(-1), deep=True))
    poly = vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cells)
    for i, (name, values) in enumerate((point_data or {}).items()):
        arr = nps.numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float32).reshape(-1), deep=True)
        arr.SetName(name)
        poly.GetPointData().AddArray(arr)
        if i == 0:
            poly.GetPointData().SetActiveScalars(name)
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(poly)
    writer.SetCompressorTypeToZLib()
    writer.SetDataModeToAppended()
    if not writer.Write():
        raise IOError(f"Failed to write {path}")
    return path


# --------------------------------------------------------------------------------------
# Software renderer
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class View:
    name: str
    azim: float  # degrees; 0 = facing the car's front (-x), 90 = its left side (-y)
    elev: float  # degrees; +90 = from above, -90 = from below


DEFAULT_VIEWS = (
    View("Front 3/4", 35, 25),
    View("Side", 90, 0),
    View("Rear 3/4", 145, 25),
    View("Top", 0, 90),
    View("Bottom", 0, -90),
)

# Colours follow the PhysGen paper figures: light-grey shaded meshes next to pressure
# rendered with the jet colormap on a near-white background. The error map (not in the
# paper) is a one-hue light->dark ramp so it cannot be confused with pressure.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MESH_COLOR = "#d2d2d0"
PRESSURE_CMAP = "jet"
ERROR_COLORS = ["#fbe6d9", "#f6b896", "#eb6834", "#b84a1c", "#6e2a10"]
# Linear colour range in m^2/s^2. Over the test set the median surface pressure is about -90,
# the 1st/99th percentiles about -450/+350, and stagnation is ~+450 (0.5 * U^2, U = 30 m/s).
DEFAULT_PRESSURE_RANGE = (-500.0, 400.0)


def _view_basis(view: View) -> np.ndarray:
    """Rows: image-right, image-up, towards-camera (orthographic)."""
    az, el = np.radians(view.azim), np.radians(view.elev)
    to_cam = np.array([-np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)])
    up_hint = np.array([0.0, np.sign(to_cam[2]), 0.0]) if abs(to_cam[2]) > 0.99 else np.array([0.0, 0.0, 1.0])
    forward = -to_cam
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack([right, up, to_cam])


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    vn = np.stack([np.bincount(faces.reshape(-1), weights=np.repeat(fn[:, j], 3), minlength=len(verts))
                   for j in range(3)], axis=1)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    return vn, fn


def _rasterize(verts, faces, basis, frame, px_per_unit, density=4.0, seed=0, chunk=4_000_000,
               depth_tol=0.02):
    """Stochastic rasterization: each triangle gets samples in proportion to its projected
    pixel area, and a z-buffer keeps the sample nearest to the camera in every pixel.
    Pixels the front surface happened to miss (they would show a surface behind it) are
    repaired from their neighbours, and remaining pin-holes are filled.
    Returns (face index map [H,W] (-1 = empty), barycentrics b1, b2 [H,W])."""
    from scipy import ndimage

    x0, x1, y0, y1 = frame
    W = int(np.ceil((x1 - x0) * px_per_unit))
    H = int(np.ceil((y1 - y0) * px_per_unit))
    proj = verts @ basis.T
    px = (proj[:, 0] - x0) * px_per_unit
    py = (y1 - proj[:, 1]) * px_per_unit
    pz = proj[:, 2]
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    area = 0.5 * np.abs((px[b] - px[a]) * (py[c] - py[a]) - (px[c] - px[a]) * (py[b] - py[a]))
    face_ids = np.repeat(np.arange(len(faces)), np.ceil(area * density).astype(np.int64) + 1)

    rng = np.random.default_rng(seed)
    depth = np.full(H * W, -np.inf)
    face_map = np.full(H * W, -1, dtype=np.int64)
    b1_map = np.zeros(H * W, dtype=np.float32)
    b2_map = np.zeros(H * W, dtype=np.float32)
    for s in range(0, len(face_ids), chunk):
        f = face_ids[s:s + chunk]
        r1, r2 = rng.random(len(f)), rng.random(len(f))
        flip = r1 + r2 > 1
        r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
        ia, ib, ic = a[f], b[f], c[f]
        sx = px[ia] + r1 * (px[ib] - px[ia]) + r2 * (px[ic] - px[ia])
        sy = py[ia] + r1 * (py[ib] - py[ia]) + r2 * (py[ic] - py[ia])
        sz = pz[ia] + r1 * (pz[ib] - pz[ia]) + r2 * (pz[ic] - pz[ia])
        ix, iy = sx.astype(np.int64), sy.astype(np.int64)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        pix = np.where(ok, iy * W + ix, -1)
        order = np.lexsort((-sz, pix))
        order = order[pix[order] >= 0]
        first = np.ones(len(order), dtype=bool)
        first[1:] = pix[order][1:] != pix[order][:-1]
        sel = order[first]
        sel = sel[sz[sel] > depth[pix[sel]]]
        p_sel = pix[sel]
        depth[p_sel] = sz[sel]
        face_map[p_sel] = f[sel]
        b1_map[p_sel] = r1[sel]
        b2_map[p_sel] = r2[sel]

    face_map, b1_map, b2_map = face_map.reshape(H, W), b1_map.reshape(H, W), b2_map.reshape(H, W)
    depth = depth.reshape(H, W)
    # A pixel much farther than its neighbourhood shows a hidden surface through a sampling
    # gap: replace it with the nearest-to-camera fragment among its 3x3 neighbours.
    for _ in range(2):
        pad = np.pad(depth, 1, constant_values=-np.inf)
        shifts = np.stack([pad[dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)])
        finite = np.isfinite(shifts)
        with warnings.catch_warnings():  # all-NaN windows in the background are expected
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.where(finite, shifts, np.nan), axis=0)
        bad = np.isfinite(depth) & (finite.sum(0) >= 6) & (depth < med - depth_tol)
        if not bad.any():
            break
        yy, xx = np.nonzero(bad)
        dy, dx = np.divmod(shifts.argmax(0)[yy, xx], 3)
        sy, sx = yy + dy - 1, xx + dx - 1
        face_map[yy, xx], b1_map[yy, xx], b2_map[yy, xx] = face_map[sy, sx], b1_map[sy, sx], b2_map[sy, sx]
        depth[yy, xx] = depth[sy, sx]
    # Fill pin-holes left by the sampling with the nearest covered pixel.
    mask = face_map >= 0
    holes = ndimage.binary_closing(mask, structure=np.ones((3, 3), bool)) & ~mask
    if holes.any():
        _, (ny, nx) = ndimage.distance_transform_edt(~mask, return_indices=True)
        hy, hx = np.nonzero(holes)
        face_map[hy, hx] = face_map[ny[hy, hx], nx[hy, hx]]
        b1_map[hy, hx] = b1_map[ny[hy, hx], nx[hy, hx]]
        b2_map[hy, hx] = b2_map[ny[hy, hx], nx[hy, hx]]
    return face_map, b1_map, b2_map


def _shade(raster, faces, vnormals, fnormals, scalars, basis, cmap=None, norm=None, color=MESH_COLOR):
    """Shade a raster; colour by per-vertex ``scalars`` through ``cmap``/``norm``, or uniformly
    with ``color`` when ``scalars`` is None."""
    face_map, b1_map, b2_map = raster
    H, W = face_map.shape
    img = np.empty((H, W, 3))
    img[:] = _hex_to_rgb(SURFACE)
    mask = face_map >= 0
    f = face_map[mask]
    b1, b2 = b1_map[mask].astype(np.float64), b2_map[mask].astype(np.float64)
    b0 = 1.0 - b1 - b2
    tri = faces[f]
    n = b0[:, None] * vnormals[tri[:, 0]] + b1[:, None] * vnormals[tri[:, 1]] + b2[:, None] * vnormals[tri[:, 2]]
    length = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(length > 0.3, n / np.maximum(length, 1e-12), fnormals[f])
    to_cam = basis[2]
    n *= np.where(n @ to_cam < 0, -1.0, 1.0)[:, None]  # two-sided lighting
    light = to_cam + 0.6 * basis[1] - 0.35 * basis[0]
    light /= np.linalg.norm(light)
    half = light + to_cam
    half /= np.linalg.norm(half)
    diffuse = np.clip(n @ light, 0.0, 1.0)
    specular = np.clip(n @ half, 0.0, 1.0) ** 40
    if scalars is None:
        rgb = np.broadcast_to(_hex_to_rgb(color), (len(f), 3))
    else:
        value = b0 * scalars[tri[:, 0]] + b1 * scalars[tri[:, 1]] + b2 * scalars[tri[:, 2]]
        rgb = np.asarray(cmap(norm(value)))[:, :3]
    img[mask] = np.clip(rgb * (0.5 + 0.5 * diffuse)[:, None] + 0.1 * specular[:, None], 0.0, 1.0)
    return img


def _downsample(img: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return img
    H, W = (img.shape[0] // factor) * factor, (img.shape[1] // factor) * factor
    return img[:H, :W].reshape(H // factor, factor, W // factor, factor, 3).mean(axis=(1, 3))


def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _cmaps():
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap
    return colormaps[PRESSURE_CMAP], LinearSegmentedColormap.from_list("physgen_error", ERROR_COLORS)


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------

@dataclass
class SurfaceRow:
    """One row of the figure: a surface coloured by a per-vertex scalar."""
    label: str
    verts: np.ndarray
    faces: np.ndarray
    scalars: np.ndarray
    kind: str = "pressure"  # "pressure" (jet) or "error" (one-hue ramp)
    show_mesh: bool = True  # also show the plain grey mesh in the first column


def render_surface_figure(rows: Sequence[SurfaceRow], out_path: str, title: str, subtitle: str = "",
                          views: Sequence[View] = DEFAULT_VIEWS, mesh_view: Optional[View] = None,
                          px_per_unit: float = 260, supersample: int = 2,
                          pressure_range: Tuple[float, float] = DEFAULT_PRESSURE_RANGE,
                          error_max: Optional[float] = None, dpi: int = 100) -> str:
    """Render one PNG: per row a grey mesh (paper style) followed by the surface coloured by
    pressure (or error) from each of ``views``."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter

    p_cmap, e_cmap = _cmaps()
    p_norm = Normalize(vmin=pressure_range[0], vmax=pressure_range[1])
    if error_max is None:
        errs = [r.scalars for r in rows if r.kind == "error"]
        error_max = float(np.percentile(np.concatenate(errs), 99)) if errs else 1.0
    e_norm = Normalize(vmin=0.0, vmax=max(error_max, 1e-6))

    mesh_view = mesh_view or views[0]
    has_mesh_col = any(r.show_mesh for r in rows)
    all_views = list(views) + ([] if mesh_view in views else [mesh_view])
    mesh_idx = all_views.index(mesh_view)
    bases = [_view_basis(v) for v in all_views]
    frames = []
    for basis in bases:
        proj = np.concatenate([r.verts @ basis[:2].T for r in rows])
        lo, hi = proj.min(0), proj.max(0)
        pad = 0.03 * (hi - lo).max()
        frames.append((lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad))

    # Rasterize each distinct mesh once per view; rows sharing a mesh reuse it.
    geom_cache, images = {}, []
    for r in rows:
        key = (id(r.verts), id(r.faces))
        if key not in geom_cache:
            vn, fn = _vertex_normals(r.verts, r.faces)
            rasters = [_rasterize(r.verts, r.faces, basis, frame, px_per_unit * supersample)
                       for basis, frame in zip(bases, frames)]
            geom_cache[key] = (vn, fn, rasters)
        vn, fn, rasters = geom_cache[key]
        cmap, norm = (p_cmap, p_norm) if r.kind == "pressure" else (e_cmap, e_norm)
        cells = []
        if has_mesh_col:
            cells.append(_downsample(_shade(rasters[mesh_idx], r.faces, vn, fn, None, bases[mesh_idx]), supersample)
                         if r.show_mesh else None)
        cells += [_downsample(_shade(rasters[j], r.faces, vn, fn, r.scalars, bases[j], cmap, norm), supersample)
                  for j in range(len(views))]
        images.append(cells)

    # Layout in pixels.
    headers = ([("Mesh", mesh_view.name)] if has_mesh_col else []) + [("Surface pressure", v.name) for v in views]
    n_cols = len(headers)
    col_w = [max(images[i][j].shape[1] for i in range(len(rows)) if images[i][j] is not None) for j in range(n_cols)]
    row_h = max(img.shape[0] for row in images for img in row if img is not None)
    left, right, top, bottom, gap_c, gap_r, head = 215, 150, 84, 18, 10, 14, 48
    mesh_gap = 28 if has_mesh_col else 0  # extra space between the mesh column and the pressure views
    col_x, x = [], left
    for j in range(n_cols):
        col_x.append(x)
        x += col_w[j] + (gap_c + mesh_gap if (has_mesh_col and j == 0) else gap_c)
    W = x - gap_c + right
    H = top + head + len(rows) * row_h + gap_r * (len(rows) - 1) + bottom
    fig = Figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=SURFACE)
    FigureCanvasAgg(fig)

    def axes_px(x, y, w, h):
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

    fig.text(left / W, 1 - 30 / H, title, fontsize=17, color=TEXT_PRIMARY, weight="semibold", va="baseline")
    if subtitle:
        fig.text(left / W, 1 - 58 / H, subtitle, fontsize=12, color=TEXT_SECONDARY, va="baseline")
    # Two-level column headers: group ("Mesh" / "Surface pressure") over view names.
    j = 0
    while j < n_cols:
        k = j
        while k + 1 < n_cols and headers[k + 1][0] == headers[j][0]:
            k += 1
        x0, x1 = col_x[j], col_x[k] + col_w[k]
        fig.text((x0 + x1) / 2 / W, 1 - (top + 14) / H, headers[j][0], fontsize=12, color=TEXT_PRIMARY,
                 ha="center", va="baseline", weight="semibold")
        if k > j:
            line = axes_px(x0, top + 22, x1 - x0, 1)
            line.set_axis_off()
            line.axhline(0.5, color="#c3c2b7", lw=1)
        j = k + 1
    for j in range(n_cols):
        fig.text((col_x[j] + col_w[j] / 2) / W, 1 - (top + head - 8) / H, headers[j][1], fontsize=11,
                 color=TEXT_SECONDARY, ha="center", va="baseline")

    row_y = []
    for i, r in enumerate(rows):
        y = top + head + i * (row_h + gap_r)
        row_y.append(y)
        fig.text((left - 16) / W, 1 - (y + row_h / 2) / H, r.label, fontsize=12, color=TEXT_PRIMARY,
                 ha="right", va="center", linespacing=1.4)
        for j in range(n_cols):
            img = images[i][j]
            if img is None:
                continue
            ax = axes_px(col_x[j] + (col_w[j] - img.shape[1]) / 2, y + (row_h - img.shape[0]) / 2,
                         img.shape[1], img.shape[0])
            ax.imshow(img, interpolation="none")
            ax.set_axis_off()

    # One colour bar per run of consecutive rows of the same kind.
    i = 0
    while i < len(rows):
        k = i
        while k + 1 < len(rows) and rows[k + 1].kind == rows[i].kind:
            k += 1
        y0, y1 = row_y[i] + 8, row_y[k] + row_h - 8
        cax = axes_px(W - right + 22, y0, 14, y1 - y0)
        if rows[i].kind == "pressure":
            sm, extend, label = ScalarMappable(norm=p_norm, cmap=p_cmap), "both", "pressure p  [m²/s²]"
        else:
            sm, extend, label = ScalarMappable(norm=e_norm, cmap=e_cmap), "max", "|p_pred − p_GT|  [m²/s²]"
        cb = fig.colorbar(sm, cax=cax, extend=extend)
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=10, colors=TEXT_SECONDARY, length=3)
        cb.set_label(label, fontsize=11, color=TEXT_SECONDARY)
        i = k + 1

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE)
    return out_path


# --------------------------------------------------------------------------------------
# High-level entry points
# --------------------------------------------------------------------------------------

def _rel_l2(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm(pred - gt) / max(np.linalg.norm(gt), 1e-12))


def save_test_visualization(out_dir: str, uid: str, gt_vtk_path: str,
                            gt_points: np.ndarray, gt_pressure: np.ndarray, pred_pressure_on_gt: np.ndarray,
                            pred_verts: Optional[np.ndarray], pred_faces: Optional[np.ndarray],
                            pred_pressure_on_pred: Optional[np.ndarray],
                            cd_gt: Optional[float] = None, cd_pred: Optional[float] = None,
                            shape_iou: Optional[float] = None, **render_kwargs) -> Dict:
    """Save one SP-VAE test sample: meshes with pressure (.vtp), metrics (.json) and a PNG with
    rows [GT pressure on the CFD surface, predicted pressure on the reconstructed shape,
    |error| on the CFD surface]. Pressures are in dataset units (not normalized)."""
    sample_dir = os.path.join(out_dir, uid)
    os.makedirs(sample_dir, exist_ok=True)

    # The full-resolution pressure cache stores the VTK vertices in their original order,
    # so the VTK triangles index the test points directly; otherwise map by nearest neighbour.
    raw_points, tris, _ = read_vtk_surface(gt_vtk_path)
    gt_verts = np.asarray(gt_points, dtype=np.float64)
    if len(raw_points) != len(gt_verts):
        from scipy.spatial import cKDTree
        idx = cKDTree(gt_verts).query(normalize_points(raw_points))[1]
        gt_verts, gt_pressure, pred_pressure_on_gt = normalize_points(raw_points), gt_pressure[idx], pred_pressure_on_gt[idx]
    abs_err = np.abs(pred_pressure_on_gt - gt_pressure)

    metrics = {
        "uid": uid,
        "category": uid_category(uid),
        "cd_gt": cd_gt,
        "cd_pred": cd_pred,
        "cd_abs_err": None if cd_gt is None or cd_pred is None else abs(cd_pred - cd_gt),
        "pressure_mae": float(abs_err.mean()),
        "pressure_rmse": float(np.sqrt((abs_err ** 2).mean())),
        "pressure_rel_l2": _rel_l2(pred_pressure_on_gt, gt_pressure),
        "shape_iou": shape_iou,
        "n_gt_vertices": int(len(gt_verts)),
        "n_pred_vertices": None if pred_verts is None else int(len(pred_verts)),
    }

    write_vtp(os.path.join(sample_dir, "gt_surface_pressure.vtp"), gt_verts, tris,
              {"p_gt": gt_pressure, "p_pred": pred_pressure_on_gt, "abs_err": abs_err})
    rows = [SurfaceRow("Ground truth\nCFD surface", gt_verts, tris, gt_pressure)]
    if pred_verts is not None and len(pred_verts) > 0:
        write_vtp(os.path.join(sample_dir, "pred_shape_pressure.vtp"), pred_verts, pred_faces,
                  {"p_pred": pred_pressure_on_pred})
        rows.append(SurfaceRow("Prediction\nreconstructed shape", pred_verts, pred_faces, pred_pressure_on_pred))
    rows.append(SurfaceRow("Absolute error\non CFD surface", gt_verts, tris, abs_err, kind="error", show_mesh=False))
    with open(os.path.join(sample_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    parts = []
    if cd_gt is not None and cd_pred is not None:
        parts.append(f"Cd  GT {cd_gt:.4f} · pred {cd_pred:.4f} (Δ {cd_pred - cd_gt:+.4f})")
    parts.append(f"pressure  MAE {metrics['pressure_mae']:.1f} m²/s² · rel. L2 {metrics['pressure_rel_l2']:.3f}")
    if shape_iou is not None:
        parts.append(f"shape IoU {shape_iou:.3f}")
    render_surface_figure(rows, os.path.join(out_dir, f"{uid}.png"), f"{uid}  ·  {describe_category(uid)}",
                          "      ".join(parts), **render_kwargs)

    summary = os.path.join(out_dir, "summary.csv")
    new = not os.path.exists(summary)
    with open(summary, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics))
        if new:
            w.writeheader()
        w.writerow(metrics)
    return metrics


def render_dataset_sample(uid: str, data_root: str, out_path: str, split_dir: Optional[str] = None,
                          csv_path: Optional[str] = None, **render_kwargs) -> str:
    """Render a dataset sample (GT pressure on the CFD surface) in the same format as the test figures."""
    uid = uid_from_path(uid)
    raw_points, tris, pressure = read_vtk_surface(os.path.join(data_root, "pressure", "PressureVTK", f"{uid}.vtk"))
    if pressure is None:
        raise KeyError(f"No point array 'p' in the VTK file of {uid}")
    verts = normalize_points(raw_points)

    parts = []
    csv_path = csv_path or os.path.join(data_root, "drag", "DrivAerNetPlusPlus_Cd_8k_Updated.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for rec in csv.reader(f):
                if rec and rec[0] == uid:
                    parts.append(f"Cd {float(rec[1]):.4f}")
                    break
    if split_dir:
        for split in ("train", "val", "test"):
            path = os.path.join(split_dir, f"{split}.json")
            if os.path.exists(path):
                with open(path) as f:
                    if uid in {uid_from_path(p) for p in json.load(f)}:
                        parts.append(f"split: {split}")
                        break
    parts.append(f"{len(verts):,} vertices · p ∈ [{pressure.min():,.0f}, {pressure.max():,.0f}] m²/s²")
    rows = [SurfaceRow("Ground truth\nCFD surface", verts, tris, pressure)]
    return render_surface_figure(rows, out_path, f"{uid}  ·  {describe_category(uid)}", "      ".join(parts),
                                 **render_kwargs)
