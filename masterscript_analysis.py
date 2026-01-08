#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: L. Gonzalo Espinoza-Arcos

What this script produces (CSV + plots)
--------------------------------------
1) distances_wide.csv
   COM(F216) → closest membrane phosphate distance (nm), wide format (rep1..repN + MEAN)

2) deformation_wide.csv
   Membrane thickness deformation relative to frame 0 (Δ thickness, nm), wide format (rep1..repN + MEAN)

3) exchange_rate_np2mem_types_wide.csv
   NP→Mem exchange rate by nanoparticle lipid type (replicate-average), in events/µs

4) exchange_rate_mem2np_top_types_wide.csv
   Mem→NP exchange rate (upper leaflet only) by membrane lipid type (replicate-average), in events/µs

5) exchange_cumulative_np2mem_types_wide.csv
   Cumulative NP→Mem exchange by nanoparticle lipid type (replicate-average) — monotonic increasing curves

6) exchange_cumulative_mem2np_top_types_wide.csv
   Cumulative Mem→NP (upper leaflet) exchange by membrane lipid type (replicate-average)

Typical usage
-------------
python3 deff.py -s box_final.gro -t rep1.xtc rep2.xtc rep3.xtc \
  --stride 5 --tmin-us 0 --tmax-us 5 \
  --np2mem-cutoff-nm 0.72 --mem2np-cutoff-nm 0.60 \
  --tau-ns 20 --rate-window-ns 20 --plot-smooth 1

Notes on definitions (kept explicit on purpose)
----------------------------------------------
- "NP→Mem": an NP-coated lipid is counted as exchanged once it maintains contact with membrane PO4
  for at least τ (residence time threshold).
- "Mem→NP (TOP)": a membrane lipid (upper leaflet) is counted as exchanged once it maintains contact
  with the NP for at least τ.
- Rates are computed from the cumulative curves using a moving window (rate-window-ns), reported as events/µs.
"""

import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.lib import distances


# -------------------- CLI --------------------
p = argparse.ArgumentParser(
    description="Wide CSVs (X=time_us) + lightweight diagnostic plots for distance, deformation, and exchange kinetics."
)
p.add_argument("-s", "--structure", required=True, help="Structure file (.gro/.tpr/.pdb)")
p.add_argument("-t", "--trajectories", nargs="+", required=True, help="Trajectory files .xtc (≥1 replica)")
p.add_argument("--stride", type=int, default=1, help="Frame stride (default: 1)")
p.add_argument("--tmin-us", type=float, default=0.0, help="Minimum time (µs) for CSV/plots (default: 0.0)")
p.add_argument("--tmax-us", type=float, default=5.0, help="Maximum time (µs) for CSV/plots (default: 5.0)")

# Exchange / residence parameters
p.add_argument(
    "--np2mem-cutoff-nm",
    type=float,
    default=0.72,
    help="Cutoff (nm) for NP→Mem contacts (NP lipid ↔ membrane PO4)",
)
p.add_argument(
    "--mem2np-cutoff-nm",
    type=float,
    default=0.60,
    help="Cutoff (nm) for Mem→NP contacts (membrane lipid ↔ NP); upper leaflet only",
)
p.add_argument("--tau-ns", type=float, default=20.0, help="Minimum residence time (ns) to count an exchange event")
p.add_argument(
    "--rate-window-ns",
    type=float,
    default=20.0,
    help="Window (ns) used to estimate a rate from the cumulative curves",
)

# Plot cosmetics (diagnostic only)
p.add_argument(
    "--plot-smooth",
    type=int,
    default=1,
    help="Odd window size for moving-average smoothing in plots (1 = no smoothing).",
)
args = p.parse_args()


# -------------------- Selections --------------------
MEM_LIPIDS = ["POPC", "POPE", "POPS", "POSM", "CHOL"]
NP_LIPIDS = ["PXPC", "PXPE", "PXPS", "PXSM", "CHXL"]


def get_nanoparticle(u: mda.Universe):
    """Try to grab the NP group. Primary: resname F216. Fallback: resid-based selection."""
    g = u.select_atoms("resname F216")
    if g.n_atoms > 0:
        return g

    # Fallback: try a conservative guess (kept from the original logic).
    g = u.select_atoms("resid 216 and not (resname " + " ".join(MEM_LIPIDS) + ")")
    if g.n_atoms == 0:
        raise ValueError("Nanoparticle group not found (expected resname F216).")
    return g


def get_mem_po4(u: mda.Universe):
    """Return membrane phosphates. Try common Martini/GROMACS naming variants."""
    sel_res = "resname " + " ".join(MEM_LIPIDS)
    for sel in (f"({sel_res}) and type PO4", f"({sel_res}) and name PO4", f"({sel_res}) and name P"):
        try:
            g = u.select_atoms(sel)
            if g.n_atoms > 0:
                return g
        except Exception:
            pass
    raise ValueError("Membrane phosphates not found (tried: type PO4 / name PO4 / name P).")


def get_np_lipids(u: mda.Universe):
    return u.select_atoms("resname " + " ".join(NP_LIPIDS))


def get_membrane_lipids(u: mda.Universe):
    return u.select_atoms("resname " + " ".join(MEM_LIPIDS))


def wrap_delta(dz, Lz):
    """Minimum-image convention along z (works if Lz is known)."""
    return dz - Lz * np.rint(dz / Lz) if (Lz and Lz > 0.0) else dz


def capped_pairs(Apos, Bpos, cutoff_nm, box):
    """
    Return robust 1-D index arrays (i, j) for A-B contacts within cutoff_nm (nm).
    MDAnalysis uses Å internally, hence the ×10 conversion.
    """
    maxcut = cutoff_nm * 10.0  # Å

    # MDAnalysis changed argument names across versions; handle both.
    try:
        out = distances.capped_distance(Apos, Bpos, maxcut, box=box, return_distances=False)
    except TypeError:
        out = distances.capped_distance(Apos, Bpos, max_cutoff=maxcut, box=box, return_distances=False)

    # Depending on version, the return can be a tuple (idxA, idxB) or an Nx2 array.
    if isinstance(out, tuple) and len(out) >= 2:
        i = np.asarray(out[0]).ravel().astype(int, copy=False)
        j = np.asarray(out[1]).ravel().astype(int, copy=False)
        return i, j

    arr = np.asarray(out)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 0].astype(int, copy=False), arr[:, 1].astype(int, copy=False)

    return np.array([], dtype=int), np.array([], dtype=int)


def rolling_mean(y, w):
    """Centered moving average (odd window enforced)."""
    w = max(1, int(w))
    if w % 2 == 0:
        w += 1
    if w == 1:
        return y
    k = np.ones(w, dtype=float) / w
    return np.convolve(y, k, mode="same")


def rate_from_cumulative(time_us, cum_counts, win_ns):
    """Estimate rate (events/µs) using a sliding time window (ns)."""
    if len(time_us) == 0:
        return np.array([])

    win_us = float(win_ns) / 1000.0
    t = np.asarray(time_us, float)
    c = np.asarray(cum_counts, float)

    out = np.full_like(t, np.nan, dtype=float)
    j = 0
    for k in range(len(t)):
        while j < k and (t[k] - t[j]) > win_us:
            j += 1
        span = t[k] - t[j]
        if span > 0:
            out[k] = (c[k] - c[j]) / span
    return out


def build_common_grid(tmin, tmax, reps_times):
    """
    Build a uniform common time grid using the median Δt across replicas.
    A few safeguards keep the output CSVs manageable.
    """
    dts = []
    for t in reps_times:
        if len(t) > 1:
            dts.append(np.median(np.diff(np.asarray(t, float))))

    if len(dts) == 0:
        return np.linspace(tmin, tmax, 1000)

    dt = float(np.median(dts))
    npoints = int(np.clip(np.round((tmax - tmin) / max(dt, 1e-9)) + 1, 200, 5000))
    return np.linspace(tmin, tmax, npoints)


def write_wide_csv(fname, grid_t, series_dict, include_mean=True):
    """
    Wide CSV: columns = time_us, <series keys...>, [MEAN]
    series_dict: { name: (t, y) }
    """
    names = list(series_dict.keys())
    mat = []

    for nm in names:
        t, y = series_dict[nm]
        t = np.asarray(t, float)
        y = np.asarray(y, float)
        yi = np.interp(grid_t, t, y, left=np.nan, right=np.nan)
        mat.append(yi)

    M = np.vstack(mat) if mat else np.empty((0, grid_t.size))
    mean = np.nanmean(M, axis=0) if (include_mean and M.size) else None

    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        header = ["time_us"] + names + (["MEAN"] if include_mean else [])
        w.writerow(header)

        for i, ti in enumerate(grid_t):
            row = [float(ti)]
            for k in range(len(names)):
                val = M[k, i] if M.size else np.nan
                row.append(float(val) if np.isfinite(val) else "")
            if include_mean:
                mv = mean[i] if mean is not None else np.nan
                row.append(float(mv) if np.isfinite(mv) else "")
            w.writerow(row)


def plot_lines(grid_t, series_dict, title, ylabel, outfile, smooth=1, add_mean=True):
    """Minimal plot helper for quick sanity checks."""
    plt.figure(figsize=(9, 5.5))

    for name, (t, y) in series_dict.items():
        yi = np.interp(grid_t, np.asarray(t, float), np.asarray(y, float), left=np.nan, right=np.nan)
        yi_plot = rolling_mean(yi, smooth) if smooth > 1 else yi
        plt.plot(grid_t, yi_plot, lw=1.9, label=name, alpha=0.95)

    if add_mean and len(series_dict) > 1:
        mat = []
        for _, (t, y) in series_dict.items():
            yi = np.interp(grid_t, np.asarray(t, float), np.asarray(y, float), left=np.nan, right=np.nan)
            mat.append(yi)
        M = np.vstack(mat)
        mean = np.nanmean(M, axis=0)
        mean_plot = rolling_mean(mean, smooth) if smooth > 1 else mean
        plt.plot(grid_t, mean_plot, lw=2.6, color="k", label="MEAN")

    for sp in ("top", "right"):
        plt.gca().spines[sp].set_visible(False)
    plt.gca().spines["left"].set_linewidth(1.6)
    plt.gca().spines["bottom"].set_linewidth(1.6)

    plt.grid(True, axis="y", color="0.92")
    plt.xlabel("Time (µs)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


# -------------------- Per-replica analysis --------------------
def analyze_rep(structure, traj, stride, tau_ns, cut_np2mem_nm, cut_mem2np_nm):
    u = mda.Universe(structure, traj)
    NP = get_nanoparticle(u)
    PO4 = get_mem_po4(u)
    mem = get_membrane_lipids(u)
    npL = get_np_lipids(u)

    # Baseline thickness at frame 0 (difference between average upper vs lower leaflet PO4 z positions).
    u.trajectory[0]
    box = u.trajectory.ts.dimensions
    Lz = float(box[2]) if (box is not None and len(box) >= 3) else 0.0

    z_mid0 = float(PO4.center_of_mass(wrap=True)[2])
    dz0 = wrap_delta(PO4.positions[:, 2] - z_mid0, Lz)
    th0 = np.nanmean(dz0[dz0 >= 0]) - np.nanmean(dz0[dz0 < 0])

    # Residue maps + lipid types (kept as arrays for fast masking)
    np_residues = list(npL.residues)
    np_resids = np.array([res.resid for res in np_residues], dtype=int)
    np_types = np.array([res.resname for res in np_residues], dtype=object)
    np_resid_to_idx = {r: i for i, r in enumerate(np_resids)}
    uniq_np_types = [t for t in NP_LIPIDS if t in set(np_types)]

    mem_residues = list(mem.residues)
    mem_resids = np.array([res.resid for res in mem_residues], dtype=int)
    mem_types = np.array([res.resname for res in mem_residues], dtype=object)
    mem_resid_to_idx = {r: i for i, r in enumerate(mem_resids)}
    uniq_mem_types = [t for t in MEM_LIPIDS if t in set(mem_types)]

    PO4_resids_arr = np.asarray(PO4.resids, dtype=int)

    # Time series
    t_us, dist_nm, deform_nm = [], [], []
    flags_np2mem = np.zeros((len(np_resids), 0), dtype=bool)
    flags_mem2np_top = np.zeros((len(mem_resids), 0), dtype=bool)
    times_ns = []

    for ts in u.trajectory[::stride]:
        box = ts.dimensions
        Lz = float(box[2]) if (box is not None and len(box) >= 3) else 0.0

        t_us.append(ts.time / 1e6)
        times_ns.append(ts.time / 1000.0)

        # COM distance (NP COM -> minimum distance to any membrane PO4)
        try:
            com_np = NP.center_of_mass(wrap=True)
        except Exception:
            com_np = NP.center_of_geometry(wrap=True)

        dcom = distances.distance_array(com_np[None, :], PO4.positions, box=box)[0]
        dist_nm.append(float(np.min(dcom)) / 10.0)

        # Thickness deformation relative to frame 0
        z_mid = float(PO4.center_of_mass(wrap=True)[2])
        dz = wrap_delta(PO4.positions[:, 2] - z_mid, Lz)
        th = np.nanmean(dz[dz >= 0]) - np.nanmean(dz[dz < 0])
        deform_nm.append((th - th0) / 10.0)

        # NP→Mem: NP-lipid atoms vs membrane PO4 atoms
        i_np2, _j_po4 = capped_pairs(npL.positions, PO4.positions, cut_np2mem_nm, box)
        col_np = np.zeros(len(np_resids), dtype=bool)
        if i_np2.size > 0:
            res_hit = np.unique(npL.atoms[i_np2].resids)
            for r in res_hit:
                idx = np_resid_to_idx.get(int(r), None)
                if idx is not None:
                    col_np[idx] = True
        flags_np2mem = col_np[:, None] if flags_np2mem.size == 0 else np.hstack([flags_np2mem, col_np[:, None]])

        # Mem→NP (TOP): NP atoms vs membrane atoms; keep only upper leaflet contacts
        _i_np3, j_mem3 = capped_pairs(NP.positions, mem.positions, cut_mem2np_nm, box)
        col_mem_top = np.zeros(len(mem_resids), dtype=bool)
        if j_mem3.size > 0:
            res_contact = np.unique(mem.atoms[j_mem3].resids)
            if res_contact.size > 0:
                dz_po4 = wrap_delta(PO4.positions[:, 2] - com_np[2], Lz)
                res_top_by_po4 = set(PO4_resids_arr[dz_po4 > 0.0])

                res_contact_set = set(int(r) for r in res_contact.tolist())

                # If we can map the lipid residue via its PO4, that's the cleanest split.
                for r in res_contact_set.intersection(res_top_by_po4):
                    idx = mem_resid_to_idx.get(r, None)
                    if idx is not None:
                        col_mem_top[idx] = True

                # Fallback: classify by lipid COM relative to NP COM (still PBC-aware).
                rest = res_contact_set.difference(res_top_by_po4)
                for r in rest:
                    idx = mem_resid_to_idx.get(r, None)
                    if idx is None:
                        continue
                    try:
                        z_res = float(mem_residues[idx].atoms.center_of_mass(wrap=True)[2])
                    except Exception:
                        z_res = float(mem_residues[idx].atoms.center_of_geometry(wrap=True)[2])
                    if wrap_delta(z_res - com_np[2], Lz) > 0.0:
                        col_mem_top[idx] = True

        flags_mem2np_top = (
            col_mem_top[:, None] if flags_mem2np_top.size == 0 else np.hstack([flags_mem2np_top, col_mem_top[:, None]])
        )

    # -------------------- Exchange counting with residence τ --------------------
    times_ns = np.asarray(times_ns, float)
    dt_ns = np.median(np.diff(times_ns)) if len(times_ns) > 1 else 1.0

    # Number of frames needed to satisfy τ, based on the observed dt.
    # (uses args.tau_ns intentionally: τ should be consistent with CLI even if called with other vars)
    need_len = max(1, int(np.round(args.tau_ns / max(dt_ns, 1e-9))))

    # NP→Mem cumulative exchange per NP lipid type
    exch_np_once = np.zeros(len(np_resids), dtype=bool)
    exch_np_cum_by_type = {ty: np.zeros(flags_np2mem.shape[1], dtype=int) for ty in uniq_np_types}

    for k in range(flags_np2mem.shape[1]):
        a = max(0, k - need_len + 1)
        b = k + 1
        win = flags_np2mem[:, a:b]

        if win.shape[1] == need_len:
            sustained = np.all(win, axis=1)
            newly = np.logical_and(sustained, ~exch_np_once)
            if np.any(newly):
                exch_np_once[newly] = True

        for ty in uniq_np_types:
            mask_ty = (np_types == ty)
            exch_np_cum_by_type[ty][k] = int(np.sum(np.logical_and(exch_np_once, mask_ty)))

    # Mem→NP (TOP) cumulative exchange per membrane lipid type
    exch_mem_once = np.zeros(len(mem_resids), dtype=bool)
    exch_mem_cum_by_type = {ty: np.zeros(flags_mem2np_top.shape[1], dtype=int) for ty in uniq_mem_types}

    for k in range(flags_mem2np_top.shape[1]):
        a = max(0, k - need_len + 1)
        b = k + 1
        win = flags_mem2np_top[:, a:b]

        if win.shape[1] == need_len:
            sustained = np.all(win, axis=1)
            newly = np.logical_and(sustained, ~exch_mem_once)
            if np.any(newly):
                exch_mem_once[newly] = True

        for ty in uniq_mem_types:
            mask_ty = (mem_types == ty)
            exch_mem_cum_by_type[ty][k] = int(np.sum(np.logical_and(exch_mem_once, mask_ty)))

    # Convert exchange timelines to µs and compute rates (events/µs)
    t_ex_us = times_ns / 1000.0
    rate_np_types = {ty: rate_from_cumulative(t_ex_us, exch_np_cum_by_type[ty], args.rate_window_ns) for ty in uniq_np_types}
    rate_mem_types = {
        ty: rate_from_cumulative(t_ex_us, exch_mem_cum_by_type[ty], args.rate_window_ns) for ty in uniq_mem_types
    }

    return {
        "label": os.path.basename(traj),
        "time_us": np.array(t_us, dtype=float),
        "dist_nm": np.array(dist_nm, dtype=float),
        "deform_nm": np.array(deform_nm, dtype=float),
        "time_us_ex": t_ex_us,
        "rate_np_types": rate_np_types,        # dict: type -> array
        "rate_mem_types": rate_mem_types,      # dict: type -> array
        "cum_np_types": exch_np_cum_by_type,   # dict: type -> array
        "cum_mem_types": exch_mem_cum_by_type, # dict: type -> array
    }


# -------------------- Run all replicas --------------------
rep_objs = [
    analyze_rep(
        args.structure,
        tr,
        args.stride,
        args.tau_ns,
        args.np2mem_cutoff_nm,
        args.mem2np_cutoff_nm,
    )
    for tr in args.trajectories
]

# Standardized labels rep1..repN (based on the order given in -t)
rep_labels = [f"rep{i+1}" for i in range(len(rep_objs))]
label_map = {rep_objs[i]["label"]: rep_labels[i] for i in range(len(rep_objs))}

tmin, tmax = float(args.tmin_us), float(args.tmax_us)

# -------------------- Distance --------------------
series_dist = {}
times_list_dist = []
for ro in rep_objs:
    m = (ro["time_us"] >= tmin) & (ro["time_us"] <= tmax)
    series_dist[label_map[ro["label"]]] = (ro["time_us"][m], ro["dist_nm"][m])
    times_list_dist.append(ro["time_us"][m])

grid_dist = build_common_grid(tmin, tmax, times_list_dist)
write_wide_csv("distances_wide.csv", grid_dist, series_dist, include_mean=True)
plot_lines(
    grid_dist,
    series_dist,
    "COM(F216) → PO4 distance (nm)",
    "Distance (nm)",
    "plot_distances.png",
    smooth=args.plot_smooth,
    add_mean=True,
)

# -------------------- Deformation --------------------
series_def = {}
times_list_def = []
for ro in rep_objs:
    m = (ro["time_us"] >= tmin) & (ro["time_us"] <= tmax)
    series_def[label_map[ro["label"]]] = (ro["time_us"][m], ro["deform_nm"][m])
    times_list_def.append(ro["time_us"][m])

grid_def = build_common_grid(tmin, tmax, times_list_def)
write_wide_csv("deformation_wide.csv", grid_def, series_def, include_mean=True)
plot_lines(
    grid_def,
    series_def,
    "Membrane deformation (Δ vs frame 0)",
    "Δ thickness (nm)",
    "plot_deformation.png",
    smooth=args.plot_smooth,
    add_mean=True,
)

# -------------------- Type-resolved exchange rates (replicate-averaged) --------------------
grid_rate = build_common_grid(tmin, tmax, [ro["time_us_ex"] for ro in rep_objs])

# NP→Mem rate by NP lipid type
np_types_present = sorted(
    {ty for ro in rep_objs for ty in ro["rate_np_types"].keys()},
    key=lambda s: NP_LIPIDS.index(s) if s in NP_LIPIDS else 999,
)

series_np_types_mean = {}
for ty in NP_LIPIDS:
    if ty not in np_types_present:
        continue
    rows = []
    for ro in rep_objs:
        if ty in ro["rate_np_types"]:
            t = ro["time_us_ex"]
            y = ro["rate_np_types"][ty]
            yi = np.interp(grid_rate, t, y, left=np.nan, right=np.nan)
            rows.append(yi)
    if rows:
        M = np.vstack(rows)
        series_np_types_mean[ty] = (grid_rate, np.nanmean(M, axis=0))

write_wide_csv("exchange_rate_np2mem_types_wide.csv", grid_rate, series_np_types_mean, include_mean=False)
plot_lines(
    grid_rate,
    series_np_types_mean,
    f"NP→Mem rate by NP lipid type (events/µs) — τ ≥ {args.tau_ns:.0f} ns",
    "Rate (events/µs)",
    "plot_rate_np2mem_types.png",
    smooth=args.plot_smooth,
    add_mean=False,
)

# Mem→NP (TOP) rate by membrane lipid type
mem_types_present = sorted(
    {ty for ro in rep_objs for ty in ro["rate_mem_types"].keys()},
    key=lambda s: MEM_LIPIDS.index(s) if s in MEM_LIPIDS else 999,
)

series_mem_types_mean = {}
for ty in MEM_LIPIDS:
    if ty not in mem_types_present:
        continue
    rows = []
    for ro in rep_objs:
        if ty in ro["rate_mem_types"]:
            t = ro["time_us_ex"]
            y = ro["rate_mem_types"][ty]
            yi = np.interp(grid_rate, t, y, left=np.nan, right=np.nan)
            rows.append(yi)
    if rows:
        M = np.vstack(rows)
        series_mem_types_mean[ty] = (grid_rate, np.nanmean(M, axis=0))

write_wide_csv("exchange_rate_mem2np_top_types_wide.csv", grid_rate, series_mem_types_mean, include_mean=False)
plot_lines(
    grid_rate,
    series_mem_types_mean,
    f"Mem→NP (TOP) rate by membrane lipid type (events/µs) — τ ≥ {args.tau_ns:.0f} ns",
    "Rate (events/µs)",
    "plot_rate_mem2np_top_types.png",
    smooth=args.plot_smooth,
    add_mean=False,
)

# -------------------- Type-resolved cumulative exchange (replicate-averaged) --------------------
# NP→Mem cumulative by NP lipid type
series_np_cum_types_mean = {}
for ty in NP_LIPIDS:
    rows = []
    for ro in rep_objs:
        if ty in ro["cum_np_types"]:
            t = ro["time_us_ex"]
            y = ro["cum_np_types"][ty]
            yi = np.interp(grid_rate, t, y, left=np.nan, right=np.nan)
            rows.append(yi)
    if rows:
        M = np.vstack(rows)
        series_np_cum_types_mean[ty] = (grid_rate, np.nanmean(M, axis=0))

write_wide_csv("exchange_cumulative_np2mem_types_wide.csv", grid_rate, series_np_cum_types_mean, include_mean=False)
plot_lines(
    grid_rate,
    series_np_cum_types_mean,
    f"Cumulative NP→Mem by NP lipid type — τ ≥ {args.tau_ns:.0f} ns",
    "Cumulative events",
    "plot_cum_np2mem_types.png",
    smooth=args.plot_smooth,
    add_mean=False,
)

# Mem→NP (TOP) cumulative by membrane lipid type
series_mem_cum_types_mean = {}
for ty in MEM_LIPIDS:
    rows = []
    for ro in rep_objs:
        if ty in ro["cum_mem_types"]:
            t = ro["time_us_ex"]
            y = ro["cum_mem_types"][ty]
            yi = np.interp(grid_rate, t, y, left=np.nan, right=np.nan)
            rows.append(yi)
    if rows:
        M = np.vstack(rows)
        series_mem_cum_types_mean[ty] = (grid_rate, np.nanmean(M, axis=0))

write_wide_csv("exchange_cumulative_mem2np_top_types_wide.csv", grid_rate, series_mem_cum_types_mean, include_mean=False)
plot_lines(
    grid_rate,
    series_mem_cum_types_mean,
    f"Cumulative Mem→NP (TOP) by membrane lipid type — τ ≥ {args.tau_ns:.0f} ns",
    "Cumulative events",
    "plot_cum_mem2np_top_types.png",
    smooth=args.plot_smooth,
    add_mean=False,
)

print("Done. Outputs written:")
print("  - distances_wide.csv                          + plot_distances.png")
print("  - deformation_wide.csv                        + plot_deformation.png")
print("  - exchange_rate_np2mem_types_wide.csv         + plot_rate_np2mem_types.png")
print("  - exchange_rate_mem2np_top_types_wide.csv     + plot_rate_mem2np_top_types.png")
print("  - exchange_cumulative_np2mem_types_wide.csv   + plot_cum_np2mem_types.png")
print("  - exchange_cumulative_mem2np_top_types_wide.csv + plot_cum_mem2np_top_types.png")
