#!/usr/bin/env python3
"""
analyze_timing.py — Migration Stage Timing Analysis
Target: ARM64, Linux 6.1.4, 4KB pages, real NUMA (2 nodes)

Reads CSV files written by the kernel's mig_timing debugfs interface
and produces:
  1. Per-file stage breakdown table (mean/median/P95/P99/CV/% of total)
  2. 4KB vs 2MB THP comparison with scaling classification
     (or analytical THP projection if THP unavailable on system)
  3. Sharing degree scaling analysis (rmap walk cost vs mapcount)
  4. PNG plots if matplotlib is available

Usage:
    python3 analyze_timing.py timing_*.csv
    python3 analyze_timing.py timing_4kb_512.csv timing_2mb_32.csv

ARM64 interpretation notes (embedded in output):
  - Unmap cost: driven by TLBI VAE1IS + DSB ISH barrier, not x86 IPIs.
  - Copy cost: __pi_memcpy (NEON).
  - Lock CV > 1.0: normal — ARM64 LDAXR/STXR exclusive on folio->flags
    occasionally sees cache line bouncing from concurrent page reclaim.
"""

import sys
import os
import re
import numpy as np
from collections import defaultdict

# Optional matplotlib — graceful fallback
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("NOTE: matplotlib not found — text-only output (pip install matplotlib)")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path):
    """
    Load the CSV produced by mig_timing debugfs.
    Returns dict of {column_name: np.ndarray} or None on failure.
    """
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return None

    with open(path) as f:
        lines = f.readlines()

    if len(lines) < 2:
        print(f"  SKIP: {path} is empty (no migration records)")
        return None

    header = lines[0].strip().split(",")
    rows = []
    for ln in lines[1:]:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split(",")
        if len(parts) == len(header):
            rows.append(parts)

    if not rows:
        print(f"  SKIP: {path} has header but no data rows")
        return None

    data = {}
    for ci, col in enumerate(header):
        raw = [row[ci] for row in rows]
        try:
            data[col] = np.array([int(v) for v in raw])
        except ValueError:
            try:
                data[col] = np.array([float(v) for v in raw])
            except ValueError:
                data[col] = np.array(raw)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Page-order filtering helpers
# ─────────────────────────────────────────────────────────────────────────────

def filter_by_order(data, order):
    """Return a copy of data keeping only rows where page_order == order."""
    if "page_order" not in data:
        return data   # old CSV without page_order — pass through unchanged
    mask = data["page_order"] == order
    if not mask.any():
        return None
    return {col: arr[mask] for col, arr in data.items()}


def report_thp_composition(data, label):
    """Print order=9 (whole THP) vs order=0 (split 4KB) breakdown."""
    if "page_order" not in data:
        return
    total   = len(data["page_order"])
    n_thp   = int(np.sum(data["page_order"] == 9))
    n_4kb   = int(np.sum(data["page_order"] == 0))
    n_other = total - n_thp - n_4kb
    print(f"  THP composition ({label}):")
    print(f"    order=9 whole THP  : {n_thp:6d}  ({100*n_thp/total:.1f}%)")
    print(f"    order=0 split 4KB  : {n_4kb:6d}  ({100*n_4kb/total:.1f}%)")
    if n_other:
        print(f"    other orders       : {n_other:6d}")
    if n_thp == 0:
        print("  WARNING: no whole-THP records — all migrations were split.")
        print("           Use timing_thp_real_*.csv for real THP analysis.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Stage breakdown analysis
# ─────────────────────────────────────────────────────────────────────────────

STAGES      = ["lock_ns", "unmap_ns", "copy_ns", "remap_ns", "unlock_ns"]
STAGE_NAMES = ["Lock",    "Unmap",    "Copy",    "Remap",    "Unlock"]


def pct_str(val, total):
    return f"{100.0 * val / total:5.1f}%" if total > 0 else "  n/a "


def analyze_stage_breakdown(data, label):
    """
    Core analysis: decompose total migration time into 5 stages.
    Prints a table and returns per-stage means for cross-file comparison.

    Columns:
      Mean, Median — central tendency (µs)
      StdDev       — absolute spread
      P5, P95, P99 — tail behaviour
      CV           — coefficient of variation = stddev/mean.
                     CV < 0.3 : very consistent (bounded by HW)
                     CV 0.3-0.7: moderate variance (OS scheduling)
                     CV > 1.0  : high variance (contention, outliers)
      %Total       — fraction of mean total time
    """
    n = len(data["total_ns"])
    total = data["total_ns"].astype(float)
    total_mean = float(np.mean(total))

    print()
    print("=" * 84)
    print(f"  STAGE BREAKDOWN: {label}")
    print(f"  n={n}  |  ARM64 / 4KB pages / real NUMA (2 nodes)")
    print("=" * 84)

    fmt_hdr = f"  {'Stage':<10}  {'Mean':>8}  {'Median':>8}  {'StdDev':>8}  " \
              f"{'P5':>8}  {'P95':>8}  {'P99':>8}  {'CV':>6}  {'%Total':>7}"
    print(fmt_hdr)
    print("  " + "─" * 82)

    stage_means = {}

    for col, name in zip(STAGES, STAGE_NAMES):
        if col not in data:
            print(f"  {name:<10}  (column missing in CSV)")
            continue

        v = data[col].astype(float)
        mean   = float(np.mean(v))
        median = float(np.median(v))
        std    = float(np.std(v))
        p5     = float(np.percentile(v, 5))
        p95    = float(np.percentile(v, 95))
        p99    = float(np.percentile(v, 99))
        cv     = std / mean if mean > 0 else 0.0
        pct    = 100.0 * mean / total_mean if total_mean > 0 else 0.0

        stage_means[col] = mean

        print(f"  {name:<10}  {mean/1e3:>8.2f}  {median/1e3:>8.2f}  "
              f"{std/1e3:>8.2f}  {p5/1e3:>8.2f}  {p95/1e3:>8.2f}  "
              f"{p99/1e3:>8.2f}  {cv:>6.2f}  {pct:>6.1f}%")

    # Overhead row
    accounted = sum(data[s].astype(float) for s in STAGES if s in data)
    overhead  = np.maximum(total - accounted, 0.0)
    oh_mean   = float(np.mean(overhead))
    oh_pct    = 100.0 * oh_mean / total_mean if total_mean > 0 else 0.0

    print(f"  {'Overhead':<10}  {oh_mean/1e3:>8.2f}  {np.median(overhead)/1e3:>8.2f}  "
          f"{np.std(overhead)/1e3:>8.2f}  "
          f"{np.percentile(overhead,5)/1e3:>8.2f}  "
          f"{np.percentile(overhead,95)/1e3:>8.2f}  "
          f"{np.percentile(overhead,99)/1e3:>8.2f}  "
          f"{'':>6}  {oh_pct:>6.1f}%")

    print("  " + "─" * 82)
    print(f"  {'TOTAL':<10}  {total_mean/1e3:>8.2f}  "
          f"{np.median(total)/1e3:>8.2f}  "
          f"{np.std(total)/1e3:>8.2f}  "
          f"{np.percentile(total,5)/1e3:>8.2f}  "
          f"{np.percentile(total,95)/1e3:>8.2f}  "
          f"{np.percentile(total,99)/1e3:>8.2f}")

    # ── Key Insights ──
    print()
    print("  KEY INSIGHTS (ARM64-specific):")

    if stage_means:
        dom_col  = max(stage_means, key=stage_means.get)
        dom_name = STAGE_NAMES[STAGES.index(dom_col)]
        dom_pct  = 100.0 * stage_means[dom_col] / total_mean
        print(f"  • Dominant stage: {dom_name} ({dom_pct:.1f}% of total)")

    # Copy vs Unmap ratio → tells us which optimisation is highest priority
    if "copy_ns" in stage_means and "unmap_ns" in stage_means:
        c = stage_means["copy_ns"]
        u = stage_means["unmap_ns"]
        if c > u:
            print(f"  • Copy/Unmap = {c/u:.1f}×  → DMA offloading is highest priority")
        else:
            print(f"  • Unmap/Copy = {u/c:.1f}×  → TLBI batching is highest priority")

    # High-variance stages
    for col, name in zip(STAGES, STAGE_NAMES):
        if col not in data:
            continue
        v  = data[col].astype(float)
        cv = np.std(v) / np.mean(v) if np.mean(v) > 0 else 0.0
        if cv > 1.0:
            print(f"  • {name}: CV={cv:.2f} (high) — likely cache-line bouncing "
                  f"on ARM64 exclusive pair (LDAXR/STXR) or scheduler jitter")
        elif cv > 0.5:
            print(f"  • {name}: CV={cv:.2f} (moderate) — TLBI broadcast timing variation")

    # Lock contention
    if "lock_contended" in data:
        nc = int(np.sum(data["lock_contended"].astype(int)))
        rate = 100.0 * nc / n
        if nc > 0:
            mask = data["lock_contended"].astype(int) == 1
            slow = data["lock_ns"].astype(float)[mask]
            print(f"  • Lock contention: {rate:.1f}% ({nc}/{n}) — "
                  f"slow path mean={np.mean(slow)/1e3:.1f}µs  "
                  f"max={np.max(slow)/1e3:.1f}µs")
        else:
            print(f"  • Lock contention: none — all uncontended "
                  f"(fast LDAXR/STXR path)")

    # Unmap ARM64 commentary
    if "unmap_ns" in data:
        u_mean = np.mean(data["unmap_ns"].astype(float))
        u_p95  = np.percentile(data["unmap_ns"].astype(float), 95)
        print(f"  • Unmap mean={u_mean/1e3:.1f}µs, P95={u_p95/1e3:.1f}µs")
        print(f"    ARM64: cost = rmap walk + TLBI VAE1IS + DSB ISH per PTE")

    return stage_means


# ─────────────────────────────────────────────────────────────────────────────
# 4KB vs 2MB comparison
# ─────────────────────────────────────────────────────────────────────────────

def analyze_size_comparison(data_4kb, data_2mb):
    """
    Compare stage means between 4KB and 2MB THP migrations.
    Classifies each stage as O(1) (fixed cost) or O(size) (copy-bound).
    """
    print()
    print("=" * 70)
    print("  PAGE SIZE COMPARISON: 4KB vs 2MB THP")
    print("  ARM64 / 4KB base pages / real NUMA")
    print("=" * 70)

    print(f"\n  {'Stage':<10}  {'4KB (µs)':>10}  {'2MB (µs)':>10}  "
          f"{'Ratio':>8}  {'Class':>12}  Notes")
    print("  " + "─" * 68)

    total_4kb = float(np.mean(data_4kb["total_ns"].astype(float)))
    total_2mb = float(np.mean(data_2mb["total_ns"].astype(float)))

    for col, name in zip(STAGES, STAGE_NAMES):
        if col not in data_4kb or col not in data_2mb:
            continue
        m4 = float(np.mean(data_4kb[col].astype(float)))
        m2 = float(np.mean(data_2mb[col].astype(float)))
        ratio = m2 / m4 if m4 > 0 else float("inf")

        if ratio > 100:
            cls   = "O(size)"
            notes = "scales with data — DMA offload target"
        elif ratio > 5:
            cls   = "partial"
            notes = "partial scaling"
        else:
            cls   = "O(1)"
            notes = "fixed cost — amortised by batching"

        print(f"  {name:<10}  {m4/1e3:>10.2f}  {m2/1e3:>10.2f}  "
              f"{ratio:>8.1f}  {cls:>12}  {notes}")

    print("  " + "─" * 68)
    total_ratio = total_2mb / total_4kb if total_4kb > 0 else 0
    print(f"  {'TOTAL':<10}  {total_4kb/1e3:>10.2f}  {total_2mb/1e3:>10.2f}  "
          f"{total_ratio:>8.1f}")

    print()
    print("  ANALYSIS:")
    print(f"  • Total ratio = {total_ratio:.1f}×  (theoretical max = 512× "
          f"for pure copy)")
    print(f"  • Ratio < 512× confirms that O(1) stages (lock/unmap/remap)")
    print(f"    are a non-trivial fraction of 4KB total cost")

    if "copy_ns" in data_4kb and "copy_ns" in data_2mb:
        c4 = float(np.mean(data_4kb["copy_ns"].astype(float)))
        c2 = float(np.mean(data_2mb["copy_ns"].astype(float)))
        cr = c2 / c4 if c4 > 0 else 0

        bw_4kb_gbs = (4096 / (c4 / 1e9)) / 1e9 if c4 > 0 else 0
        bw_2mb_gbs = (2 * 1024 * 1024 / (c2 / 1e9)) / 1e9 if c2 > 0 else 0

        print(f"  • Copy ratio = {cr:.1f}× (theoretical 512× for linear scaling)")
        print(f"    4KB copy BW: {bw_4kb_gbs:.1f} GB/s  |  "
              f"2MB copy BW: {bw_2mb_gbs:.1f} GB/s")
        print(f"    ARM64 note: NEON __pi_memcpy benefits more from hardware")
        print(f"    prefetcher activation on the longer 2MB copy path")
        print()
        print(f"  OPTIMISATION PRIORITY:")
        print(f"  • For 4KB pages: O(1) stages dominate → batch TLB shootdowns")
        print(f"  • For 2MB THPs:  copy dominates      → DMA offloading")


# ─────────────────────────────────────────────────────────────────────────────
# Analytical THP projection (when THP unavailable on test system)
# ─────────────────────────────────────────────────────────────────────────────

def analytical_thp_projection(data_4kb):
    """
    Project 2MB THP migration costs from 4KB baseline measurements.
    Used when CONFIG_TRANSPARENT_HUGEPAGE_ALWAYS is not set and THP
    cannot be allocated on the test system.

    Assumptions:
      - Copy scales linearly with data size (512x for 2MB vs 4KB)
      - Lock, Unmap, Remap, Unlock are O(1) — fixed per-folio cost
        regardless of folio size (one rmap entry per THP, not per sub-page)
      - DMA submission overhead estimated at ~200ns (descriptor write)
    """
    print()
    print("=" * 70)
    print("  ANALYTICAL THP PROJECTION (2MB)")
    print("  THP unavailable on test system (CONFIG_TRANSPARENT_HUGEPAGE_ALWAYS")
    print("  not set). Projecting from 4KB measured baseline.")
    print("=" * 70)

    pages_per_thp = 512  # 2MB / 4KB

    copy_4kb   = float(np.mean(data_4kb["copy_ns"].astype(float)))
    unmap_4kb  = float(np.mean(data_4kb["unmap_ns"].astype(float)))
    remap_4kb  = float(np.mean(data_4kb["remap_ns"].astype(float)))
    lock_4kb   = float(np.mean(data_4kb["lock_ns"].astype(float)))
    unlock_4kb = float(np.mean(data_4kb["unlock_ns"].astype(float)))

    # Copy scales linearly — 512x more data to transfer
    copy_2mb = copy_4kb * pages_per_thp

    # All other stages are O(1) — one rmap entry per THP folio,
    # one lock/unlock per folio regardless of size
    unmap_2mb  = unmap_4kb
    remap_2mb  = remap_4kb
    lock_2mb   = lock_4kb
    unlock_2mb = unlock_4kb

    total_2mb = copy_2mb + unmap_2mb + remap_2mb + lock_2mb + unlock_2mb

    # DMA scenario: replace CPU copy with descriptor submission (~200ns)
    dma_submission_ns = 200.0
    total_2mb_dma = dma_submission_ns + unmap_2mb + remap_2mb + lock_2mb + unlock_2mb

    stages = [
        ("Lock",   lock_4kb,   lock_2mb),
        ("Unmap",  unmap_4kb,  unmap_2mb),
        ("Copy",   copy_4kb,   copy_2mb),
        ("Remap",  remap_4kb,  remap_2mb),
        ("Unlock", unlock_4kb, unlock_2mb),
    ]

    print(f"\n  {'Stage':<10}  {'4KB base (µs)':>14}  {'2MB proj (µs)':>14}  "
          f"{'Ratio':>7}  {'% of 2MB total':>15}")
    print("  " + "─" * 66)

    for name, val_4, val_2mb in stages:
        ratio = val_2mb / val_4 if val_4 > 0 else 0
        pct   = 100.0 * val_2mb / total_2mb
        print(f"  {name:<10}  {val_4/1e3:>14.3f}  {val_2mb/1e3:>14.1f}  "
              f"{ratio:>7.1f}  {pct:>14.1f}%")

    print("  " + "─" * 66)
    print(f"  {'TOTAL':<10}  {'':>14}  {total_2mb/1e3:>14.1f}")

    print()
    print("  INTERPRETATION:")
    print(f"  • Projected copy = {copy_2mb/1e3:.1f} µs "
          f"({100*copy_2mb/total_2mb:.1f}% of total)")
    bw_gbs = (4096 / (copy_4kb / 1e9)) / 1e9 if copy_4kb > 0 else 0
    print(f"  • Copy bandwidth = {bw_gbs:.1f} GB/s (from 4KB measured)")
    print()
    print(f"  DMA OFFLOADING IMPACT (projected):")
    print(f"  • CPU copy time:        {copy_2mb/1e3:>8.1f} µs  (eliminated)")
    print(f"  • DMA submission:       {dma_submission_ns/1e3:>8.3f} µs  (replaces copy)")
    print(f"  • Total with DMA:       {total_2mb_dma/1e3:>8.2f} µs")
    print(f"  • Total without DMA:    {total_2mb/1e3:>8.1f} µs")
    print(f"  • Projected speedup:    {total_2mb/total_2mb_dma:>8.1f}x")
    print()
    print(f"  NOTE: O(1) stages (lock/unmap/remap) are unchanged by DMA.")
    print(f"  At {(unmap_2mb+remap_2mb)/1e3:.2f} µs combined, they become the new")
    print(f"  bottleneck after DMA — validating the need for TLB batching too.")

    if HAS_MPL:
        _plot_thp_projection(stages, total_2mb, total_2mb_dma, dma_submission_ns)


def _plot_thp_projection(stages, total_2mb, total_2mb_dma, dma_ns):
    """Bar chart comparing CPU copy vs DMA offload for projected 2MB THP."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: projected stage breakdown pie
    names  = [s[0] for s in stages]
    values = [s[2] / 1e3 for s in stages]
    colors = COLORS[:len(names)]
    axes[0].pie(values, labels=names, colors=colors,
                autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Projected 2MB THP Stage Breakdown\n(analytical from 4KB baseline)")

    # Right: CPU copy vs DMA comparison
    scenarios  = ["CPU copy\n(current)", "DMA offload\n(projected)"]
    totals     = [total_2mb / 1e3, total_2mb_dma / 1e3]
    bar_colors = ["#e74c3c", "#2ecc71"]
    bars = axes[1].bar(scenarios, totals, color=bar_colors, width=0.4, alpha=0.8)
    axes[1].set_ylabel("Migration time (µs)")
    axes[1].set_title("DMA Offloading Impact\n(projected 2MB THP)")
    for bar, val in zip(bars, totals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f} µs", ha="center", va="bottom", fontweight="bold")
    axes[1].set_ylim(0, max(totals) * 1.2)

    plt.tight_layout()
    out = "thp_projection.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Plot → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Sharing degree scaling
# ─────────────────────────────────────────────────────────────────────────────

def analyze_sharing_scaling(files_by_degree):
    """
    Analyse how unmap (TLBI per-mapping) and remap costs scale
    with sharing degree.

    Expected pattern on ARM64:
      unmap_ns ≈ α × degree + β   (linear: one TLBI per PTE per page)
      remap_ns ≈ γ × degree       (linear: one PTE write per mapping)
      copy_ns  ≈ constant         (data size unchanged)

    α = per-PTE TLBI+rmap walk cost
    β = fixed entry cost (rmap lock, anon_vma tree traversal root)
    """
    if not files_by_degree:
        return

    print()
    print("=" * 80)
    print("  SHARING DEGREE SCALING (rmap walk + TLBI broadcast)")
    print("  ARM64: one TLBI VAE1IS per PTE per sharing process")
    print("=" * 80)

    print(f"\n  {'Degree':>8}  {'Unmap (µs)':>12}  {'Remap (µs)':>12}  "
          f"{'Copy (µs)':>10}  {'Total (µs)':>12}  {'rmap %':>8}")
    print("  " + "─" * 66)

    rows = []
    for deg, path in sorted(files_by_degree.items()):
        d = load_csv(path)
        if d is None:
            continue
        um  = float(np.mean(d["unmap_ns"].astype(float)))
        rm  = float(np.mean(d["remap_ns"].astype(float)))
        cp  = float(np.mean(d["copy_ns"].astype(float)))
        tot = float(np.mean(d["total_ns"].astype(float)))
        rmap_pct = 100.0 * (um + rm) / tot if tot > 0 else 0.0
        rows.append((deg, um, rm, cp, tot, rmap_pct))
        print(f"  {deg:>8}  {um/1e3:>12.2f}  {rm/1e3:>12.2f}  "
              f"{cp/1e3:>10.2f}  {tot/1e3:>12.2f}  {rmap_pct:>7.1f}%")

    if len(rows) >= 2:
        degrees = np.array([r[0] for r in rows], dtype=float)
        unmaps  = np.array([r[1] for r in rows], dtype=float)
        remaps  = np.array([r[2] for r in rows], dtype=float)

        def linreg(x, y):
            xm, ym = np.mean(x), np.mean(y)
            slope = np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)
            intercept = ym - slope * xm
            return slope, intercept

        u_slope, u_int = linreg(degrees, unmaps)
        r_slope, r_int = linreg(degrees, remaps)

        print()
        print("  LINEAR REGRESSION (cost = α×degree + β):")
        print(f"    Unmap:  α={u_slope/1e3:6.2f} µs/process,  "
              f"β={u_int/1e3:6.2f} µs  (per-TLBI + rmap walk cost)")
        print(f"    Remap:  α={r_slope/1e3:6.2f} µs/process,  "
              f"β={r_int/1e3:6.2f} µs  (per-PTE write cost)")
        print()
        print("  ARM64 TLBI INTERPRETATION:")
        print(f"    Per-process TLBI overhead ≈ {u_slope/1e3:.2f} µs")
        print(f"    This is TLBI VAE1IS + DSB ISH per PTE mapping.")
        print(f"    Batching N shootdowns into one TLBI range instruction")
        print(f"    would reduce Stage 2 cost toward the β={u_int/1e3:.2f} µs baseline.")

        if HAS_MPL:
            plot_sharing_scaling(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

COLORS = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6"]


def plot_breakdown(data, label, prefix):
    if not HAS_MPL:
        return

    stages_us = [np.mean(data[s].astype(float)) / 1e3 for s in STAGES
                 if s in data]
    total_us  = np.mean(data["total_ns"].astype(float)) / 1e3
    overhead  = max(total_us - sum(stages_us), 0)

    values = stages_us + [overhead]
    names  = STAGE_NAMES[:len(stages_us)] + ["Overhead"]
    colors = COLORS[:len(stages_us)] + ["#95a5a6"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].pie(values, labels=names, colors=colors,
                autopct="%1.1f%%", startangle=90)
    axes[0].set_title(f"Stage Breakdown\n{label}")

    total_arr = data["total_ns"].astype(float) / 1e3
    axes[1].hist(total_arr, bins=50, color="#3498db", edgecolor="#2c3e50", alpha=0.8)
    axes[1].axvline(np.mean(total_arr), color="red", linestyle="--",
                    label=f"Mean: {np.mean(total_arr):.1f} µs")
    axes[1].axvline(np.percentile(total_arr, 99), color="orange", linestyle="--",
                    label=f"P99: {np.percentile(total_arr, 99):.1f} µs")
    axes[1].set_xlabel("Migration time (µs)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Total Time Distribution\n{label}")
    axes[1].legend()

    plt.tight_layout()
    out = f"{prefix}_pie_hist.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Plot → {out}")


def plot_boxplot(data, label, prefix):
    if not HAS_MPL:
        return

    stage_data  = [data[s].astype(float) / 1e3 for s in STAGES if s in data]
    valid_names = [n for s, n in zip(STAGES, STAGE_NAMES) if s in data]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(stage_data, labels=valid_names, patch_artist=True,
                    showfliers=True,
                    flierprops=dict(markersize=2, alpha=0.4))
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Time (µs) — log scale")
    ax.set_yscale("log")
    ax.set_title(f"Per-Stage Time Distributions (log scale)\n{label}")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = f"{prefix}_boxplot.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Plot → {out}")


def plot_sharing_scaling(rows):
    if not HAS_MPL or not rows:
        return

    degrees = [r[0] for r in rows]
    unmaps  = [r[1] / 1e3 for r in rows]
    remaps  = [r[2] / 1e3 for r in rows]
    totals  = [r[4] / 1e3 for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(degrees, unmaps, "o-", color="#e74c3c", label="Unmap (TLBI)")
    ax.plot(degrees, remaps, "s-", color="#f39c12", label="Remap (PTE write)")
    ax.plot(degrees, totals, "^-", color="#3498db", label="Total")
    ax.set_xlabel("Sharing degree (number of processes mapping each page)")
    ax.set_ylabel("Mean time (µs)")
    ax.set_title("Unmap / Remap scaling vs sharing degree\nARM64 TLBI broadcast cost")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = "sharing_scaling.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Plot → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def categorise_files(paths):
    """
    Sort files into 4kb, 2mb, and shared categories by filename convention.
    Returns (list_4kb, list_2mb, dict_shared {degree: path}).
    """
    f4, f2 = [], []
    fsh = {}

    for p in paths:
        base = os.path.basename(p).lower()
        if "4kb" in base:
            f4.append(p)
        elif "thp_real" in base:
            f2.append(p)   # pre-filtered real THP records (order=9 only)
        elif "2mb" in base:
            f2.append(p)
        elif "shared" in base:
            m = re.search(r"deg(\d+)", base)
            if m:
                fsh[int(m.group(1))] = p
        else:
            print(f"NOTE: {p} — filename doesn't match 4kb/2mb/shared pattern, "
                  f"analysing as standalone")
            f4.append(p)

    return f4, f2, fsh


# ─────────────────────────────────────────────────────────────────────────────
# try_to_migrate() sub-timing analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_try_migrate(data, label):
    """
    Breaks Stage 2 (unmap_ns) into two components:
      try_migrate_ns  — time inside try_to_migrate() itself
                        (rmap walk + per-PTE TLBI VAE1IS + DSB ISH)
      overhead_ns     — surrounding kernel work in __unmap_and_move()
                        (folio_mapped check, VM_BUG_ON, anon_vma lookup)

    This decomposition shows how much of Stage 2 scales with mapcount
    vs how much is fixed overhead — directly quantifies the savings
    achievable from TLB batching.
    """
    if "try_migrate_ns" not in data or "unmap_ns" not in data:
        return

    # Only use records where the page was actually mapped (unmap was real)
    mapped = data.get("page_was_mapped")
    if mapped is not None:
        mask = mapped.astype(int) == 1
    else:
        mask = data["unmap_ns"].astype(float) > 0

    tm  = data["try_migrate_ns"].astype(float)[mask]
    un  = data["unmap_ns"].astype(float)[mask]

    if len(tm) == 0:
        return

    overhead = np.maximum(un - tm, 0.0)

    tm_mean  = float(np.mean(tm))
    tm_p95   = float(np.percentile(tm, 95))
    oh_mean  = float(np.mean(overhead))
    un_mean  = float(np.mean(un))

    frac_tm  = 100.0 * tm_mean / un_mean if un_mean > 0 else 0.0
    frac_oh  = 100.0 * oh_mean / un_mean if un_mean > 0 else 0.0

    print()
    print(f"  TRY_TO_MIGRATE BREAKDOWN — {label}")
    print(f"  {'Component':<35}  {'Mean':>8}  {'P95':>8}  {'% of unmap':>10}")
    print("  " + "─" * 65)
    print(f"  {'try_to_migrate() [rmap+TLBI+DSB]':<35}  "
          f"{tm_mean/1e3:>8.2f}  {tm_p95/1e3:>8.2f}  {frac_tm:>9.1f}%")
    print(f"  {'surrounding overhead (fixed)':<35}  "
          f"{oh_mean/1e3:>8.2f}  {'':>8}  {frac_oh:>9.1f}%")
    print(f"  {'Stage 2 total (unmap_ns)':<35}  "
          f"{un_mean/1e3:>8.2f}  {'':>8}  {'100.0':>9}%")
    print()
    print(f"  TLB BATCHING IMPLICATION:")
    print(f"  • try_to_migrate() is {frac_tm:.1f}% of Stage 2")
    print(f"  • This is the portion that scales with mapcount (O(N) DSB ISH stalls)")
    print(f"  • Fixed overhead ({oh_mean/1e3:.2f} µs) is irreducible regardless of strategy")
    print(f"  • Batching reduces DSB ISH from N stalls to 1 stall per migration")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage: python3 analyze_timing.py timing_*.csv")
        sys.exit(1)

    input_files = [f for f in sys.argv[1:] if os.path.exists(f)]
    missing     = [f for f in sys.argv[1:] if not os.path.exists(f)]

    if missing:
        print(f"WARNING: files not found: {', '.join(missing)}")

    if not input_files:
        print("ERROR: no valid input files")
        sys.exit(1)

    files_4kb, files_2mb, files_shared = categorise_files(input_files)

    data_4kb_first = None
    data_2mb_first = None

    # ── 4KB analyses ──
    for f in files_4kb:
        d = load_csv(f)
        if d is None:
            continue
        label = f"4KB ({os.path.basename(f)})"
        analyze_stage_breakdown(d, label)
        analyze_try_migrate(d, label)
        prefix = f.replace(".csv", "")
        plot_breakdown(d, label, prefix)
        plot_boxplot(d, label, prefix)
        if data_4kb_first is None:
            data_4kb_first = d

    # ── 2MB analyses ──
    for f in files_2mb:
        d = load_csv(f)
        if d is None:
            continue

        base = os.path.basename(f)
        is_prefiltered = "thp_real" in base.lower()

        if is_prefiltered:
            # Already filtered to order=9 by mig_bench timing_read_filtered()
            d_thp = d
        else:
            # Mixed file — show composition then extract order=9 rows only
            report_thp_composition(d, base)
            d_thp = filter_by_order(d, 9)

        if d_thp is None or len(d_thp.get("copy_ns", [])) == 0:
            print(f"  SKIP THP analysis for {base}: no order=9 records.")
            print(f"  All migrations were split — use timing_thp_real_*.csv")
            continue

        n_thp = len(d_thp["copy_ns"])
        label = f"2MB THP ({base}, n={n_thp} whole-THP)"
        analyze_stage_breakdown(d_thp, label)
        analyze_try_migrate(d_thp, label)
        prefix = f.replace(".csv", "_thp")
        plot_breakdown(d_thp, label, prefix)
        plot_boxplot(d_thp, label, prefix)
        if data_2mb_first is None:
            data_2mb_first = d_thp

    # ── Cross-comparisons ──
    if data_4kb_first is not None and data_2mb_first is not None:
        analyze_size_comparison(data_4kb_first, data_2mb_first)
    elif data_4kb_first is not None and data_2mb_first is None:
        print()
        print("NOTE: No 2MB THP data found — using analytical projection from "
              "4KB baseline.")
        print("      (System has CONFIG_TRANSPARENT_HUGEPAGE_ALWAYS not set)")
        analytical_thp_projection(data_4kb_first)

    # ── Sharing degree scaling ──
    analyze_sharing_scaling(files_shared)

    print()
    print("=" * 84)
    print("  Analysis complete.")
    if HAS_MPL:
        print("  PNG plots written alongside input CSVs.")
    print("=" * 84)
    print()


if __name__ == "__main__":
    main()