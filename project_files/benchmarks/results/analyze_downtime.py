#!/usr/bin/env python3
"""
analyze_downtime.py — Benchmark 3: Migration Downtime Analysis
ARM64 / Linux 6.1.4

Usage:
    python3 analyze_downtime.py downtime_samenode.csv [downtime_crossnode.csv ...]

Optionally also pass kernel-side CSVs:
    python3 analyze_downtime.py downtime_samenode.csv kernel_downtime_samenode.csv

The script handles two CSV formats automatically:
    Userspace format: sample_idx, t_before, t_after, ticks, latency_ns, is_stall
    Kernel format:    fault_addr, pfn, page_order, cpu, pid, was_already_done,
                      wait_start_ns, total_start_ns, total_end_ns, wait_ns, total_ns
"""

import sys
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("Note: matplotlib not available — skipping plots")


# ── Constants ──────────────────────────────────────────────────────────────────

STALL_THRESH_NS   = 5_000.0     # 5 µs — below this is a normal/elevated access
ARM64_FAULT_OVERHEAD_NS = 2_500.0  # ~2.5 µs estimated ARM64 fault path overhead


# ── CSV loading ────────────────────────────────────────────────────────────────

def load_csv(filename):
    """Load a CSV file, auto-detecting format. Returns (header_list, data_dict)."""
    with open(filename) as f:
        header = f.readline().strip().split(',')
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) == len(header):
                try:
                    rows.append([float(x) for x in parts])
                except ValueError:
                    continue

    data = {col: np.array([row[i] for row in rows])
            for i, col in enumerate(header)}
    return header, data


def detect_format(header):
    """Return 'userspace' or 'kernel' based on CSV columns."""
    if 'latency_ns' in header:
        return 'userspace'
    if 'total_ns' in header:
        return 'kernel'
    return 'unknown'


# ── Userspace CSV analysis ─────────────────────────────────────────────────────

def analyze_userspace(filename, label):
    header, data = load_csv(filename)
    fmt = detect_format(header)
    if fmt != 'userspace':
        print(f"  WARNING: {filename} does not look like a userspace CSV (got {fmt})")
        return None

    latencies = data['latency_ns']
    is_stall  = data['is_stall']
    n = len(latencies)

    normal = latencies[is_stall == 0]
    stalls = latencies[is_stall == 1]

    print(f"\n{'='*68}")
    print(f"  USERSPACE DOWNTIME: {label}")
    print(f"  File   : {filename}")
    print(f"  Samples: {n:,}")
    print(f"{'='*68}")

    print(f"\n  Normal accesses ({len(normal):,}):")
    if len(normal) > 0:
        print(f"    Mean   : {np.mean(normal):.1f} ns")
        print(f"    Median : {np.median(normal):.1f} ns")
        print(f"    P95    : {np.percentile(normal, 95):.1f} ns")
        print(f"    P99    : {np.percentile(normal, 99):.1f} ns")
        print(f"    StdDev : {np.std(normal):.1f} ns")

    if len(stalls) == 0:
        print(f"\n  No stalls detected (threshold {STALL_THRESH_NS/1000:.0f} µs).")
        print("  The access thread did not hit any migration PTEs.")
        print("  Try: run multiple times, or check CPU pinning in mig_bench.c.")
        return None

    max_stall   = float(np.max(stalls))
    mean_stall  = float(np.mean(stalls))
    max_stall_idx = int(np.where(is_stall == 1)[0][np.argmax(stalls)])

    print(f"\n  Stall events ({len(stalls)}):")
    print(f"    Max    : {max_stall:.1f} ns  =  {max_stall/1000:.2f} µs  "
          f"← DOWNTIME at sample {max_stall_idx}")
    print(f"    Mean   : {mean_stall:.1f} ns  =  {mean_stall/1000:.2f} µs")
    print(f"    Min    : {np.min(stalls):.1f} ns  =  {np.min(stalls)/1000:.2f} µs")

    wait_ns = max_stall - ARM64_FAULT_OVERHEAD_NS
    print(f"\n  Downtime decomposition (max stall):")
    print(f"    Total downtime              : {max_stall/1000:.2f} µs")
    print(f"    Fixed fault overhead (ARM64): ~{ARM64_FAULT_OVERHEAD_NS/1000:.1f} µs")
    print(f"    Migration wait (copy+remap) : ~{wait_ns/1000:.2f} µs")

    print(f"\n  DMA offload implication:")
    print(f"    DMA frees the CPU during the ~{wait_ns/1000:.2f} µs copy/remap,")
    print(f"    but the faulting thread still stalls for {max_stall/1000:.2f} µs.")
    print(f"    Downtime is unchanged — benefit is off the application's critical path.")

    if len(stalls) <= 10:
        print(f"\n  Individual stall events:")
        print(f"  {'Sample':>8}  {'ns':>12}  {'µs':>10}")
        print(f"  {'------':>8}  {'---':>12}  {'---':>10}")
        stall_indices = np.where(is_stall == 1)[0]
        for idx in stall_indices:
            lat = latencies[idx]
            marker = " ← MAX" if idx == max_stall_idx else ""
            print(f"  {idx:>8}  {lat:>12.0f}  {lat/1000:>10.2f}{marker}")

    if HAS_PLOT:
        _plot_userspace(filename, label, latencies, is_stall, max_stall_idx)

    return {
        'label':       label,
        'n_normal':    len(normal),
        'n_stall':     len(stalls),
        'normal_mean': float(np.mean(normal)) if len(normal) else 0.0,
        'max_stall':   max_stall,
        'mean_stall':  mean_stall,
    }


def _plot_userspace(filename, label, latencies, is_stall, max_idx):
    n = len(latencies)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

    colors = ['steelblue' if s == 0 else 'crimson' for s in is_stall]
    ax1.scatter(np.arange(n), latencies / 1000, c=colors, s=1, alpha=0.4)
    if np.any(is_stall == 1):
        si = np.where(is_stall == 1)[0]
        ax1.scatter(si, latencies[si] / 1000, c='crimson', s=40,
                    zorder=5, label=f'Stalls (n={len(si)})')
        ax1.legend(fontsize=9)
    ax1.set_yscale('log')
    ax1.set_xlabel('Sample index')
    ax1.set_ylabel('Access latency (µs, log)')
    ax1.set_title(f'Access Latency Time Series — {label}')

    # Zoom around max stall
    w = 300
    ws = max(0, max_idx - w)
    we = min(n - 1, max_idx + w)
    lx = np.arange(ws, we + 1)
    ll = latencies[ws:we+1] / 1000
    lc = ['steelblue' if s == 0 else 'crimson' for s in is_stall[ws:we+1]]
    ax2.scatter(lx, ll, c=lc, s=5)
    ax2.axvline(max_idx, color='crimson', linestyle='--', linewidth=1.2,
                label=f'Max stall: {latencies[max_idx]/1000:.2f} µs')
    ax2.set_xlabel('Sample index')
    ax2.set_ylabel('Access latency (µs)')
    ax2.set_title(f'Zoom: ±{w} samples around max stall')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = filename.replace('.csv', '_plot.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Plot saved : {out}")


# ── Kernel CSV analysis ────────────────────────────────────────────────────────

def analyze_kernel(filename, label):
    header, data = load_csv(filename)
    fmt = detect_format(header)
    if fmt != 'kernel':
        print(f"  WARNING: {filename} does not look like a kernel CSV (got {fmt})")
        return

    n = len(data['total_ns'])
    if n == 0:
        print(f"\n  Kernel CSV {filename}: empty (no downtime events recorded)")
        return

    total_ns = data['total_ns']
    wait_ns  = data['wait_ns']
    already  = data['was_already_done']

    real_waits = wait_ns[already == 0]
    already_done = np.sum(already == 1)

    print(f"\n{'='*68}")
    print(f"  KERNEL DOWNTIME: {label}")
    print(f"  File   : {filename}")
    print(f"  Records: {n}")
    print(f"{'='*68}")

    print(f"\n  Records where migration was still in progress : {int(np.sum(already==0))}")
    print(f"  Records where migration had already finished  : {int(already_done)}")

    if len(real_waits) > 0:
        print(f"\n  Actual wait time (migration_entry_wait() duration):")
        print(f"    Max    : {np.max(real_waits)/1000:.2f} µs")
        print(f"    Mean   : {np.mean(real_waits)/1000:.2f} µs")
        print(f"    Min    : {np.min(real_waits)/1000:.2f} µs")
        print(f"\n  Total time in migration_entry_wait() (incl. overhead):")
        print(f"    Max    : {np.max(total_ns)/1000:.2f} µs")

    if 'page_order' in header:
        orders = data['page_order']
        for order in sorted(set(orders)):
            mask = orders == order
            size = f"{4 << int(order)}KB" if order < 9 else "2MB"
            n_order = int(np.sum(mask))
            w = wait_ns[mask & (already == 0)]
            if len(w) > 0:
                print(f"\n  Page size {size} (order {int(order)}, n={n_order}):")
                print(f"    Mean wait : {np.mean(w)/1000:.2f} µs")
                print(f"    Max wait  : {np.max(w)/1000:.2f} µs")


# ── Comparison table ──────────────────────────────────────────────────────────

def compare(results):
    if len(results) < 2:
        return
    valid = [r for r in results if r is not None]
    if len(valid) < 2:
        return

    print(f"\n{'='*68}")
    print(f"  COMPARISON")
    print(f"{'='*68}")
    print(f"  {'Config':<35} {'Normal(ns)':>12} {'MaxStall(µs)':>13} {'Stalls':>8}")
    print(f"  {'------':<35} {'----------':>12} {'-----------':>13} {'------':>8}")
    for r in valid:
        print(f"  {r['label']:<35} {r['normal_mean']:>12.1f}"
              f" {r['max_stall']/1000:>13.2f} {r['n_stall']:>8}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    files = sys.argv[1:]
    results = []

    for path in files:
        if not os.path.exists(path):
            print(f"  ERROR: file not found: {path}")
            continue

        _, data = load_csv(path)
        header_line = open(path).readline().strip().split(',')
        fmt   = detect_format(header_line)
        label = os.path.basename(path).replace('.csv', '').replace('downtime_', '')

        if fmt == 'userspace':
            r = analyze_userspace(path, label)
            results.append(r)
        elif fmt == 'kernel':
            analyze_kernel(path, label)
        else:
            print(f"  WARNING: unrecognised format in {path}")

    compare(results)


if __name__ == '__main__':
    main()