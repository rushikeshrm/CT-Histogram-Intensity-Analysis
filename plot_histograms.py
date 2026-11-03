import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")


def is_slice_csv(path):
    with open(path) as f:
        return f.readline().startswith("slice_index")


def load_histogram_csv(path):
    with open(path) as f:
        lines = f.read().splitlines()
    start = lines.index("bin_center,count")
    stats = {}
    for line in lines[1:start]:
        if line.startswith("#") or "," not in line:
            continue
        key, val = line.split(",", 1)
        try:
            stats[key] = float(val)
        except ValueError:
            pass
    notes = [l[1:].strip() for l in lines[1:start] if l.startswith("#")]
    hist = pd.read_csv(path, skiprows=start)
    return stats, notes, hist


def plot_histogram(name, stats, notes, hist):
    fig, ax = plt.subplots(figsize=(9, 5))
    width = hist["bin_center"].diff().median()
    ax.bar(hist["bin_center"], hist["count"], width=width, color="steelblue")
    ax.set_yscale("log")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Voxel count (log scale)")
    title = (f"{name}: mean={stats['mean']:.1f}, std={stats['stddev']:.1f}, "
             f"skew={stats['skewness']:.2f}, kurt={stats['kurtosis_excess']:.2f}")
    if notes:
        title += "\n" + " | ".join(notes)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    out = f"{name}_histogram.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved", out)


def plot_slices(path):
    name = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path)
    df = df[df["voxels"] > 0]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["slice_index"], df["mean"], color="darkred", label="mean")
    ax.fill_between(df["slice_index"], df["mean"] - df["stddev"], df["mean"] + df["stddev"],
                    color="darkred", alpha=0.2, label="mean +/- std")
    ax.set_xlabel("Slice index")
    ax.set_ylabel("Intensity")
    ax.set_title(f"{name}: per-slice statistics")
    ax.legend()
    fig.tight_layout()
    out = f"{name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved", out)


def find_default_csvs():
    # Look in the ususal folders for CSV files when none were provided.
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (os.getcwd(), here, os.path.join(here, "sample_output")):
        found = sorted(glob.glob(os.path.join(folder, "*_stats.csv")) +
                       glob.glob(os.path.join(folder, "*_slices.csv")))
        if found:
            return found
    return []


def main(paths):
    if not paths:
        paths = find_default_csvs()
    if not paths:
        print("Could not find any CSV files.")
        print("Usage: python plot_histograms.py a_stats.csv b_stats.csv metal_slices.csv")
        print("(or put *_stats.csv / *_slices.csv files next to this script)")
        return

    histograms = []
    for p in paths:
        if is_slice_csv(p):
            plot_slices(p)
        else:
            name = os.path.splitext(os.path.basename(p))[0]
            stats, notes, hist = load_histogram_csv(p)
            histograms.append((name, stats, hist))
            plot_histogram(name, stats, notes, hist)

    if len(histograms) > 1:
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, stats, hist in histograms:
            lo, hi = stats["min"], stats["max"]
            x = (hist["bin_center"] - lo) / \
                (hi - lo) if hi > lo else hist["bin_center"]
            ax.plot(x, hist["count"] / hist["count"].sum(), label=name)
        ax.set_yscale("log")
        ax.set_xlabel("Intensity (rescaled to 0-1 of each dataset's range)")
        ax.set_ylabel("Fraction of voxels (log scale)")
        ax.set_title("Histogram shape comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig("comparison_histogram.png", dpi=150)
        plt.close(fig)
        print("Saved comparison plot: comparison_histogram.png")


if __name__ == "__main__":
    main(sys.argv[1:])
