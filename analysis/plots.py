"""Matplotlib charts for the demo. Every figure carries a
"SYNTHETIC DEMO DATA" watermark and uses tight layouts so no labels clip.

All numbers plotted here come from the synthetic generator + demo
detector in this repository. They describe demo behavior only.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: no display server needed
import matplotlib.pyplot as plt
import numpy as np

WATERMARK = "SYNTHETIC DEMO DATA"


def _watermark(ax: plt.Axes) -> None:
    """Stamp the synthetic-data watermark on an axes.

    Args:
        ax: Matplotlib axes to annotate.
    """
    ax.text(0.5, 0.5, WATERMARK, transform=ax.transAxes, fontsize=28,
            color="gray", alpha=0.18, ha="center", va="center", rotation=30,
            zorder=10)


def plot_detection_overlay(image: np.ndarray, annotated: np.ndarray,
                           path: str, title: str = "") -> None:
    """Save a side-by-side original vs annotated detection image.

    Args:
        image: Original BGR plate image.
        annotated: BGR image with detection overlays.
        path: Output PNG path.
        title: Optional figure suptitle.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, img, label in zip(axes,
                              [image, annotated],
                              ["Synthetic input plate", "Detections"]):
        ax.imshow(img[:, :, ::-1])  # BGR -> RGB for display
        ax.set_title(label, fontsize=11)
        ax.axis("off")
        _watermark(ax)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_density(n_true_list: list[int],
                             abs_err_list: list[int],
                             path: str) -> None:
    """Scatter absolute count error against true colony density.

    Args:
        n_true_list: Ground-truth counts, one per plate.
        abs_err_list: Absolute count errors, one per plate.
        path: Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    n_true = np.asarray(n_true_list, dtype=float)
    err = np.asarray(abs_err_list, dtype=float)
    ax.scatter(n_true, err, s=42, alpha=0.75, edgecolors="k", linewidths=0.5)
    # Binned mean trend so the density dependence is readable.
    if len(n_true) >= 4:
        order = np.argsort(n_true)
        xs, ys = n_true[order], err[order]
        bins = np.array_split(np.arange(len(xs)), max(3, len(xs) // 4))
        bx = [xs[b].mean() for b in bins if len(b)]
        by = [ys[b].mean() for b in bins if len(b)]
        ax.plot(bx, by, "r--", lw=1.6, label="binned mean")
        ax.legend(fontsize=9)
    ax.set_xlabel("True colony count (synthetic)", fontsize=11)
    ax.set_ylabel("Absolute count error |detected - true|", fontsize=11)
    ax.set_title("Demo counting error vs colony density", fontsize=12)
    ax.grid(True, alpha=0.3)
    _watermark(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_timing(sizes: list[int], times_ms: list[float], path: str) -> None:
    """Plot per-image processing time against image side length.

    Args:
        sizes: Image side lengths in pixels (square images).
        times_ms: Mean milliseconds per image at each size.
        path: Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(sizes, times_ms, "o-", lw=1.8, ms=7)
    for x, y in zip(sizes, times_ms):
        ax.annotate(f"{y:.0f} ms", (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Image side length (px)", fontsize=11)
    ax.set_ylabel("Mean processing time per image (ms)", fontsize=11)
    ax.set_title("Demo pipeline timing (synthetic images)", fontsize=12)
    ax.grid(True, alpha=0.3)
    _watermark(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_size_histogram(radii: list[float], path: str) -> None:
    """Histogram of detected colony radii (sanity check of size filter).

    Args:
        radii: Detected radii in pixels, pooled across plates.
        path: Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(radii, bins=24, edgecolor="k", linewidth=0.5, alpha=0.8)
    ax.set_xlabel("Detected radius (px)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Detected colony size distribution (synthetic)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    _watermark(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
