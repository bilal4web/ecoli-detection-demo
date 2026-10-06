#!/usr/bin/env python3
"""One-command end-to-end demo: generate -> detect -> evaluate -> plots -> dashboard.

Usage:
    python3 run_demo.py [--densities 10 40 80 120] [--seed 7] [--size 512]

Pipeline:
    1. Generate synthetic plates at several colony densities (fixed seeds).
    2. Run the classical CV detector on each (timed).
    3. Evaluate detections against known ground truth.
    4. Render plots and a static dashboard.html.

All outputs land in results/. Every number produced is synthetic-demo
output; nothing here describes real-world detection performance.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import time

import cv2
import numpy as np

from analysis.plots import (plot_accuracy_vs_density, plot_detection_overlay,
                            plot_size_histogram, plot_timing)
from synthetic.plates import PlateSpec, generate_plate, save_plate
from vision.detect import DetectorConfig, detect_colonies
from vision.evaluate import evaluate

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

DISCLAIMER = (
    "EDUCATIONAL DEMONSTRATION ONLY — not a medical device, not for "
    "diagnostic use, not the actual CISNR system or its data. "
    "All images are synthetic and all metrics describe synthetic-demo "
    "behavior only."
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the demo run.

    Returns:
        Parsed argument namespace.
    """
    p = argparse.ArgumentParser(description="E. coli detection CV demo")
    p.add_argument("--densities", type=int, nargs="+",
                   default=[0, 10, 40, 80, 120],
                   help="Colony counts per plate to generate")
    p.add_argument("--plates-per-density", type=int, default=4,
                   help="Plates to generate at each density")
    p.add_argument("--seed", type=int, default=7,
                   help="Base RNG seed (each plate gets seed+index)")
    p.add_argument("--size", type=int, default=512,
                   help="Plate image side length in pixels")
    return p.parse_args()


def run_detection_suite(densities: list[int], plates_per_density: int,
                        seed: int, size: int) -> dict:
    """Generate plates, detect, evaluate, and collect summary statistics.

    Args:
        densities: Colony counts to generate per plate.
        plates_per_density: Number of plates at each density.
        seed: Base RNG seed.
        size: Image side length in pixels.

    Returns:
        Summary dict with per-plate records and aggregate metrics.
    """
    cfg = DetectorConfig()
    records: list[dict] = []
    all_radii: list[float] = []
    plate_idx = 0

    for n in densities:
        for rep in range(plates_per_density):
            spec = PlateSpec(size=(size, size), n_colonies=n,
                             seed=seed + plate_idx)
            plate = generate_plate(spec)
            img_path = os.path.join(RESULTS, f"plate_{plate_idx:02d}.png")
            save_plate(plate, img_path)

            t0 = time.perf_counter()
            result = detect_colonies(plate.image, cfg, keep_debug=True)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            metrics = evaluate(result.detections, plate.colonies, cfg)
            all_radii.extend(d.radius for d in result.detections)
            records.append({
                "plate": plate_idx,
                "n_true": metrics.n_true,
                "n_detected": metrics.n_detected,
                "tp": metrics.tp,
                "fp": metrics.fp,
                "fn": metrics.fn,
                "count_error": metrics.count_error,
                "abs_count_error": metrics.abs_count_error,
                "precision": round(metrics.precision, 3),
                "recall": round(metrics.recall, 3),
                "time_ms": round(dt_ms, 1),
            })
            # Keep one annotated sample per density for the dashboard.
            if rep == 0 and result.annotated is not None:
                plot_detection_overlay(
                    plate.image, result.annotated,
                    os.path.join(RESULTS, f"overlay_density_{n}.png"),
                    title=f"Synthetic plate, {n} colonies "
                          f"(seed {seed + plate_idx})")
            plate_idx += 1

    n_true = [r["n_true"] for r in records]
    abs_err = [r["abs_count_error"] for r in records]
    times = [r["time_ms"] for r in records]
    summary = {
        "disclaimer": DISCLAIMER,
        "n_plates": len(records),
        "densities": densities,
        "mean_abs_count_error": round(float(np.mean(abs_err)), 2),
        "mean_time_ms": round(float(np.mean(times)), 1),
        "mean_precision": round(float(np.mean([r["precision"] for r in records])), 3),
        "mean_recall": round(float(np.mean([r["recall"] for r in records])), 3),
        "records": records,
    }
    return summary, all_radii, n_true, abs_err


def run_timing_sweep(sizes: list[int]) -> tuple[list[int], list[float]]:
    """Measure mean per-image processing time at several image sizes.

    Args:
        sizes: Square image side lengths in pixels.

    Returns:
        (sizes, mean milliseconds per image).
    """
    cfg = DetectorConfig()
    times: list[float] = []
    for s in sizes:
        spec = PlateSpec(size=(s, s), n_colonies=40, seed=1234)
        plate = generate_plate(spec)
        # Warmup, then timed repeats.
        detect_colonies(plate.image, cfg)
        t0 = time.perf_counter()
        reps = 5
        for _ in range(reps):
            detect_colonies(plate.image, cfg)
        times.append((time.perf_counter() - t0) * 1000.0 / reps)
    return sizes, times


def _svg_pipeline() -> str:
    """Return an inline SVG diagram of the demo pipeline stages.

    Returns:
        SVG markup string (capture -> preprocess -> detect -> count).
    """
    stages = ["Capture\n(synthetic)", "Preprocess\n(denoise+normalize)",
              "Detect\n(threshold+morphology\n+watershed)", "Count\n+ evaluate"]
    x0, w, gap = 20, 150, 30
    parts = ['<svg viewBox="0 0 780 120" width="100%" '
             'style="max-width:760px" role="img" '
             'aria-label="Pipeline diagram">']
    for i, label in enumerate(stages):
        x = x0 + i * (w + gap)
        parts.append(
            f'<rect x="{x}" y="20" width="{w}" height="80" rx="10" '
            f'fill="#eef4f3" stroke="#0f766e" stroke-width="2"/>')
        lines = label.split("\n")
        for j, line in enumerate(lines):
            parts.append(
                f'<text x="{x + w // 2}" y="{48 + j * 18}" '
                f'text-anchor="middle" font-size="13" fill="#1f2937">'
                f'{html.escape(line)}</text>')
        if i < len(stages) - 1:
            x1 = x + w
            parts.append(
                f'<line x1="{x1}" y1="60" x2="{x1 + gap}" y2="60" '
                f'stroke="#0f766e" stroke-width="2" marker-end="url(#arr)"/>')
    parts.append('<defs><marker id="arr" markerWidth="8" markerHeight="8" '
                 'refX="7" refY="4" orient="auto">'
                 '<path d="M0,0 L8,4 L0,8" fill="none" stroke="#0f766e" '
                 'stroke-width="2"/></marker></defs>')
    parts.append("</svg>")
    return "".join(parts)


def write_dashboard(summary: dict, densities: list[int]) -> None:
    """Write results/dashboard.html summarizing the demo run.

    Args:
        summary: Summary dict from run_detection_suite.
        densities: Densities used, for overlay image gallery.
    """
    cards = [
        ("Plates processed", str(summary["n_plates"])),
        ("Mean |count error|", str(summary["mean_abs_count_error"])),
        ("Mean time / image", f"{summary['mean_time_ms']} ms"),
        ("Mean precision*", str(summary["mean_precision"])),
        ("Mean recall*", str(summary["mean_recall"])),
    ]
    card_html = "".join(
        f'<div class="card"><div class="num">{html.escape(v)}</div>'
        f'<div class="lbl">{html.escape(k)}</div></div>'
        for k, v in cards)
    gallery = "".join(
        f'<figure><img src="overlay_density_{n}.png" '
        f'alt="Detection overlay at density {n}">'
        f'<figcaption>{n} colonies (synthetic)</figcaption></figure>'
        for n in densities)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>E. coli Detection Kit — Computer Vision Demo (Synthetic)</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
max-width:960px;margin:0 auto;padding:24px;color:#1f2937;background:#fafafa}}
.banner{{background:#fef3c7;border:2px solid #d97706;border-radius:10px;
padding:14px 18px;margin-bottom:24px;font-size:14px}}
h1{{color:#0f766e}}h2{{color:#0f766e;margin-top:36px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap}}
.card{{background:#fff;border:1px solid #d1d5db;border-radius:10px;
padding:14px 18px;min-width:130px;text-align:center}}
.num{{font-size:26px;font-weight:700;color:#0f766e}}
.lbl{{font-size:12px;color:#6b7280;margin-top:4px}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
gap:16px}}figure{{margin:0;background:#fff;border:1px solid #d1d5db;
border-radius:10px;overflow:hidden}}figure img{{width:100%;display:block}}
figcaption{{padding:8px 12px;font-size:13px;color:#4b5563}}
table{{border-collapse:collapse;width:100%;font-size:13px;background:#fff}}
th,td{{border:1px solid #d1d5db;padding:6px 10px;text-align:right}}
th{{background:#eef4f3}}td:first-child,th:first-child{{text-align:left}}
.note{{font-size:12px;color:#6b7280;margin-top:24px}}
img.chart{{max-width:100%;background:#fff;border:1px solid #d1d5db;
border-radius:10px;margin-top:12px}}
</style>
</head>
<body>
<div class="banner"><strong>Disclaimer:</strong> {html.escape(DISCLAIMER)}</div>
<h1>E. coli Detection Kit — Computer Vision Demo</h1>
<p>Educational demonstration of the <em>detection/vision</em> side of an
IoT rapid-detection kit: synthetic plate images &rarr; classical OpenCV
pipeline (no machine learning) &rarr; colony counting with known ground
truth. Companion to the
<a href="https://github.com/bilal4web/iot-sensor-pipeline">sensor-to-cloud
pipeline simulator</a> (project #2).</p>
<h2>Pipeline</h2>
{_svg_pipeline()}
<h2>Run summary</h2>
<div class="cards">{card_html}</div>
<p class="note">* Precision/recall measured against synthetic ground truth
only — NOT a claim about real-world performance.</p>
<h2>Detection examples</h2>
<div class="gallery">{gallery}</div>
<h2>Analysis</h2>
<img class="chart" src="accuracy_vs_density.png"
alt="Counting error vs colony density chart">
<img class="chart" src="timing.png" alt="Processing time chart">
<img class="chart" src="size_histogram.png"
alt="Detected colony size histogram">
<h2>Honesty note</h2>
<p class="note">Everything in this repository is synthetic: the images are
procedurally generated, the "colonies" are drawn blobs, and every metric
measures the demo pipeline on that synthetic data. It is inspired by the
<em>architecture</em> of Bilal Ahmad's CISNR Junior Research Engineer work
(Dec 2022 – Feb 2024) on an E. coli rapid-detection kit, but it is not the
actual CISNR system, contains none of its data, and makes no claim about
real detection accuracy.</p>
</body>
</html>
"""
    path = os.path.join(RESULTS, "dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


def main() -> None:
    """Run the full demo end to end."""
    args = parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    print("[gen+detect] densities:", args.densities,
          "| plates/density:", args.plates_per_density)
    summary, all_radii, n_true, abs_err = run_detection_suite(
        args.densities, args.plates_per_density, args.seed, args.size)
    print(f"[eval] {summary['n_plates']} plates | "
          f"mean |count error| = {summary['mean_abs_count_error']} | "
          f"mean time = {summary['mean_time_ms']} ms")

    with open(os.path.join(RESULTS, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[plots] rendering charts...")
    plot_accuracy_vs_density(n_true, abs_err,
                             os.path.join(RESULTS, "accuracy_vs_density.png"))
    sizes, times = run_timing_sweep([256, 512, 1024])
    plot_timing(sizes, times, os.path.join(RESULTS, "timing.png"))
    plot_size_histogram(all_radii,
                        os.path.join(RESULTS, "size_histogram.png"))

    print("[dashboard] writing dashboard.html...")
    write_dashboard(summary, args.densities)
    print("[done] results/ ready — open results/dashboard.html "
          "(all data synthetic).")


if __name__ == "__main__":
    main()
