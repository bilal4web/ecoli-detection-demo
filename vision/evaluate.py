"""Evaluation of the demo detector against synthetic ground truth.

Because the plate images are procedurally generated, the exact position
and radius of every colony-like blob is known. This module matches
detections to ground truth with a greedy nearest-neighbor assignment and
reports counting metrics.

READ THIS BEFORE QUOTING ANY NUMBER: every metric here measures the demo
pipeline on SYNTHETIC data only. It is NOT a claim about real-world
detection accuracy, sensitivity, specificity, or field performance. The
numbers exist to show the pipeline behaving sanely (error grows with
density, stays near zero on clean plates), nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision.detect import Detection, DetectorConfig


@dataclass
class EvalMetrics:
    """Counting metrics for one synthetic plate.

    Attributes:
        n_true: Ground-truth colony count.
        n_detected: Raw detector output count.
        tp: True positives (detections matched to a true colony).
        fp: False positives (detections with no nearby true colony).
        fn: False negatives (true colonies with no nearby detection).
        count_error: n_detected - n_true (signed).
        abs_count_error: abs(count_error).
        precision: tp / (tp + fp), 0.0 when no detections.
        recall: tp / (tp + fn), 0.0 when no true colonies.
    """

    n_true: int
    n_detected: int
    tp: int
    fp: int
    fn: int
    count_error: int
    abs_count_error: int
    precision: float
    recall: float


def match_detections(
    detections: list[Detection],
    ground_truth: list[tuple[float, float, float]],
    tolerance: float = 1.0,
) -> tuple[list[tuple[Detection, tuple[float, float, float]]],
           list[Detection],
           list[tuple[float, float, float]]]:
    """Greedily match detections to ground-truth colonies.

    Each detection is assigned to its nearest unmatched ground-truth
    colony if the center distance is within ``tolerance * true_radius``.
    Greedy matching is O(N*M) but N and M are small here.

    Args:
        detections: Detector output list.
        ground_truth: List of (x, y, radius) true colonies.
        tolerance: Match radius as a fraction of the true colony radius.

    Returns:
        (matches, unmatched_detections, unmatched_ground_truth) where
        matches is a list of (detection, ground_truth) pairs.
    """
    remaining_gt = list(enumerate(ground_truth))
    matches: list[tuple[Detection, tuple[float, float, float]]] = []
    unmatched_det: list[Detection] = []

    for det in detections:
        best_idx: int | None = None
        best_dist = float("inf")
        for i, (gx, gy, gr) in remaining_gt:
            dist = float(np.hypot(det.x - gx, det.y - gy))
            if dist <= tolerance * gr and dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is None:
            unmatched_det.append(det)
        else:
            gt = ground_truth[best_idx]
            matches.append((det, gt))
            remaining_gt = [(i, g) for i, g in remaining_gt if i != best_idx]

    unmatched_gt = [g for _, g in remaining_gt]
    return matches, unmatched_det, unmatched_gt


def evaluate(detections: list[Detection],
             ground_truth: list[tuple[float, float, float]],
             cfg: DetectorConfig | None = None) -> EvalMetrics:
    """Compute counting metrics for one plate.

    Args:
        detections: Detector output list.
        ground_truth: List of (x, y, radius) true colonies.
        cfg: DetectorConfig for the match tolerance; defaults used
            when None.

    Returns:
        EvalMetrics for this plate. All values describe synthetic-demo
        behavior only.
    """
    cfg = cfg or DetectorConfig()
    matches, unmatched_det, unmatched_gt = match_detections(
        detections, ground_truth, tolerance=cfg.match_tolerance)
    tp = len(matches)
    fp = len(unmatched_det)
    fn = len(unmatched_gt)
    n_true = len(ground_truth)
    n_detected = len(detections)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return EvalMetrics(
        n_true=n_true,
        n_detected=n_detected,
        tp=tp,
        fp=fp,
        fn=fn,
        count_error=n_detected - n_true,
        abs_count_error=abs(n_detected - n_true),
        precision=precision,
        recall=recall,
    )
