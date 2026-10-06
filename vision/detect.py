"""Classical computer-vision colony detection pipeline.

No machine learning is used here — deliberately. The pipeline is a
transparent sequence of standard image-processing steps:

    grayscale -> denoise -> white top-hat foreground extraction
        -> fixed threshold -> morphological cleanup -> watershed split
        of touching blobs -> contour analysis -> size/circularity
        filtering -> count

Every step's parameters are explicit and inspectable. Known limitation
(documented, not hidden): heavily overlapping colonies may still merge
into one detection, so counts underestimate at very high densities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    """Tunable parameters of the detection pipeline.

    Attributes:
        blur_kernel: Gaussian blur kernel size (odd) for denoising.
        tophat_kernel: White top-hat structuring element size. Must exceed
            the largest expected colony diameter so colonies are treated
            as foreground features.
        tophat_thresh: Fixed threshold applied to the top-hat response.
        open_kernel: Morphological opening kernel size (removes specks).
        min_area: Minimum contour area in px^2 to keep.
        max_area: Maximum contour area in px^2 to keep.
        min_circularity: Minimum 4*pi*area/perimeter^2 to keep.
        match_tolerance: Fraction of true radius used when matching
            detections to ground truth in evaluation.
    """

    blur_kernel: int = 7
    tophat_kernel: int = 37
    tophat_thresh: int = 25
    open_kernel: int = 3
    min_area: float = 60.0
    max_area: float = 6000.0
    min_circularity: float = 0.45
    match_tolerance: float = 1.0


@dataclass
class Detection:
    """One detected colony-like blob.

    Attributes:
        x: Centroid x-coordinate in pixels.
        y: Centroid y-coordinate in pixels.
        radius: Enclosing-circle radius in pixels.
        area: Contour area in px^2.
        circularity: 4*pi*area/perimeter^2 (1.0 = perfect circle).
    """

    x: float
    y: float
    radius: float
    area: float
    circularity: float


@dataclass
class DetectionResult:
    """Full output of one detection run (for debugging/visualization)."""

    detections: list[Detection] = field(default_factory=list)
    binary: np.ndarray | None = None      # thresholded mask
    cleaned: np.ndarray | None = None     # after morphology + watershed
    annotated: np.ndarray | None = None   # BGR image with overlays


def preprocess(gray: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    """Denoise a grayscale plate image with Gaussian blur.

    Args:
        gray: Grayscale uint8 image.
        cfg: DetectorConfig with blur_kernel.

    Returns:
        Blurred grayscale uint8 image.
    """
    k = cfg.blur_kernel if cfg.blur_kernel % 2 == 1 else cfg.blur_kernel + 1
    return cv2.GaussianBlur(gray, (k, k), 0)


def extract_foreground(blurred: np.ndarray,
                       cfg: DetectorConfig) -> np.ndarray:
    """Extract bright colony-like features via white top-hat transform.

    The white top-hat (image minus its morphological opening) keeps only
    bright structures smaller than the structuring element while
    suppressing slow background variations (vignette, glare, texture).
    This is far more robust than adaptive thresholding on CLAHE-boosted
    images, which amplifies sensor noise into speckle.

    Args:
        blurred: Denoised grayscale uint8 image.
        cfg: DetectorConfig with tophat_kernel and tophat_thresh.

    Returns:
        Binary uint8 foreground mask.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.tophat_kernel, cfg.tophat_kernel))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
    _, binary = cv2.threshold(tophat, cfg.tophat_thresh, 255,
                              cv2.THRESH_BINARY)
    return binary


def _watershed_split(binary: np.ndarray,
                     min_peak: float = 4.0,
                     peak_window: int = 9) -> np.ndarray:
    """Split touching blobs using peak-based watershed on the distance transform.

    Instead of a global fraction-of-max threshold (which under-splits when
    colony sizes vary), this finds *local maxima* of the distance transform
    — one per blob center — and uses them as watershed markers.

    Args:
        binary: Binary uint8 mask (foreground = 255).
        min_peak: Minimum distance-transform value to count as a blob
            center (filters noise ripples; ~half the smallest colony
            radius).
        peak_window: Neighborhood size in pixels for local-maximum
            detection (should be smaller than a colony diameter).

    Returns:
        Binary uint8 mask where touching blobs are separated by
        1-pixel background boundaries where the split succeeded.
    """
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return binary
    # Local maxima: pixels equal to the dilated neighborhood maximum.
    kernel = np.ones((peak_window, peak_window), np.uint8)
    dilated = cv2.dilate(dist, kernel)
    peaks = (dist == dilated) & (dist >= min_peak)
    peaks_u8 = np.uint8(peaks) * 255
    n_markers, markers = cv2.connectedComponents(peaks_u8)
    if n_markers <= 1:
        # No usable peaks (e.g., empty mask) — nothing to split.
        return binary
    # Prepare watershed markers: background=1, unknown=0.
    sure_bg = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=3)
    markers = markers + 1
    unknown = cv2.subtract(sure_bg, np.uint8(markers > 1) * 255)
    markers[unknown == 255] = 0
    canvas = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.watershed(canvas, markers)
    out = binary.copy()
    out[markers == -1] = 0
    return out


def detect_colonies(image: np.ndarray,
                    cfg: DetectorConfig | None = None,
                    keep_debug: bool = False) -> DetectionResult:
    """Run the full detection pipeline on a plate image.

    Args:
        image: BGR or grayscale uint8 plate image.
        cfg: DetectorConfig; defaults are used when None.
        keep_debug: If True, attach binary/cleaned/annotated debug images
            to the result (costs memory; off by default).

    Returns:
        DetectionResult with the filtered detection list and optional
        debug images.
    """
    cfg = cfg or DetectorConfig()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    blurred = preprocess(gray, cfg)

    # White top-hat isolates bright colony-sized features and suppresses
    # background drift; fixed threshold is stable across plates.
    binary = extract_foreground(blurred, cfg)

    # Morphological opening removes isolated specks (dust) while keeping
    # colony-sized blobs.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Split touching colonies before contour analysis.
    split = _watershed_split(cleaned)

    contours, _ = cv2.findContours(
        split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[Detection] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (cfg.min_area <= area <= cfg.max_area):
            continue
        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perim * perim)
        if circularity < cfg.min_circularity:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        m = cv2.moments(cnt)
        if m["m00"] > 0:
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        else:
            cx, cy = x, y
        detections.append(Detection(
            x=float(cx), y=float(cy), radius=float(radius),
            area=float(area), circularity=float(circularity)))

    result = DetectionResult(detections=detections)
    if keep_debug:
        result.binary = binary
        result.cleaned = split
        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for d in detections:
            cv2.circle(annotated, (int(d.x), int(d.y)), int(d.radius),
                       (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(annotated, (int(d.x), int(d.y)), 3,
                       (0, 0, 255), -1, cv2.LINE_AA)
        result.annotated = annotated
    return result


def count_colonies(image: np.ndarray,
                   cfg: DetectorConfig | None = None) -> int:
    """Count colony-like blobs in a plate image.

    Args:
        image: BGR or grayscale uint8 plate image.
        cfg: DetectorConfig; defaults are used when None.

    Returns:
        Number of detected blobs after filtering.
    """
    return len(detect_colonies(image, cfg).detections)
