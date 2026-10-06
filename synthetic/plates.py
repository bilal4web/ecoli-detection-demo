"""Procedural generator of synthetic culture-plate-like images.

EVERY image produced by this module is SYNTHETIC — drawn by seeded
pseudo-random number generators for demonstration purposes only. Nothing
here is a photograph of a real culture plate, and no real biological,
detection-accuracy, or field data is used or implied anywhere.

The generator models the *visual structure* of a plate photograph
(background texture, bright colony-like blobs, common imaging artifacts)
so that a classical computer-vision pipeline has something realistic to
chew on. Ground truth (exact blob positions and radii) is known because we
drew the blobs ourselves, which lets the demo measure its own counting
error on synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class PlateSpec:
    """Configuration for one synthetic plate image.

    Attributes:
        size: Output image size as (height, width) in pixels.
        n_colonies: Number of colony-like blobs to draw.
        radius_range: (min, max) blob radius in pixels.
        seed: RNG seed for reproducibility.
        add_artifacts: Whether to draw dust/scratch/glare artifacts.
        overlap_prob: Probability that a new blob is placed near an
            existing one (simulates clustered growth).
    """

    size: tuple[int, int] = (512, 512)
    n_colonies: int = 40
    radius_range: tuple[int, int] = (6, 16)
    seed: int = 0
    add_artifacts: bool = True
    overlap_prob: float = 0.25


@dataclass
class SyntheticPlate:
    """A generated plate image plus its ground truth.

    Attributes:
        image: BGR uint8 image, shape (H, W, 3).
        colonies: List of (x, y, radius) ground-truth blobs.
        spec: The PlateSpec used to generate it.
    """

    image: np.ndarray
    colonies: list[tuple[float, float, float]] = field(default_factory=list)
    spec: PlateSpec | None = None


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int],
                  scale: float = 0.04) -> np.ndarray:
    """Return smooth low-frequency noise in [0, 1] for background texture.

    Args:
        rng: Seeded numpy random generator.
        shape: (height, width) of the noise field.
        scale: Relative amplitude of the texture.

    Returns:
        Float32 array in [0, 1] with smooth spatial variation.
    """
    h, w = shape
    # Coarse random field, upscaled with smooth interpolation -> cloudy texture.
    coarse = rng.random((max(h // 32, 2), max(w // 32, 2))).astype(np.float32)
    smooth = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    smooth = (smooth - smooth.min()) / (smooth.ptp() + 1e-9)
    return (smooth * scale).astype(np.float32)


def _radial_vignette(h: int, w: int, strength: float = 0.25) -> np.ndarray:
    """Return a radial vignette mask in [1-strength, 1].

    Args:
        h: Image height in pixels.
        w: Image width in pixels.
        strength: How much the corners darken (0 = none).

    Returns:
        Float32 (H, W) multiplier mask.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist = dist / dist.max()
    return (1.0 - strength * dist ** 2).astype(np.float32)


def _draw_blob(canvas: np.ndarray, x: float, y: float, r: float,
               intensity: float) -> None:
    """Draw one soft-edged bright blob (a synthetic "colony") in place.

    The blob has a bright core fading to the background, approximating the
    look of a raised colony under diffuse light.

    Args:
        canvas: Float32 (H, W) single-channel image, modified in place.
        x: Blob center x-coordinate in pixels.
        y: Blob center y-coordinate in pixels.
        r: Blob radius in pixels.
        intensity: Peak brightness added at the blob center.
    """
    h, w = canvas.shape
    ir = int(np.ceil(r * 2.5))
    x0, x1 = max(int(x) - ir, 0), min(int(x) + ir + 1, w)
    y0, y1 = max(int(y) - ir, 0), min(int(y) + ir + 1, h)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / max(r, 1e-6)
    # Smooth falloff: bright core, soft edge.
    profile = np.clip(1.0 - dist ** 1.8, 0.0, 1.0) ** 0.8
    canvas[y0:y1, x0:x1] += (profile * intensity).astype(np.float32)


def _draw_artifacts(rng: np.random.Generator, canvas: np.ndarray) -> None:
    """Add common imaging artifacts: dust specks, scratches, glare.

    These are intentionally harder than colonies (darker, thinner, larger
    soft regions) so the detector must reject them by shape/size.

    IMPORTANT: artifacts are blended additively/subtractively via masks.
    Using cv2 drawing functions with a scalar "color" on a float canvas
    would *replace* pixel values instead of modulating them.

    Args:
        rng: Seeded numpy random generator.
        canvas: Float32 (H, W) image, modified in place.
    """
    h, w = canvas.shape
    # Dust specks: tiny dark dots (subtractive).
    for _ in range(int(rng.integers(15, 40))):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.uniform(1.0, 2.5))
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (x, y), r, 1.0, -1, cv2.LINE_AA)
        canvas -= mask * float(rng.uniform(0.15, 0.35))
    # Scratches: thin dark lines (subtractive).
    for _ in range(int(rng.integers(1, 4))):
        x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
        ang = rng.uniform(0, np.pi)
        length = rng.uniform(w * 0.1, w * 0.4)
        x2 = int(x1 + length * np.cos(ang))
        y2 = int(y1 + length * np.sin(ang))
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.line(mask, (x1, y1), (x2, y2), 1.0, 1, cv2.LINE_AA)
        canvas -= mask * float(rng.uniform(0.10, 0.25))
    # Glare: one large soft bright ellipse (additive).
    cx, cy = int(rng.integers(w // 4, 3 * w // 4)), int(rng.integers(h // 4, 3 * h // 4))
    axes = (int(rng.uniform(w * 0.08, w * 0.18)),
            int(rng.uniform(h * 0.06, h * 0.14)))
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    # Soften the glare edge with a blur so it looks like diffuse light.
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    canvas += mask * float(rng.uniform(0.08, 0.18))


def generate_plate(spec: PlateSpec) -> SyntheticPlate:
    """Generate one synthetic plate image with known ground truth.

    Args:
        spec: PlateSpec describing the desired image.

    Returns:
        SyntheticPlate with the BGR image and (x, y, radius) colony list.
        Returns an empty colony list when spec.n_colonies == 0.
    """
    rng = np.random.default_rng(spec.seed)
    h, w = spec.size

    # Base background: mid-gray with vignette and cloudy texture.
    canvas = np.full((h, w), 0.55, dtype=np.float32)
    canvas *= _radial_vignette(h, w)
    canvas += _smooth_noise(rng, (h, w))
    # Fine sensor noise.
    canvas += rng.normal(0.0, 0.02, (h, w)).astype(np.float32)

    colonies: list[tuple[float, float, float]] = []
    rmin, rmax = spec.radius_range
    margin = rmax + 4
    placed: list[tuple[float, float, float]] = []

    def _too_close(x: float, y: float, r: float) -> bool:
        """Check if (x, y, r) overlaps any placed colony too heavily.

        Args:
            x: Proposed center x-coordinate.
            y: Proposed center y-coordinate.
            r: Proposed radius.

        Returns:
            True if the center distance to any placed colony is less
            than 1.0 * (r1 + r2), i.e. more than just touching. Colonies
            may touch but not overlap, which keeps the synthetic blobs
            separable by the watershed stage.
        """
        for px, py, pr in placed:
            if np.hypot(x - px, y - py) < 1.0 * (r + pr):
                return True
        return False

    for _ in range(spec.n_colonies):
        r = float(rng.uniform(rmin, rmax))
        x, y = margin, margin  # placeholder
        for _attempt in range(30):
            if placed and rng.random() < spec.overlap_prob:
                # Cluster near an existing colony, but keep centers at
                # least (r1+r2) apart so blobs touch rather than overlap.
                px, py, pr = placed[int(rng.integers(len(placed)))]
                ang = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(1.0 * (r + pr), 1.8 * (r + pr))
                x = float(np.clip(px + dist * np.cos(ang),
                                  margin, w - margin))
                y = float(np.clip(py + dist * np.sin(ang),
                                  margin, h - margin))
            else:
                x = float(rng.uniform(margin, w - margin))
                y = float(rng.uniform(margin, h - margin))
            if not _too_close(x, y, r):
                break
        intensity = float(rng.uniform(0.30, 0.45))
        _draw_blob(canvas, x, y, r, intensity)
        colonies.append((x, y, r))
        placed.append((x, y, r))

    if spec.add_artifacts:
        _draw_artifacts(rng, canvas)

    canvas = np.clip(canvas, 0.0, 1.0)
    # Warm tint -> BGR uint8 image.
    gray8 = (canvas * 255).astype(np.uint8)
    bgr = cv2.merge([
        np.clip(gray8.astype(np.float32) * 0.92, 0, 255).astype(np.uint8),
        gray8,
        np.clip(gray8.astype(np.float32) * 1.04, 0, 255).astype(np.uint8),
    ])
    return SyntheticPlate(image=bgr, colonies=colonies, spec=spec)


def save_plate(plate: SyntheticPlate, path: str) -> None:
    """Write a synthetic plate image to disk as PNG.

    Args:
        plate: The SyntheticPlate to save.
        path: Destination file path (PNG recommended).

    Raises:
        IOError: If the image cannot be written.
    """
    ok = cv2.imwrite(path, plate.image)
    if not ok:
        raise IOError(f"Failed to write plate image to {path}")
