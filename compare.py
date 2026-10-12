"""Preprocessing, geometry, evaluation and plots for image-pair experiments."""
import argparse
import csv
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from matching import Matcher


def resize_image(image, long_side=640):
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a non-empty uint8 BGR image")
    if long_side < 32 or not image.size:
        raise ValueError("long_side must be at least 32")
    h, w = image.shape[:2]
    factor = min(1.0, long_side / max(h, w))
    nw, nh = max(1, round(w * factor)), max(1, round(h * factor))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA) if factor < 1 else image.copy()
    return resized, np.diag([nw / w, nh / h, 1.0])


def normalize_pair(image0, image1, strength=0.45):
    """Optional shared luminance adjustment, bounded to 24 * strength L levels.

    Flat images are left unchanged. This is an experimental preprocessing
    option; it does not restore missing overlap or correct viewpoint changes.
    """
    if not 0 <= strength <= 0.65:
        raise ValueError("strength must be between 0 and 0.65")
    labs = [cv2.cvtColor(im, cv2.COLOR_BGR2LAB) for im in (image0, image1)]
    stats = [np.percentile(lab[:, :, 0], [10, 50, 90]) for lab in labs]
    target = float(np.mean([s[1] for s in stats]))
    result = []
    for image, lab, (low, median, high) in zip((image0, image1), labs, stats):
        if high - low < 8 or strength == 0:
            result.append(image.copy())
            continue
        delta = float(np.clip(target - median, -24, 24)) * strength
        lab[:, :, 0] = np.clip(np.round(lab[:, :, 0].astype(float) + delta), 0, 255).astype(np.uint8)
        result.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
    return tuple(result)


def geometric_check(p0, p1, shape0, shape1, threshold_px=3.0):
    """Fit a homography with MAGSAC. Passing is evidence of consistency only."""
    p0, p1 = np.asarray(p0, np.float32), np.asarray(p1, np.float32)
    mask = np.zeros(len(p0), dtype=bool)
    info = dict(matches=len(p0), inliers=0, inlier_ratio=0.0, coverage0=0.0,
                coverage1=0.0, geometry_pass=False, reason="too_few_matches")
    if len(p0) < 4:
        return info, None, mask
    matrix, raw_mask = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, threshold_px,
                                        maxIters=10000, confidence=0.999)
    if matrix is None or raw_mask is None or not np.isfinite(matrix).all():
        info["reason"] = "homography_failed"
        return info, None, mask
    mask = raw_mask.reshape(-1).astype(bool)
    info.update(inliers=int(mask.sum()), inlier_ratio=float(mask.mean()))
    for index, (points, shape) in enumerate(((p0, shape0), (p1, shape1))):
        h, w = shape[:2]
        area = cv2.contourArea(cv2.convexHull(points[mask])) if mask.sum() >= 3 else 0.0
        info[f"coverage{index}"] = float(area / (w * h))
    h0, w0 = shape0[:2]
    corners = np.float32([[0, 0], [w0 - 1, 0], [w0 - 1, h0 - 1], [0, h0 - 1]])
    projected = cv2.perspectiveTransform(corners[None], matrix)[0]
    contour = projected.astype(np.float32)
    h1, w1 = shape1[:2]
    area = abs(cv2.contourArea(contour))
    sane = (np.isfinite(projected).all() and cv2.isContourConvex(contour)
            and area > 1 and area < 10 * w1 * h1)
    if info["inliers"] < 18:
        info["reason"] = "too_few_inliers"
    elif info["inlier_ratio"] < 0.2:
        info["reason"] = "low_inlier_ratio"
    elif min(info["coverage0"], info["coverage1"]) < 0.005:
        info["reason"] = "low_coverage"
    elif not sane:
        info["reason"] = "invalid_projection"
    else:
        info.update(geometry_pass=True, reason="passed")
    return info, matrix, mask


def corner_error(matrix, truth, shape):
    """Mean corner transfer error in prepared query pixels, if truth is known."""
    if matrix is None:
        return None
    h, w = shape[:2]
    points = np.float32([[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]])
    actual = cv2.perspectiveTransform(points, matrix)
    expected = cv2.perspectiveTransform(points, np.asarray(truth, dtype=float))
    error = float(np.linalg.norm(actual - expected, axis=-1).mean())
    return error if np.isfinite(error) else None


def evaluate_pair(matcher, image0, image1, truth=None, normalize=False):
    prepared = normalize_pair(image0, image1) if normalize else (image0, image1)
    matcher.synchronize()
    start = time.perf_counter()
    p0, p1 = matcher(*prepared)
    info, matrix, mask = geometric_check(p0, p1, image0.shape, image1.shape)
    matcher.synchronize()
    info["pair_ms"] = (time.perf_counter() - start) * 1000
    if truth is not None:
        info["corner_error_px"] = corner_error(matrix, truth, image0.shape)
    else:
        info["corner_error_px"] = None
    return info, p0, p1, mask, matrix


def plot_matches(image0, image1, p0, p1, mask, title="Matches", matrix=None):
    h = max(image0.shape[0], image1.shape[0])
    offset = image0.shape[1]
    canvas = np.full((h, offset + image1.shape[1], 3), 245, dtype=np.uint8)
    canvas[:image0.shape[0], :offset] = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
    canvas[:image1.shape[0], offset:] = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.imshow(canvas)
    for i in np.linspace(0, len(p0) - 1, min(60, len(p0)), dtype=int) if len(p0) else []:
        color = "#0c9975" if mask[i] else "#e25757"
        ax.plot([p0[i, 0], p1[i, 0] + offset], [p0[i, 1], p1[i, 1]], color=color,
                alpha=0.7, linewidth=0.6)
    if matrix is not None:
        ih, iw = image0.shape[:2]
        corners = np.float32([[[0, 0], [iw - 1, 0], [iw - 1, ih - 1], [0, ih - 1], [0, 0]]])
        outline = cv2.perspectiveTransform(corners, matrix)[0]
        ax.plot(outline[:, 0] + offset, outline[:, 1], color="#2374cb", linewidth=2)
    ax.set_title(title, fontsize=13, loc="left", pad=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


def make_demo(seed=7):
    """Generate a reproducible planar scene and its known perspective warp."""
    rng = np.random.default_rng(seed)
    image = np.full((360, 480, 3), 238, np.uint8)
    for _ in range(180):
        center = tuple(int(x) for x in rng.integers([10, 10], [470, 350]))
        color = tuple(int(x) for x in rng.integers(20, 210, 3))
        cv2.circle(image, center, int(rng.integers(2, 11)), color, -1, cv2.LINE_AA)
    for x in range(40, 450, 70):
        cv2.putText(image, str(x), (x, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    matrix = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [479, 0], [479, 359], [0, 359]]),
        np.float32([[38, 26], [448, 12], [465, 332], [14, 345]]))
    query = cv2.warpPerspective(image, matrix, (480, 360), borderValue=(245, 245, 245))
    return image, query, matrix


def benchmark(pairs, models=("sift",), output="runs", long_side=640,
              normalize=False, device="auto", repeats=3, seed=7):
    """Compare models on a shared list of pairs; retain failed pair records.

    Each pair has reference/query paths and optional H (original pixel units).
    A warm-up is excluded. pair_ms includes both feature extractions, matching
    and MAGSAC; I/O, resizing, normalization, plotting and truth scoring are
    excluded. Per-pair runtime is the median of `repeats` measured calls.
    """
    if not pairs or repeats < 1:
        raise ValueError("Provide at least one image pair and repeats >= 1")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows, settings = [], {}
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    for kind in models:
        start = time.perf_counter()
        matcher = Matcher(kind, device=device)
        matcher.synchronize()
        settings[kind] = {**matcher.settings, "device": matcher.device,
                          "model_load_s": time.perf_counter() - start}
        if matcher.torch is not None:
            matcher.torch.manual_seed(seed)
        warmed_up = False
        for index, pair in enumerate(pairs):
            row = dict(model=kind, pair=index, normalized=normalize, status="ok",
                       matches=0, inliers=0, inlier_ratio=0.0, coverage0=0.0,
                       coverage1=0.0, geometry_pass=False, reason="", pair_ms=None,
                       corner_error_px=None)
            try:
                originals = [cv2.imread(str(pair[key])) for key in ("reference", "query")]
                a, s0 = resize_image(originals[0], long_side)
                b, s1 = resize_image(originals[1], long_side)
                truth = s1 @ np.asarray(pair["H"], float) @ np.linalg.inv(s0) if "H" in pair else None
                prepared = normalize_pair(a, b) if normalize else (a, b)
                if not warmed_up:
                    evaluate_pair(matcher, *prepared)
                    warmed_up = True
                measurements = [evaluate_pair(matcher, *prepared) for _ in range(repeats)]
                # Use the middle measured run for correspondence diagnostics.
                selected = sorted(measurements, key=lambda r: r[0]["pair_ms"])[repeats // 2]
                info, p0, p1, mask, matrix = selected
                row.update(info, pair_ms=float(np.median([r[0]["pair_ms"] for r in measurements])))
                row["corner_error_px"] = corner_error(matrix, truth, a.shape) if truth is not None else None
                fig = plot_matches(a, b, p0, p1, mask,
                                   f"{matcher.name} | pair {index} | {info['inliers']} inliers",
                                   matrix if info["geometry_pass"] else None)
                fig.savefig(out / f"{kind}_{index:03d}.png", dpi=140)
                plt.close(fig)
            except (OSError, ValueError, cv2.error) as error:
                row.update(status="error", reason=f"{type(error).__name__}: {error}")
            # CUDA/model failures propagate: retry in a fresh kernel, not in a broken context.
            rows.append(row)
            with (out / "metrics.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        del matcher
        gc.collect()
    versions = {}
    for package in ("numpy", "opencv-python", "opencv-python-headless", "torch", "lightglue", "kornia", "romatch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "versions": versions, "long_side": long_side, "normalize": normalize,
                   "normalization_strength": 0.45 if normalize else 0.0,
                   "repeats": repeats, "seed": seed, "models": settings,
                   "geometry": {"method": "USAC_MAGSAC", "threshold_px": 3.0,
                                "min_inliers": 18, "min_ratio": 0.2, "min_coverage": 0.005}}
    torch = __import__("sys").modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        environment.update(gpu=torch.cuda.get_device_name(0), torch_cuda=torch.version.cuda)
    (out / "run.json").write_text(json.dumps(environment, indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare image matching methods.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--demo", action="store_true", help="Create a synthetic pair with known geometry")
    inputs.add_argument("--pairs", type=Path, help="JSON list of reference/query paths; optional H")
    parser.add_argument("--models", nargs="+", choices=("sift", "lightglue", "loftr", "roma"), default=["sift"])
    parser.add_argument("--output", default="runs")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--long-side", type=int, default=640)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.demo:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        a, b, matrix = make_demo()
        cv2.imwrite(str(out / "reference.png"), a)
        cv2.imwrite(str(out / "query.png"), b)
        pairs = [{"reference": str(out / "reference.png"), "query": str(out / "query.png"), "H": matrix.tolist()}]
    else:
        pairs = json.loads(args.pairs.read_text())
    rows = benchmark(pairs, args.models, args.output, args.long_side,
                     args.normalize, args.device, args.repeats)
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
