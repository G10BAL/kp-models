"""Offline checks for geometry, failure accounting and the classical demo."""
import json

import cv2
import numpy as np

from compare import (benchmark, corner_error, geometric_check, make_demo,
                     normalize_pair, resize_image)


def test_resize_preserves_original_to_prepared_geometry():
    image = np.zeros((480, 1000, 3), np.uint8)
    result, scale = resize_image(image, 640)
    assert result.shape == (307, 640, 3)
    assert np.allclose(scale @ [500, 240, 1], [320, 153.5, 1])


def test_magsac_rejects_outliers_and_recovers_known_transform():
    rng = np.random.default_rng(9)
    p0 = rng.uniform([20, 20], [450, 330], (80, 2)).astype(np.float32)
    truth = np.array([[0.9, 0.02, 15], [0.01, 0.95, 8], [0.00005, 0, 1]])
    p1 = cv2.perspectiveTransform(p0[None], truth)[0]
    p1[-20:] = rng.uniform([0, 0], [480, 360], (20, 2))
    info, matrix, mask = geometric_check(p0, p1, (360, 480), (360, 480))
    assert info['geometry_pass']
    assert mask[:60].sum() >= 58
    assert mask[-20:].sum() <= 2
    assert corner_error(matrix, truth, (360, 480)) < 0.1


def test_many_clustered_matches_do_not_pass_coverage():
    rng = np.random.default_rng(3)
    p0 = rng.uniform([100, 100], [105, 105], (40, 2)).astype(np.float32)
    p1 = p0 + [10, 20]
    info, _, _ = geometric_check(p0, p1, (360, 480), (360, 480))
    assert not info['geometry_pass']
    assert info['reason'] == 'low_coverage'


def test_empty_matches_are_a_failure():
    info, matrix, mask = geometric_check(np.empty((0, 2)), np.empty((0, 2)), (360, 480), (360, 480))
    assert not info['geometry_pass'] and matrix is None and not len(mask)


def test_normalization_keeps_flat_images_unchanged():
    a = np.full((80, 80, 3), 30, np.uint8)
    b = np.full_like(a, 220)
    adjusted = normalize_pair(a, b)
    assert np.array_equal(a, adjusted[0]) and np.array_equal(b, adjusted[1])


def test_sift_demo_and_missing_input_remain_in_the_report(tmp_path):
    a, b, truth = make_demo()
    cv2.imwrite(str(tmp_path / 'a.png'), a)
    cv2.imwrite(str(tmp_path / 'b.png'), b)
    pairs = [dict(reference=str(tmp_path / 'a.png'), query=str(tmp_path / 'b.png'), H=truth.tolist()),
             dict(reference=str(tmp_path / 'missing.png'), query=str(tmp_path / 'b.png'))]
    rows = benchmark(pairs, output=tmp_path / 'run', repeats=1)
    assert len(rows) == 2
    assert rows[0]['geometry_pass'] and rows[0]['corner_error_px'] < 2
    assert rows[1]['status'] == 'error' and not rows[1]['geometry_pass']
    assert rows[1]['pair_ms'] is None
    report = json.loads((tmp_path / 'run' / 'run.json').read_text())
    assert report['models']['sift']['device'] == 'cpu'
    assert (tmp_path / 'run' / 'sift_000.png').is_file()
