"""Unit checks for the MCAP redaction pipeline. No GPU, no weights, no footage needed.

    python tests/test_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import redact_videos as rv  # noqa: E402
from plan_face_cuts import complement, merge  # noqa: E402


def test_smoother_holds_a_mask_across_frames():
    """A detection on one frame must stay masked across the whole window, or a face flickers back."""
    sm = rv.TemporalSmoother(5, "max")
    masks = [np.zeros((4, 4), np.uint8) for _ in range(10)]
    masks[5][1, 1] = 1
    out = []
    for i, m in enumerate(masks):
        out += sm.push(i, np.full((4, 4, 3), i, np.uint8), m)
    out += sm.flush()
    assert [o[0] for o in out] == list(range(10))
    assert all(o[1][0, 0, 0] == o[0] for o in out), "frame and mask fell out of step"
    assert [o[0] for o in out if o[2][1, 1]] == [3, 4, 5, 6, 7]
    print("smoother hold OK")


def test_dilate_grows_and_erodes():
    m = np.zeros((400, 400), np.uint8)
    m[190:210, 190:210] = 1
    assert rv.dilate_fast(m, 30).sum() > m.sum() * 4, "positive radius must grow the mask"
    assert rv.dilate_fast(m, -5).sum() < m.sum(), "negative radius must shrink it"
    assert rv.dilate_fast(m, 0).sum() == m.sum()
    assert rv.dilate_fast(np.zeros((40, 40), np.uint8), 20).sum() == 0
    print("dilate grow/erode OK")


def test_fill_covers_exactly_the_mask():
    frame = np.full((200, 200, 3), 200, np.uint8)
    mask = np.zeros((200, 200), np.uint8)
    mask[50:150, 50:150] = 1
    for mode in ("grey", "blur", "pixelate"):
        out = rv.apply_fill(frame, mask, mode, 128, 16)
        assert out.shape == frame.shape
        assert (out[0:40, 0:40] == 200).all(), f"{mode} leaked outside the mask"
    assert (rv.apply_fill(frame, mask, "grey", 128, 16)[60:140, 60:140] == 128).all()
    print("fill coverage OK")


def test_keyframe_detection():
    """Segment starts snap to IDR frames; getting this wrong makes clips that will not decode."""
    idr = b"\x00\x00\x00\x01\x09\xf0" + b"\x00\x00\x00\x01" + bytes([0x65]) + b"payload"
    inter = b"\x00\x00\x00\x01\x09\xf0" + b"\x00\x00\x00\x01" + bytes([0x41]) + b"payload"
    assert rv.is_keyframe(idr) is True
    assert rv.is_keyframe(inter) is False
    assert rv.is_keyframe(b"") is False
    print("keyframe detection OK")


def test_cut_planning():
    """Merging and the minimum clip length decide how many cuts you end up with."""
    faces = [[10.0, 11.0], [12.0, 13.0], [100.0, 101.0]]
    assert len(merge(faces, 0)) == 3
    assert len(merge(faces, 2)) == 2, "detections 1s apart should merge at gap 2"
    keep = complement(merge(faces, 2), 200.0, 30.0)
    assert keep == [[13.0, 100.0], [101.0, 200.0]], keep
    # a 30s minimum must discard anything shorter
    assert complement([[0.0, 10.0], [20.0, 30.0]], 200.0, 30.0) == [[30.0, 200.0]]
    print("cut planning OK")


def test_colour_rejection():
    """Opt-in glove filter: blue interiors are rejected, skin tones are not."""
    blue = np.zeros((100, 100, 3), np.uint8); blue[:, :] = (200, 60, 30)     # BGR blue
    skin = np.zeros((100, 100, 3), np.uint8); skin[:, :] = (90, 140, 200)    # BGR skin
    box = np.array([[10, 10, 90, 90, 0.9]], np.float32)
    assert len(rv.reject_coloured_boxes(blue, box, 0.35)) == 0
    assert len(rv.reject_coloured_boxes(skin, box, 0.35)) == 1
    print("colour rejection OK")


if __name__ == "__main__":
    test_smoother_holds_a_mask_across_frames()
    test_dilate_grows_and_erodes()
    test_fill_covers_exactly_the_mask()
    test_keyframe_detection()
    test_cut_planning()
    test_colour_rejection()
    print("ALL OK")
