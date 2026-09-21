#!/usr/bin/env python
"""Plan the fewest cuts that remove every face, keeping the longest clean runs.

Takes a detect_faces_mcap report and answers: where do I cut, and what am I left with? The objective
is not minimum footage removed but maximum usable run length - a dataset of long contiguous clips is
worth more than the same minutes chopped into fragments.

Two knobs drive it:
  --pad-s      widen each detection before cutting (safety margin around a face)
  --merge-s    detections closer together than this become ONE cut

Raising --merge-s trades a little extra footage for far fewer cuts, which is usually the right trade:
merging two cuts 3 s apart costs 3 s and removes a useless 3 s fragment from the output. The tool
sweeps --merge-s so you can see that curve before committing.

  python plan_face_cuts.py survey.json
  python plan_face_cuts.py survey.json --merge-s 30 --min-keep-s 60 --json plan.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def merge(intervals: list, gap: float) -> list:
    out: list = []
    for a, b in sorted(intervals):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def complement(drop: list, duration: float, min_keep: float) -> list:
    keep, cur = [], 0.0
    for a, b in drop:
        if a - cur >= min_keep:
            keep.append([cur, a])
        cur = max(cur, b)
    if duration - cur >= min_keep:
        keep.append([cur, duration])
    return keep


def load(report: Path) -> tuple:
    rep = json.loads(report.read_text())
    files = [f for f in rep["files"] if f.get("status") == "ok"]
    if not files:
        sys.exit("no successful files in the report")
    f = files[0]
    t0 = f["log_start_time"]
    raw = []
    for c in f["channels"]:
        off = c["start_log_time"] - t0
        raw += [[off + a, off + b] for a, b in c["intervals_rel_s"]]
    return f, raw, f["duration_s"]


def summarise(keep: list, drop: list, duration: float) -> dict:
    kept = sum(b - a for a, b in keep)
    return {
        "cuts": len(drop),
        "removed_s": round(sum(b - a for a, b in drop), 1),
        "kept_s": round(kept, 1),
        "kept_pct": round(100 * kept / duration, 1),
        "segments": len(keep),
        "longest_s": round(max((b - a for a, b in keep), default=0), 1),
        "median_s": round(sorted(b - a for a, b in keep)[len(keep) // 2], 1) if keep else 0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path, help="detect_faces_mcap JSON")
    ap.add_argument("--pad-s", type=float, default=1.0)
    ap.add_argument("--merge-s", type=float, default=None, help="if omitted, sweep a range and show the curve")
    ap.add_argument("--min-keep-s", type=float, default=30.0, help="discard kept segments shorter than this")
    ap.add_argument("--json", type=Path, default=None, help="write the chosen plan here")
    a = ap.parse_args(argv)

    f, raw, duration = load(a.report)
    padded = [[max(0.0, x - a.pad_s), y + a.pad_s] for x, y in raw]
    print(f"{Path(f['file']).name}")
    print(f"  {duration/60:.1f} min, {len(raw)} face intervals detected\n")

    if a.merge_s is None:
        print(f"  {'merge':>6} {'cuts':>5} {'removed':>9} {'kept':>9} {'segs':>5} {'longest':>9} {'median':>8}")
        for m in (0, 2, 5, 10, 20, 30, 60, 120, 300):
            d = merge(padded, m)
            k = complement(d, duration, a.min_keep_s)
            s = summarise(k, d, duration)
            print(f"  {m:>6} {s['cuts']:>5} {s['removed_s']:>8.0f}s {s['kept_s']:>8.0f}s "
                  f"{s['segments']:>5} {s['longest_s']:>8.0f}s {s['median_s']:>7.0f}s")
        print("\n  Bigger --merge-s means fewer, longer clean runs at the cost of a little more footage.")
        print("  Re-run with --merge-s N to write that plan.")
        return 0

    drop = merge(padded, a.merge_s)
    keep = complement(drop, duration, a.min_keep_s)
    s = summarise(keep, drop, duration)
    print(f"  merge {a.merge_s}s, pad {a.pad_s}s, min keep {a.min_keep_s}s -> {json.dumps(s)}\n")
    print("  KEEP (face-free):")
    for i, (x, y) in enumerate(keep):
        print(f"    {i:>3}  {x:8.1f} - {y:8.1f}s   ({y-x:6.1f}s)")
    print("  CUT:")
    for x, y in drop:
        print(f"         {x:8.1f} - {y:8.1f}s   ({y-x:6.1f}s)")
    if a.json:
        a.json.write_text(json.dumps({
            "source": f["file"], "duration_s": duration,
            "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
            "summary": s, "keep": [[round(x, 3), round(y, 3)] for x, y in keep],
            "cut": [[round(x, 3), round(y, 3)] for x, y in drop],
        }, indent=2) + "\n")
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
