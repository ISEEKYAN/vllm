# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise C3 comparison. Usage: python c3_compare.py <run_dir> [<other_run_dir>]

One dir: every scenario vs graph/<family>_single (families: greedy, full, sample).
Two dirs: same config/scenario across runs. Non-finite values fail.
"""

import json
import os
import sys

REQUIRED = ("graph", "graph_cache", "eager", "mixed")


def load(d):
    out = {}
    for name in sorted(os.listdir(d)):
        if name.startswith("c3v2-") and name.endswith(".json"):
            with open(os.path.join(d, name)) as f:
                r = json.load(f)
            out[r["config"]] = r
    return out


def family(scenario):
    for f in ("full", "sample"):
        if scenario.startswith(f):
            return f
    return "greedy"


def reference(fam):
    return {"greedy": "single", "full": "full_single", "sample": "sample_single"}[fam]


def diff(a, b):
    bad = total = 0
    assert len(a) == len(b), (len(a), len(b))
    for ra, rb in zip(a, b):
        assert len(ra) == len(rb), (len(ra), len(rb))
        for sa, sb in zip(ra, rb):
            total += 1
            values = [sa["lp"], sb["lp"]]
            if family_of_step(sa) != "full":
                values += [v for _, v in sa["top"] + sb["top"]]
            if (
                any(v in ("nan", "inf", "-inf") for v in values)
                or sa.get("nonfinite", 0)
                or sb.get("nonfinite", 0)
                or sa["tok"] != sb["tok"]
                or sa["lp"] != sb["lp"]
                or sa["top"] != sb["top"]
            ):
                bad += 1
    return bad, total


def family_of_step(step):
    return "full" if "nonfinite" in step else "other"


def main():
    runs = [load(d) for d in sys.argv[1:]]
    rows = []
    if len(runs) == 1:
        run = runs[0]
        for cfg, r in run.items():
            for sc, val in r["scenarios"].items():
                ref_sc = reference(family(sc))
                if cfg == "graph" and sc == ref_sc:
                    continue
                if sc == "pd_prefill":
                    ref_sc = "single"
                rows.append(
                    (f"{cfg}/{sc}", *diff(run["graph"]["scenarios"][ref_sc], val))
                )
    else:
        a, b = runs
        for cfg in sorted(set(a) & set(b)):
            for sc in sorted(set(a[cfg]["scenarios"]) & set(b[cfg]["scenarios"])):
                rows.append(
                    (
                        f"{cfg}/{sc}",
                        *diff(a[cfg]["scenarios"][sc], b[cfg]["scenarios"][sc]),
                    )
                )
    missing = [c for run in runs for c in REQUIRED if c not in run]
    ok = not missing and all(bad == 0 and total > 0 for _, bad, total in rows)
    for c in missing:
        print(f"MISSING config {c}")
    for name, bad, total in rows:
        print(f"{name:28s} diff={bad}/{total}")
    print("C3_ALL_ZERO_DIFF" if ok else "C3_DIFF_FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
