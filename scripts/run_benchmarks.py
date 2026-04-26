"""Benchmark every algorithm across a fixed suite of voyages.

Output: a markdown table on stdout + a CSV at ``benchmarks/out/benchmark_<timestamp>.csv``.
Suitable for a thesis chapter or a paper appendix.

Three benchmark suites are run by default:

1. **BR-academic** — first 50 problems from BR1 (the canonical 3D-BPP test bed). Reefer-only
   items are unused; this is pure space-utilisation territory.
2. **Alex-realistic-presets** — 50 voyages of 30 items each from the Alexandria-mix preset
   strategy (deterministic — useful for diff'ing two runs).
3. **Alex-realistic-real** — 50 voyages from the Wadaboa-pool real strategy (closest
   approximation to operational data).

Run::

    PYTHONPATH=. python -m scripts.run_benchmarks --suites all
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.algorithms import get_algorithm
from app.algorithms.base import solve
from app.algorithms.ga import GeneticAlgorithm
from app.catalog.loader import get_container
from app.data.alexandria_sampler import AlexandriaSampler, SamplerConfig
from app.data.br_loader import (
    br_container_to_isolike,
    br_problem_to_items,
    list_br_problems,
)

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "benchmarks" / "out"


@dataclass
class _Row:
    suite: str
    algorithm: str
    voyage: int
    util: float
    weight_used: float
    cog_long: float
    cog_vert: float
    unstable: int
    imdg: int
    placed: int
    items: int
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------


def _br_suite(n: int = 50):
    problems = list_br_problems()[:n]
    suite = []
    for p in problems:
        suite.append((br_container_to_isolike(p), br_problem_to_items(p)))
    return "br-academic", suite


def _alex_presets_suite(n: int = 50):
    cont = get_container("40HC")
    suite = []
    for s in range(n):
        items = AlexandriaSampler(SamplerConfig(n_items=30, strategy="presets", seed=s)).sample()
        suite.append((cont, items))
    return "alex-presets", suite


def _alex_real_suite(n: int = 50):
    cont = get_container("40HC")
    suite = []
    for s in range(n):
        items = AlexandriaSampler(SamplerConfig(n_items=30, strategy="real", seed=s)).sample()
        suite.append((cont, items))
    return "alex-real", suite


SUITES = {
    "br": _br_suite,
    "alex-presets": _alex_presets_suite,
    "alex-real": _alex_real_suite,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


ALGORITHM_CODES = ["bl", "baf", "bssf", "blsf", "extreme_points", "ga"]


def _run_one(code: str, container, items, voyage_idx: int, suite: str) -> _Row:
    if code == "ga":
        algo = GeneticAlgorithm()
        algo.prepare(container, items)
    else:
        algo = get_algorithm(code)
    t0 = time.perf_counter()
    result, _ = solve(algorithm=algo, container=container, items=items)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    k = result.kpis
    return _Row(
        suite=suite,
        algorithm=code,
        voyage=voyage_idx,
        util=k.utilization,
        weight_used=k.weight_used,
        cog_long=k.cog_long_dev,
        cog_vert=k.cog_vert_frac,
        unstable=k.unstable_count,
        imdg=k.imdg_violation_count,
        placed=len(result.placements),
        items=len(items),
        elapsed_ms=elapsed_ms,
    )


def run_suite(suite_key: str, *, algorithms: list[str], n: int) -> list[_Row]:
    suite_name, suite = SUITES[suite_key](n=n)
    rows: list[_Row] = []
    for i, (cont, items) in enumerate(suite):
        for code in algorithms:
            rows.append(_run_one(code, cont, items, i, suite_name))
    return rows


def aggregate_and_print(rows: list[_Row]) -> None:
    """Print a markdown summary table grouped by (suite, algorithm)."""
    keyed: dict[tuple[str, str], list[_Row]] = {}
    for r in rows:
        keyed.setdefault((r.suite, r.algorithm), []).append(r)

    header = (
        "| suite | algorithm | mean util% | mean weight% | mean CoG long | mean CoG vert | "
        "mean placed/items | mean ms |"
    )
    sep = "|---|---|---:|---:|---:|---:|---|---:|"
    print(header)
    print(sep)
    for (suite, code), group in sorted(keyed.items()):
        mu = statistics.fmean(g.util for g in group) * 100
        mw = statistics.fmean(g.weight_used for g in group) * 100
        ml = statistics.fmean(g.cog_long for g in group)
        mv = statistics.fmean(g.cog_vert for g in group)
        mp = statistics.fmean(g.placed for g in group)
        mi = statistics.fmean(g.items for g in group)
        ms = statistics.fmean(g.elapsed_ms for g in group)
        print(
            f"| {suite} | {code} | {mu:.2f} | {mw:.2f} | {ml:+.3f} | {mv:.3f} | "
            f"{mp:.1f}/{mi:.1f} | {ms:.1f} |"
        )


def write_csv(rows: list[_Row]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"benchmark_{stamp}.csv"
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "suite", "algorithm", "voyage",
            "util", "weight_used", "cog_long", "cog_vert",
            "unstable", "imdg", "placed", "items", "elapsed_ms",
        ])
        for r in rows:
            writer.writerow([
                r.suite, r.algorithm, r.voyage,
                f"{r.util:.6f}", f"{r.weight_used:.6f}", f"{r.cog_long:.6f}", f"{r.cog_vert:.6f}",
                r.unstable, r.imdg, r.placed, r.items, f"{r.elapsed_ms:.3f}",
            ])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--suites",
        default="alex-presets,br",
        help="comma-separated suite keys: br,alex-presets,alex-real,all",
    )
    ap.add_argument("--algorithms", default=",".join(ALGORITHM_CODES))
    ap.add_argument("--n", type=int, default=20, help="voyages per suite")
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    suite_keys = list(SUITES) if args.suites == "all" else args.suites.split(",")
    algos = args.algorithms.split(",")

    rows: list[_Row] = []
    for k in suite_keys:
        if k not in SUITES:
            raise SystemExit(f"unknown suite {k!r}; pick from {list(SUITES)}")
        rows.extend(run_suite(k, algorithms=algos, n=args.n))

    aggregate_and_print(rows)
    if not args.no_csv:
        path = write_csv(rows)
        print(f"\nrows: {len(rows)} → {path}")


if __name__ == "__main__":
    main()
