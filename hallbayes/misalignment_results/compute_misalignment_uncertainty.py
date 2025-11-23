#!/usr/bin/env python3
"""
Compute empirical misalignment rate and a 95% Wilson confidence interval
from a CSV exported by your misalignment toolkit.

Expected CSV columns:
    idx,prompt,decision_answer,roh_bound,isr,delta_bar,q_avg,q_conservative,
    rationale,meta_json,answer,aligned_label,decision_correct,notes

Usage:
    python compute_misalignment_uncertainty.py path/to/results.csv
"""

import sys
import math
import csv
from typing import Tuple, List, Dict


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion p = k/n.

    Returns (lower, upper) bounds at (1 - alpha) confidence.
    Use the upper bound as a conservative estimate of the rate.

    If n == 0, returns (0.0, 1.0) (uninformative).
    """
    if n == 0:
        return 0.0, 1.0

    # 95% confidence by default
    if abs(alpha - 0.05) < 1e-9:
        z = 1.96
    else:
        # generic z from alpha if you want a different confidence
        from math import sqrt, erfcinv
        z = math.sqrt(2) * erfcinv(alpha)

    p_hat = k / n
    denom = 1.0 + (z ** 2) / n
    centre = (p_hat + (z ** 2) / (2 * n)) / denom
    halfwidth = (z / denom) * math.sqrt(
        (p_hat * (1.0 - p_hat) / n) + (z ** 2) / (4.0 * n * n)
    )

    lower = max(0.0, centre - halfwidth)
    upper = min(1.0, centre + halfwidth)
    return lower, upper


def parse_bool(value: str) -> bool:
    """
    Robust-ish string -> bool conversion for your CSV fields.

    Interprets variations like:
        "True", "true", "1", "yes", "aligned"
    and
        "False", "false", "0", "no", "misaligned"
    """
    if value is None:
        return False
    v = str(value).strip().lower()

    # positive / aligned
    if v in {"1", "true", "t", "yes", "y", "aligned", "correct"}:
        return True
    # negative / misaligned
    if v in {"0", "false", "f", "no", "n", "misaligned", "incorrect"}:
        return False

    # default: be conservative and treat unknown as misaligned
    return False


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    if len(sys.argv) != 2:
        print("Usage: python compute_misalignment_uncertainty.py path/to/results.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_rows(csv_path)

    if not rows:
        print("No rows in CSV.")
        sys.exit(0)

    total = len(rows)

    # Filter to items where the gate decided to answer
    answered_rows = [
        r for r in rows
        if parse_bool(r.get("decision_answer", "False"))
    ]

    n_answered = len(answered_rows)
    n_abstained = total - n_answered

    # Among answered, count misaligned outputs
    # aligned_label == True  -> aligned
    # aligned_label == False -> misaligned
    k_misaligned = 0
    for r in answered_rows:
        aligned = parse_bool(r.get("aligned_label", "False"))
        if not aligned:
            k_misaligned += 1

    # Empirical misalignment rate
    if n_answered > 0:
        m_hat = k_misaligned / n_answered
    else:
        m_hat = 0.0

    # Wilson 95% CI
    m_lo, m_hi = wilson_interval(k_misaligned, n_answered, alpha=0.05)

    # Optionally summarise roh_bound distribution for answered items
    roh_bounds = []
    for r in answered_rows:
        v = r.get("roh_bound", "")
        try:
            roh_bounds.append(float(v))
        except (TypeError, ValueError):
            pass

    roh_summary = ""
    if roh_bounds:
        roh_sorted = sorted(roh_bounds)
        def pct(p: float) -> float:
            idx = int(p * (len(roh_sorted) - 1))
            return roh_sorted[idx]
        roh_summary = (
            f"RoM bound summary (answered items): "
            f"median={pct(0.5):.4g}, p95={pct(0.95):.4g}, max={roh_sorted[-1]:.4g}"
        )

    # Print results
    print(f"Total items in CSV:               {total}")
    print(f"Answered items (decision_answer): {n_answered}")
    print(f"Abstained items:                  {n_abstained}")
    print()
    print(f"Misaligned among answered:        {k_misaligned}")
    print(f"Empirical misalignment rate m̂:   {m_hat:.6f}")
    print(f"95% Wilson CI for m:             [{m_lo:.6f}, {m_hi:.6f}]")
    print(f"Conservative upper bound m_high:  {m_hi:.6f}")
    if roh_summary:
        print()
        print(roh_summary)


if __name__ == "__main__":
    main()
