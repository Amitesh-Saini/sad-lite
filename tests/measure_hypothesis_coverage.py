# =============================================================================
# SAD-Lite — Hypothesis Coverage Measurement
# =============================================================================
#
# PURPOSE
# Measure, without asserting, how investigate.py's hypotheses and confidence
# levels fall across the generated groups and several generator seeds.
#
# WHY THIS IS SEPARATE FROM THE TESTS
# Tests check specific observable patterns using controlled examples. If the
# tests instead required every labelled abusive account to reach some
# confidence, they would push the investigator towards reproducing the
# generator's answer key. This script reads labels only to group its output;
# the investigator never sees them.
#
# Run:
#     python tests/measure_hypothesis_coverage.py [--seeds 42 1 7] [--per-group 20]
# =============================================================================

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_users  # noqa: E402
import investigate  # noqa: E402


def measure(seed, per_group):
    """Investigate a sample of each group and count the strongest confidence per hypothesis."""

    sessions, labels, _ = generate_users.build_dataset(seed=seed) # Call Generate Users
    raw = pd.DataFrame(sessions).astype(str)
    labels = pd.DataFrame(labels)

    inputs = {"sessions": raw, "sessions_sha256": f"seed-{seed}", "features": None, "rules": None, "model": None, "provenance": []}
    counts = defaultdict(Counter)

    for pattern, group in labels.groupby("pattern", sort=True):
        for account in sorted(group["user_id"])[:per_group]: # Top 20 accounts after sort - Deterministic not random
            report = investigate.investigate(inputs, "user_id", account)

            for hypothesis in report["hypotheses"]:
                counts[(pattern, hypothesis["name"])][hypothesis["confidence"]] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(description="Measure hypothesis coverage across groups and seeds.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    parser.add_argument("--per-group", type=int, default=20)
    args = parser.parse_args()

    rows = []

    for seed in args.seeds:
        for (pattern, name), levels in measure(seed, args.per_group).items():
            rows.append({"seed": seed, "group": pattern, "hypothesis": name, **levels})

    table = pd.DataFrame(rows).fillna(0)
    levels = [level for level in ["Strong", "Moderate", "Weak", "Insufficient", "Not applicable"] if level in table.columns]
    table[levels] = table[levels].astype(int)

    print(table.sort_values(["group", "hypothesis", "seed"]).to_string(index=False))
    print(f"\nSampled up to {args.per_group} accounts per group. Labels group the output only; the investigator never reads them.")


if __name__ == "__main__":
    main()