# =============================================================================
# SAD-Lite — Detection Evaluation
# =============================================================================
#
# PURPOSE
# Compare the detector's predictions with the known synthetic account labels.
# Measure what it detected correctly, what it missed, and which innocent
# accounts it incorrectly recommended for review.
#
# INPUTS
# - data/rule_scores.csv: account scores, review recommendations, and evidence.
# - data/labels.csv: known labels and generated scenarios.
#   Label 0 = innocent; label 1 = synthetic abuse.
#
# WHY THIS IS NEEDED
# Producing suspiciousness scores does not establish that detection works.
# Evaluation measures both successful detections and mistakes.
# Accuracy alone can look high when most accounts are innocent, even if
# the detector catches little or no abuse.
#
# PROCESS
# 1. Validate inputs, reject duplicate IDs, and check account coverage.
# 2. Join predictions and labels using user_id, not row position.
# 3. Compare review_recommended with the known label for each account.
# 4. Count true positives, false positives, false negatives, and true negatives.
# 5. Calculate accuracy, precision, recall, F1, and false-positive rate.
# 6. Calculate precision/recall at K = 20, 40, and 120.
# 7. Calculate average precision using scores, treating equal scores together.
# 8. Report results by scenario, including innocent shared-network groups.
# 9. Save metrics, account-level outcomes, false positives, and false negatives.
#
# OUTCOME DEFINITIONS
# True positive:  abusive account recommended for review.
# False positive: innocent account recommended for review.
# False negative: abusive account not recommended for review.
# True negative:  innocent account not recommended for review.
#
# EVALUATION RULES
# - Threshold evaluation uses the saved review_recommended decisions.
# - Top-K evaluation uses the saved ranking and its deterministic tie-break.
# - Reject invalid K values rather than silently changing the review budget.
# - Handle zero denominators explicitly and document undefined metrics.
# - Keep evaluation labels out of feature extraction and scoring.
# - Do not automatically change thresholds based on evaluation results.
# - Preserve baseline results before comparing different configurations.
#
# LIMITATIONS
# Results describe this synthetic dataset, not proven real-world performance.
# If rules are tuned using these labels, this becomes development evaluation.
# Check frozen settings on another seed; even that does not establish
# generalisation to real users or entirely different abuse behaviours.
#
# This file evaluates detection. It does not execute enforcement actions.
# =============================================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from datetime import datetime, timezone


def validate_user_ids(data, file_name):
    """Check that every row has a unique, nonempty account ID."""

    if data.empty:
        raise ValueError(f"{file_name} is empty")

    if "user_id" not in data.columns:
        raise ValueError(f"{file_name} is missing user_id")

    if data["user_id"].isna().any():
        raise ValueError(f"{file_name} contains missing user IDs")

    if data["user_id"].str.strip().eq("").any():
        raise ValueError(f"{file_name} contains blank user IDs")

    if data["user_id"].ne(data["user_id"].str.strip()).any():
        raise ValueError(f"{file_name} contains user IDs with surrounding spaces")

    if data["user_id"].duplicated().any():
        raise ValueError(f"{file_name} contains duplicate user IDs")


def load_scores(path):
    """Load and validate the saved rule scores."""

    scores = pd.read_csv(path, dtype={"user_id": "string"})
    required = {"user_id", "rank", "rule_score", "review_recommended"}
    missing = required - set(scores.columns)

    if missing:
        raise ValueError(f"Score file is missing columns: {sorted(missing)}")

    validate_user_ids(scores, "Score file")

    if {"label", "pattern"} & set(scores.columns):
        raise ValueError("Score file must not contain ground-truth labels")

    for column in ["rank", "rule_score"]:
        scores[column] = pd.to_numeric(scores[column], errors="raise")

        if not np.isfinite(scores[column]).all():
            raise ValueError(f"{column} contains missing or infinite values")

    if scores["rule_score"].lt(0).any():
        raise ValueError("Rule scores must be nonnegative")

    # Ranks must be whole numbers from 1 to the number of accounts.
    if (scores["rank"] % 1 != 0).any():
        raise ValueError("Ranks must be whole numbers")

    if set(scores["rank"]) != set(range(1, len(scores) + 1)):
        raise ValueError("Ranks must be unique and consecutive, starting at 1")

    scores["rank"] = scores["rank"].astype(int)

    # Convert explicitly; astype(bool) would treat the string 'False' as True.
    decisions = scores["review_recommended"].astype("string").str.strip().str.lower()

    if decisions.isna().any() or not decisions.isin(["true", "false"]).all():
        raise ValueError("review_recommended must contain only True or False")

    scores["review_recommended"] = decisions.map({"true": True, "false": False}).astype(bool)
    scores = scores.sort_values("rank").reset_index(drop=True)

    if not scores["rule_score"].is_monotonic_decreasing:
        raise ValueError("Ranking does not follow descending rule scores")

    # Equal scores should have the same saved threshold decision.
    if scores.groupby("rule_score")["review_recommended"].nunique().gt(1).any():
        raise ValueError("Accounts with equal scores have inconsistent review decisions")

    decisions_by_score = scores.groupby("rule_score")["review_recommended"].first()

    if not decisions_by_score.is_monotonic_increasing:
        raise ValueError("Review decisions are inconsistent with a score threshold")

    return scores


def load_labels(path):
    """Load and validate the known synthetic account labels."""

    labels = pd.read_csv(path, dtype={"user_id": "string", "pattern": "string"})
    required = {"user_id", "label", "pattern"}
    missing = required - set(labels.columns)

    if missing:
        raise ValueError(f"Label file is missing columns: {sorted(missing)}")

    validate_user_ids(labels, "Label file")

    labels["label"] = pd.to_numeric(labels["label"], errors="raise")

    if not labels["label"].isin([0, 1]).all():
        raise ValueError("Labels must contain only 0 or 1")

    labels["label"] = labels["label"].astype(int)

    if labels["pattern"].isna().any() or labels["pattern"].str.strip().eq("").any():
        raise ValueError("Every account must have a scenario pattern")

    return labels[["user_id", "label", "pattern"]].copy()


def load_evaluation_data(scores_path, labels_path):
    """Load both files and match predictions with labels by account ID."""

    scores = load_scores(scores_path)
    labels = load_labels(labels_path)

    score_ids = set(scores["user_id"])
    label_ids = set(labels["user_id"])

    missing_predictions = label_ids - score_ids
    missing_labels = score_ids - label_ids

    if missing_predictions or missing_labels:
        raise ValueError(
            f"Account coverage mismatch: {len(missing_predictions)} accounts lack predictions; "
            f"{len(missing_labels)} accounts lack labels"
        )

    # One-to-one prevents duplicate IDs from multiplying rows during the join.
    data = scores.merge(labels, on="user_id", how="inner", validate="one_to_one")

    return data.sort_values("rank").reset_index(drop=True)


def count_outcomes(data):
    """Count correct and incorrect review decisions against known labels."""

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for account in data.itertuples(index=False):
        predicted = account.review_recommended
        actual = account.label

        if predicted and actual == 1:
            tp += 1
        elif predicted and actual == 0:
            fp += 1
        elif not predicted and actual == 1:
            fn += 1
        else:
            tn += 1

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}

def calculate_metrics(outcomes):
    """Calculate evaluation metrics from TP, FP, FN, and TN counts."""

    tp = outcomes["TP"]
    fp = outcomes["FP"]
    fn = outcomes["FN"]
    tn = outcomes["TN"]

    total = tp + fp + fn + tn

    # None means the metric is undefined, rather than a measured zero.
    accuracy = (tp + tn) / total if total > 0 else None
    precision = tp / (tp + fp) if tp + fp > 0 else None
    recall = tp / (tp + fn) if tp + fn > 0 else None
    f1 = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else None
    false_positive_rate = fp / (fp + tn) if fp + tn > 0 else None

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
    }

def evaluate_top_k(data, k):
    """Evaluate the first k accounts in the saved suspiciousness ranking."""

    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("k must be an integer")

    if k < 1 or k > len(data):
        raise ValueError(f"k must be between 1 and {len(data)}")

    # Use the saved ranking, including its tie-breaking order.
    top_accounts = data.sort_values("rank").head(k).copy()

    abusive_found = int(top_accounts["label"].sum())
    innocent_found = k - abusive_found
    total_abusive = int(data["label"].sum())

    precision = abusive_found / k
    recall = abusive_found / total_abusive if total_abusive > 0 else None

    return {
        "k": k,
        "abusive_found": abusive_found,
        "innocent_found": innocent_found,
        "precision": precision,
        "recall": recall,
        "accounts": top_accounts,
    }

def calculate_average_precision(data):
    """Measure how well scores rank abusive accounts above innocent ones."""

    if data.empty:
        raise ValueError("Cannot evaluate an empty dataset")

    # Without abusive accounts, average precision is undefined.
    if data["label"].sum() == 0:
        return None

    return float(average_precision_score(data["label"], data["rule_score"]))

def calculate_scenario_breakdowns(data):
    """Calculate detection results separately for each generated scenario."""

    if data.empty:
        raise ValueError("Cannot evaluate an empty dataset")

    rows = []

    for pattern, group in data.groupby("pattern", sort=True):
        outcomes = count_outcomes(group)
        metrics = calculate_metrics(outcomes)

        rows.append({
            "pattern": pattern,
            "account_count": len(group),
            "abusive_count": outcomes["TP"] + outcomes["FN"],
            "innocent_count": outcomes["FP"] + outcomes["TN"],
            "review_count": outcomes["TP"] + outcomes["FP"],
            "TP": outcomes["TP"],
            "FP": outcomes["FP"],
            "FN": outcomes["FN"],
            "TN": outcomes["TN"],
            **metrics,
        })

    return pd.DataFrame(rows)
def save_reports(data, outcomes, metrics, average_precision, top_k_results, scenario_breakdowns, output_dir):
    """Save one evaluation run in a new folder without overwriting past runs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    # JSON stores overall metrics. Undefined values become null.
    overall = {
        "account_count": len(data),
        "review_count": outcomes["TP"] + outcomes["FP"],
        "abusive_count": outcomes["TP"] + outcomes["FN"],
        "outcomes": {name: int(value) for name, value in outcomes.items()},
        "metrics": {
            name: float(value) if value is not None else None
            for name, value in metrics.items()
        },
        "average_precision": float(average_precision) if average_precision is not None else None,
        "evaluation_scope": "Synthetic dataset",
    }

    with (output_dir / "overall_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(overall, file, indent=2, allow_nan=False)

    # Keep the joined predictions and labels for tracing each decision.
    account_results = data.copy()
    account_results["outcome"] = np.select(
        [
            (data["label"] == 1) & data["review_recommended"],
            (data["label"] == 0) & data["review_recommended"],
            (data["label"] == 1) & (~data["review_recommended"]),
        ],
        ["TP", "FP", "FN"],
        default="TN",
    )

    account_results.to_csv(output_dir / "account_results.csv", index=False)

    missed = account_results[account_results["outcome"] == "FN"]
    false_alarms = account_results[account_results["outcome"] == "FP"]

    missed.to_csv(output_dir / "false_negatives.csv", index=False)
    false_alarms.to_csv(output_dir / "false_positives.csv", index=False)

    # Exclude the nested account tables from the top-K summary.
    top_k_rows = []

    for result in top_k_results:
        top_k_rows.append({
            "k": result["k"],
            "abusive_found": result["abusive_found"],
            "innocent_found": result["innocent_found"],
            "precision": result["precision"],
            "recall": result["recall"],
        })

    columns = ["k", "abusive_found", "innocent_found", "precision", "recall"]
    pd.DataFrame(top_k_rows, columns=columns).to_csv(output_dir / "top_k_metrics.csv", index=False)

    scenario_breakdowns.to_csv(output_dir / "scenario_breakdowns.csv", index=False, na_rep="")

    return output_dir

def main():
    """Load, evaluate, display, and save detection results."""

    root = Path(__file__).resolve().parent
    data_dir = root / "data"

    data = load_evaluation_data(data_dir / "rule_scores.csv", data_dir / "labels.csv")

    outcomes = count_outcomes(data)
    metrics = calculate_metrics(outcomes)
    average_precision = calculate_average_precision(data)
    scenario_breakdowns = calculate_scenario_breakdowns(data)

    # Keep each result so it can also be saved.
    top_k_results = []

    for k in [10, 20, 40, 60, 120]:
        top_k_results.append(evaluate_top_k(data, k))

    print(f"Accounts evaluated: {len(data)}")
    print(f"Accounts recommended for review: {outcomes['TP'] + outcomes['FP']}")
    print(f"Accounts labelled abusive: {outcomes['TP'] + outcomes['FN']}")

    print("\nOutcome counts:")
    print(f"True positives:  {outcomes['TP']}")
    print(f"False positives: {outcomes['FP']}")
    print(f"False negatives: {outcomes['FN']}")
    print(f"True negatives:  {outcomes['TN']}")

    print("\nPerformance metrics:")

    for name, value in metrics.items():
        display_name = name.replace("_", " ").capitalize()

        if value is None:
            print(f"{display_name}: undefined")
        else:
            print(f"{display_name}: {value:.2%}")

    if average_precision is None:
        print("Average precision: undefined")
    else:
        print(f"Average precision: {average_precision:.2%}")

    # Display accounts involved in incorrect decisions.
    missed = data[(data["label"] == 1) & (~data["review_recommended"])]
    false_alarms = data[(data["label"] == 0) & data["review_recommended"]]
    columns = ["user_id", "pattern", "rule_score", "triggered_rules", "evidence_summary"]

    print("\nMissed abusive accounts:")

    if missed.empty:
        print("None")
    else:
        print(missed[columns].fillna({"triggered_rules": "None"}).to_string(index=False))

    print("\nInnocent accounts incorrectly flagged:")

    if false_alarms.empty:
        print("None")
    else:
        print(false_alarms[columns].fillna({"triggered_rules": "None"}).to_string(index=False))

    print("\nTop-K evaluation:")

    for result in top_k_results:
        k = result["k"]

        print(f"\nTop {k} accounts:")
        print(f"Abusive accounts found: {result['abusive_found']}")
        print(f"Innocent accounts included: {result['innocent_found']}")
        print(f"Precision@{k}: {result['precision']:.2%}")

        if result["recall"] is None:
            print(f"Recall@{k}: undefined")
        else:
            print(f"Recall@{k}: {result['recall']:.2%}")

    print("\nScenario breakdowns:")
    print(scenario_breakdowns.to_string(index=False, na_rep="undefined", float_format=lambda value: f"{value:.4f}"))
    print("Scenario metric values are fractions: 0.7500 means 75%.")

    # Use UTC timestamps with microseconds to distinguish evaluation runs.
    run_name = datetime.now(timezone.utc).strftime("rules_%Y%m%d_%H%M%S_%f_UTC")
    output_dir = root / "reports" / run_name

    saved_dir = save_reports(data, outcomes, metrics, average_precision, top_k_results, scenario_breakdowns, output_dir)

    print(f"\nReports saved to: {saved_dir}")
    print("Evaluation on synthetic data.")


if __name__ == "__main__":
    main()