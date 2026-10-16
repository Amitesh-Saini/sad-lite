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


def load_scores(path, score_column="rule_score"):
    """Load and validate saved rule scores or model anomaly scores."""

    if score_column not in ["rule_score", "anomaly_score"]:
        raise ValueError("score_column must be 'rule_score' or 'anomaly_score'")

    scores = pd.read_csv(path, dtype={"user_id": "string"})
    required = {"user_id", "rank", score_column}

    # Only the rules currently have a saved review threshold.
    if score_column == "rule_score":
        required.add("review_recommended")

    missing = required - set(scores.columns)

    if missing:
        raise ValueError(f"Score file is missing columns: {sorted(missing)}")

    validate_user_ids(scores, "Score file")

    if {"label", "pattern"} & set(scores.columns):
        raise ValueError("Score file must not contain ground-truth labels")

    for column in ["rank", score_column]:
        scores[column] = pd.to_numeric(scores[column], errors="raise")

        if not np.isfinite(scores[column]).all():
            raise ValueError(f"{column} contains missing or infinite values")

    # Both scoring methods currently produce nonnegative scores.
    if scores[score_column].lt(0).any():
        raise ValueError(f"{score_column} must be nonnegative")

    # Ranks must be whole numbers from 1 to the number of accounts.
    if (scores["rank"] % 1 != 0).any():
        raise ValueError("Ranks must be whole numbers")

    if set(scores["rank"]) != set(range(1, len(scores) + 1)):
        raise ValueError("Ranks must be unique and consecutive, starting at 1")

    scores["rank"] = scores["rank"].astype(int)
    scores = scores.sort_values("rank").reset_index(drop=True)

    if not scores[score_column].is_monotonic_decreasing:
        raise ValueError(f"Ranking does not follow descending {score_column}")

    # Validate threshold decisions only for the rules.
    if score_column == "rule_score":
        decisions = scores["review_recommended"].astype("string").str.strip().str.lower()

        if decisions.isna().any() or not decisions.isin(["true", "false"]).all():
            raise ValueError("review_recommended must contain only True or False")

        scores["review_recommended"] = decisions.map({"true": True, "false": False}).astype(bool)

        if scores.groupby(score_column)["review_recommended"].nunique().gt(1).any():
            raise ValueError("Accounts with equal scores have inconsistent review decisions")

        decisions_by_score = scores.groupby(score_column)["review_recommended"].first()

        if not decisions_by_score.is_monotonic_increasing:
            raise ValueError("Review decisions are inconsistent with a score threshold")

    elif "review_recommended" in scores.columns:
        raise ValueError("Model scores should contain rankings only; no review cutoff has been defined")

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


def load_evaluation_data(scores_path, labels_path, score_column="rule_score"):
    """Load scores and join each account with its ground-truth label."""

    scores = load_scores(scores_path, score_column=score_column)
    labels = load_labels(labels_path)

    # Both files must describe exactly the same accounts.
    score_ids = set(scores["user_id"])
    label_ids = set(labels["user_id"])

    missing_labels = score_ids - label_ids
    missing_scores = label_ids - score_ids

    if missing_labels:
        raise ValueError(f"Accounts missing labels: {sorted(missing_labels)}")

    if missing_scores:
        raise ValueError(f"Accounts missing scores: {sorted(missing_scores)}")

    # Join by user ID, not row position.
    data = scores.merge(labels, on="user_id", how="inner", validate="one_to_one")
    data = data.sort_values("rank").reset_index(drop=True)

    return data


def count_outcomes(data):
    """Count correct and incorrect review decisions against known labels."""

    if "review_recommended" not in data.columns:
        raise ValueError("Review decisions must be defined before counting outcomes.")

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

def evaluate_at_budget(data, k):
    """Evaluate decisions where exactly the top k accounts receive review."""

    top_k_result = evaluate_top_k(data, k)

    # Work on a copy so the original rules threshold decisions stay unchanged.
    decisions = data.copy()
    selected_ids = top_k_result["accounts"]["user_id"]

    decisions["review_recommended"] = decisions["user_id"].isin(selected_ids)

    outcomes = count_outcomes(decisions)
    metrics = calculate_metrics(outcomes)

    return {
        "k": k,
        "outcomes": outcomes,
        "metrics": metrics,
        "accounts": decisions,
    }

def evaluate_top_k(data, k):
    """Evaluate the top k accounts from either the rules or model ranking."""

    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("k must be an integer")

    if k < 1 or k > len(data):
        raise ValueError(f"k must be between 1 and {len(data)}")

    # Use the saved ranking, including its tie-breaking order.
    top_accounts = data.sort_values("rank").head(k).copy()

    abusive_found = int(top_accounts["label"].sum())
    innocent_found = k - abusive_found
    total_abusive = int(data["label"].sum())

    # Precision: how many reviewed accounts are abusive?
    precision = abusive_found / k

    # Recall: how much of the dataset's abuse did this review budget find?
    recall = abusive_found / total_abusive if total_abusive > 0 else None

    return {
        "k": k,
        "abusive_found": abusive_found,
        "innocent_found": innocent_found,
        "precision": precision,
        "recall": recall,
        "accounts": top_accounts,
    }
def calculate_average_precision(data, score_column="rule_score"):
    """Measure ranking quality using either rule scores or anomaly scores."""

    if data.empty:
        raise ValueError("Cannot evaluate an empty account table.")

    if score_column not in data.columns:
        raise ValueError(f"Missing score column: {score_column}")

    if data["label"].sum() == 0:
        return None

    # Use scores directly so tied scores are evaluated together.
    return float(average_precision_score(data["label"], data[score_column]))

def calculate_scenario_breakdowns(data, k=None):
    """Evaluate review decisions separately for each scenario."""

    if data.empty:
        raise ValueError("Cannot evaluate an empty account table.")

    if "pattern" not in data.columns:
        raise ValueError("Evaluation data must contain scenario names.")

    if data["pattern"].isna().any():
        raise ValueError("Scenario names must not be missing.")

    # Select the top k across all accounts before grouping by scenario.
    if k is not None:
        evaluation = evaluate_at_budget(data, k)
        decisions = evaluation["accounts"]
    else:
        if "review_recommended" not in data.columns:
            raise ValueError("Provide k when saved review decisions are unavailable.")

        decisions = data.copy()

    rows = []

    for pattern, accounts in decisions.groupby("pattern", sort=True):
        outcomes = count_outcomes(accounts)
        metrics = calculate_metrics(outcomes)

        row = {
            "pattern": pattern,
            "account_count": len(accounts),
            "abusive_count": int(accounts["label"].eq(1).sum()),
            "innocent_count": int(accounts["label"].eq(0).sum()),
            "review_count": int(accounts["review_recommended"].sum()),
            **outcomes,
            **metrics,
        }

        rows.append(row)

    return pd.DataFrame(rows)

def save_reports(results, comparison, overlap, output_dir):
    """Save rules and model evaluations in a new report folder."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    comparison.to_csv(output_dir / "budget_comparison.csv", index=False)
    overlap.to_csv(output_dir / "ranking_overlap.csv", index=False)

    summary = {
        "evaluation_scope": "Synthetic data; Isolation Forest fitted and scored on the same accounts.",
        "methods": {},
    }

    for method, result in results.items():
        method_dir = output_dir / method
        method_dir.mkdir()

        data = result["data"]
        data.to_csv(method_dir / "ranked_accounts.csv", index=False)

        summary["methods"][method] = {
            "account_count": len(data),
            "abusive_count": int(data["label"].sum()),
            "average_precision": result["average_precision"],
        }

        # Save decisions and scenario results for each review budget.
        for k, evaluation in result["budgets"].items():
            budget_dir = method_dir / f"top_{k}"
            budget_dir.mkdir()

            accounts = evaluation["accounts"].copy()
            selected = accounts["review_recommended"]
            abusive = accounts["label"].eq(1)

            accounts["outcome"] = np.select(
                [selected & abusive, selected & ~abusive, ~selected & abusive],
                ["TP", "FP", "FN"],
                default="TN",
            )

            accounts.to_csv(budget_dir / "account_results.csv", index=False)
            accounts[selected].to_csv(budget_dir / "selected_accounts.csv", index=False)
            accounts[accounts["outcome"].eq("FP")].to_csv(budget_dir / "false_positives.csv", index=False)
            accounts[accounts["outcome"].eq("FN")].to_csv(budget_dir / "false_negatives.csv", index=False)

            result["scenarios"][k].to_csv(budget_dir / "scenario_breakdowns.csv", index=False)

    # Preserve the original rules evaluation at its saved score threshold.
    rule_data = results["rules"]["data"]
    outcomes = count_outcomes(rule_data)
    metrics = calculate_metrics(outcomes)

    summary["rules_saved_threshold"] = {
        "review_count": outcomes["TP"] + outcomes["FP"],
        "outcomes": outcomes,
        "metrics": metrics,
    }

    threshold_dir = output_dir / "rules" / "saved_threshold"
    threshold_dir.mkdir()

    missed = rule_data["label"].eq(1) & ~rule_data["review_recommended"]
    false_alarms = rule_data["label"].eq(0) & rule_data["review_recommended"]

    rule_data[missed].to_csv(threshold_dir / "false_negatives.csv", index=False)
    rule_data[false_alarms].to_csv(threshold_dir / "false_positives.csv", index=False)
    calculate_scenario_breakdowns(rule_data).to_csv(threshold_dir / "scenario_breakdowns.csv", index=False)

    # Save accounts selected by only one method at each budget.
    for k in results["rules"]["budgets"]:
        rule_accounts = results["rules"]["budgets"][k]["accounts"]
        model_accounts = results["model"]["budgets"][k]["accounts"]

        rule_ids = set(rule_accounts.loc[rule_accounts["review_recommended"], "user_id"])
        model_ids = set(model_accounts.loc[model_accounts["review_recommended"], "user_id"])

        rule_accounts[rule_accounts["user_id"].isin(rule_ids - model_ids)].to_csv(output_dir / f"rules_only_top_{k}.csv", index=False)
        model_accounts[model_accounts["user_id"].isin(model_ids - rule_ids)].to_csv(output_dir / f"model_only_top_{k}.csv", index=False)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, allow_nan=False)

    return output_dir

def main():
    """Evaluate rules and Isolation Forest using the same labelled accounts."""

    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"

    rule_data = load_evaluation_data(data_dir / "rule_scores.csv", data_dir / "labels.csv")
    model_data = load_evaluation_data(data_dir / "model_scores.csv", data_dir / "labels.csv", score_column="anomaly_score")

    if set(rule_data["user_id"]) != set(model_data["user_id"]):
        raise ValueError("Rules and model results must contain the same accounts.")

    budgets = [10, 20, 40, 60, 120]

    if max(budgets) > len(rule_data):
        raise ValueError("The dataset contains fewer accounts than the largest review budget.")

    results = {}
    comparison_rows = []
    overlap_rows = []

    # Evaluate both methods using identical review budgets.
    for method, data, score_column in [
        ("rules", rule_data, "rule_score"),
        ("model", model_data, "anomaly_score"),
    ]:
        results[method] = {
            "data": data,
            "average_precision": calculate_average_precision(data, score_column=score_column),
            "budgets": {},
            "scenarios": {},
        }

        for k in budgets:
            evaluation = evaluate_at_budget(data, k)
            results[method]["budgets"][k] = evaluation
            results[method]["scenarios"][k] = calculate_scenario_breakdowns(data, k=k)

            comparison_rows.append({
                "method": method,
                "k": k,
                **evaluation["outcomes"],
                **evaluation["metrics"],
            })

    # Compare which accounts appear in both review lists.
    for k in budgets:
        rule_ids = set(evaluate_top_k(rule_data, k)["accounts"]["user_id"])
        model_ids = set(evaluate_top_k(model_data, k)["accounts"]["user_id"])

        shared_ids = rule_ids & model_ids
        combined_ids = rule_ids | model_ids

        overlap_rows.append({
            "k": k,
            "shared_accounts": len(shared_ids),
            "rules_only": len(rule_ids - model_ids),
            "model_only": len(model_ids - rule_ids),
            "overlap_fraction": len(shared_ids) / k,
            "jaccard_similarity": len(shared_ids) / len(combined_ids),
        })

    comparison = pd.DataFrame(comparison_rows).sort_values(["k", "method"]).reset_index(drop=True)
    overlap = pd.DataFrame(overlap_rows)

    print(f"Accounts evaluated per method: {len(rule_data)}")
    print(f"Accounts labelled abusive: {int(rule_data['label'].sum())}")

    # Keep the original rules threshold results separate from top-k results.
    outcomes = count_outcomes(rule_data)
    metrics = calculate_metrics(outcomes)

    print("\nRules at the saved review threshold:")
    print(f"Accounts recommended for review: {outcomes['TP'] + outcomes['FP']}")

    for name, value in outcomes.items():
        print(f"{name}: {value}")

    for name, value in metrics.items():
        display_name = name.replace("_", " ").capitalize()
        display_value = "undefined" if value is None else f"{value:.2%}"
        print(f"{display_name}: {display_value}")

    print("\nAverage precision:")

    for method, result in results.items():
        value = result["average_precision"]
        display_value = "undefined" if value is None else f"{value:.2%}"
        print(f"{method.capitalize()}: {display_value}")

    # Format a display copy without changing the saved numeric results.
    display = comparison.copy()

    for column in ["accuracy", "precision", "recall", "f1", "false_positive_rate"]:
        display[column] = display[column].map(lambda value: "undefined" if pd.isna(value) else f"{value:.2%}")

    print("\nComparison at equal review budgets:")
    print(display.to_string(index=False))

    print("\nRanking overlap:")
    print(overlap.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\nScenario breakdowns at a review budget of 40:")

    for method, result in results.items():
        print(f"\n{method.capitalize()}:")
        print(result["scenarios"][40].to_string(index=False, na_rep="undefined", float_format=lambda value: f"{value:.4f}"))

    print("\nScenario metric values are fractions: 0.7500 means 75%.")

    run_name = datetime.now(timezone.utc).strftime("comparison_%Y%m%d_%H%M%S_%f_UTC")
    output_dir = save_reports(results, comparison, overlap, project_dir / "reports" / run_name)

    print(f"\nReports saved to: {output_dir}")
    print("Evaluation on synthetic data; the model was fitted and scored on the same accounts.")


if __name__ == "__main__":
    main()