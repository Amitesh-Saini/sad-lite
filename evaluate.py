# =============================================================================
# SAD-Lite — Detection Evaluation
# =============================================================================
#
# PURPOSE
# Compare the detectors' rankings with the known synthetic account labels.
# Measure what they detected correctly, what they missed, and which innocent
# accounts they incorrectly put in front of a reviewer.
#
# INPUTS
# Published stages, each verified against its manifest, and the chain between
# them checked: features from these sessions, scores from these features
# (see provenance.py for what this does and does not guarantee).
# - data/rules/rule_scores.csv: rule scores, review recommendations, and evidence.
# - data/model/model_scores.csv: Isolation Forest anomaly scores.
# - data/truth/labels.csv: known labels and generated scenarios.
#   Label 0 = innocent; label 1 = synthetic abuse.
# - data/truth/incident_truth.json: the game incident and its participants.
# - config/rules.yaml: the review capacity (K).
# - data/features/features.csv: only to re-score for the experiment and seed checks.
#
# WHY THIS IS NEEDED
# Producing suspiciousness scores does not establish that detection works.
# Evaluation measures both successful detections and mistakes.
# Accuracy is not reported: with about 6% of accounts abusive, flagging
# nobody is about 94% accurate. That figure is printed once, as the reason.
#
# PROCESS
# 1. Validate inputs, reject duplicate IDs, and check account coverage.
# 2. Join predictions and labels using user_id, not row position.
# 3. Evaluate both rankings at the review capacity K and at larger budgets.
# 4. Calculate average precision using scores, treating equal scores together.
# 5. Report which scenarios each detector catches, and their rank ranges.
# 6. Report innocent lookalikes (the dormitory) flagged at each budget.
# 7. Check that game-incident participants stay out of both rankings' tops.
# 8. Re-score the rules with the raised shared-IP weight (the experiment).
# 9. Re-fit the model across several forest seeds to measure stability.
# 10. Re-measure precision at a realistic base rate by subsampling abusers.
# 11. Save tables, account-level outcomes, and a markdown summary.
#
# OUTCOME DEFINITIONS
# True positive:  abusive account selected for review.
# False positive: innocent account selected for review.
# False negative: abusive account not selected for review.
# True negative:  innocent account not selected for review.
#
# EVALUATION RULES
# - K is the review capacity: how many accounts one reviewer can examine.
# - Top-K evaluation uses the saved ranking and its deterministic tie-break.
# - Threshold evaluation uses the rules' saved review_recommended decisions.
# - Reject invalid K values rather than silently changing the review budget.
# - Handle zero denominators explicitly and document undefined metrics.
# - Keep evaluation labels out of feature extraction and scoring.
# - Do not automatically change thresholds based on evaluation results.
# - Average precision uses scikit-learn's definition, which treats tied
#   scores as one group instead of rewarding a lucky tie-break order.
#
# LIMITATIONS
# Results describe this synthetic dataset, not proven real-world performance.
# If rules are tuned using these labels, this becomes development evaluation.
# The model is fitted and scored on the same accounts.
# The base rate here is far above any real platform; the realistic-base-rate
# check shows how much that flatters precision.
#
# This file evaluates detection. It does not execute enforcement actions.
# =============================================================================

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score
from datetime import datetime, timezone

import provenance
import score_model
import score_rules


# Review budgets reported beyond the review capacity.
BUDGETS = [20, 40, 80, 120, 200, 300, 400]

# Budgets for the innocent-lookalike table.
LOOKALIKE_BUDGETS = [40, 120, 200, 300]

DORM_PATTERN = "dorm_shared_ip"


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

    return validate_scores(scores, score_column)


def validate_scores(scores, score_column="rule_score"):
    """Validate a ranking table, whether loaded from disk or freshly scored."""

    scores = scores.copy()
    required = {"user_id", "rank", score_column}

    # Only the rules currently have a saved review threshold.
    if score_column == "rule_score":
        required.add("review_recommended")

    missing = required - set(scores.columns)

    if missing:
        raise ValueError(f"Score file is missing columns: {sorted(missing)}")

    scores["user_id"] = scores["user_id"].astype("string")
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


def join_labels(scores, labels):
    """Join each account's score with its ground-truth label."""

    # Both tables must describe exactly the same accounts.
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


def load_evaluation_data(scores_path, labels_path, score_column="rule_score"):
    """Load scores and join each account with its ground-truth label."""

    scores = load_scores(scores_path, score_column=score_column)
    labels = load_labels(labels_path)

    return join_labels(scores, labels)


def load_review_capacity(config_path):
    """Read K, the review capacity, from the rules configuration."""

    with Path(config_path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    score_rules.validate_config(config)

    return int(config["review"]["capacity"])


def load_verified_stages(data_dir):
    """Verify every stage evaluation reads, and the chain between them.

    Returns the verified bytes of each file. Raises provenance.ProvenanceError
    if any stage is missing, incomplete, altered, or built from a different run.
    """

    generate, generate_files = provenance.load_stage(data_dir, "generate", ["sessions.csv"])
    features, features_files = provenance.load_stage(data_dir, "features", ["features.csv"])
    rules, rules_files = provenance.load_stage(data_dir, "rules", ["rule_scores.csv", "rules_config.yaml"])
    model, model_files = provenance.load_stage(data_dir, "model", ["model_scores.csv", "model_settings.json"])
    truth, truth_files = provenance.load_stage(data_dir, "truth", ["labels.csv", "incident_truth.json"])

    sessions_digest = provenance.sha256_bytes(generate_files["sessions.csv"])
    features_digest = provenance.sha256_bytes(features_files["features.csv"])

    provenance.check_input(features, "sessions", "generate", "sessions.csv", sessions_digest)
    provenance.check_input(rules, "features", "features", "features.csv", features_digest)
    provenance.check_input(model, "features", "features", "features.csv", features_digest)

    # The answer key must describe exactly these sessions.
    provenance.check_input(truth, "sessions", "generate", "sessions.csv", sessions_digest)

    return {
        "features": features_files["features.csv"],
        "rule_scores": rules_files["rule_scores.csv"],
        "model_scores": model_files["model_scores.csv"],
        "labels": truth_files["labels.csv"],
        "incident": truth_files["incident_truth.json"],
    }


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

    # None means the metric is undefined, rather than a measured zero.
    # Accuracy is deliberately not reported; see the file header.
    precision = tp / (tp + fp) if tp + fp > 0 else None
    recall = tp / (tp + fn) if tp + fn > 0 else None
    f1 = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else None
    false_positive_rate = fp / (fp + tn) if fp + tn > 0 else None

    return {
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


def caught_by_pattern(results, k):
    """Show, for each scenario, how many accounts each method puts in its top k."""

    rows = []
    patterns = sorted(results["rules"]["data"]["pattern"].unique())

    for pattern in patterns:
        row = {"pattern": pattern}

        for method, result in results.items():
            data = result["data"]
            total = int(data["pattern"].eq(pattern).sum())
            found = int(data.head(k)["pattern"].eq(pattern).sum())
            row[f"{method}_top_{k}"] = f"{found}/{total}"

        rows.append(row)

    return pd.DataFrame(rows)


def rank_ranges(results):
    """Report the best, median, and worst rank of each abusive scenario."""

    rows = []

    for method, result in results.items():
        data = result["data"]
        abusive = data[data["label"].eq(1)]

        for pattern, accounts in abusive.groupby("pattern", sort=True):
            rows.append({
                "method": method,
                "pattern": pattern,
                "best_rank": int(accounts["rank"].min()),
                "median_rank": float(accounts["rank"].median()),
                "worst_rank": int(accounts["rank"].max()),
            })

    return pd.DataFrame(rows)


def lookalike_table(results, budgets=LOOKALIKE_BUDGETS):
    """Count innocent lookalike accounts (the dormitory) flagged at each budget."""

    rows = []

    for k in budgets:
        row = {"k": k}

        for method, result in results.items():
            row[f"dorm_flagged_by_{method}"] = int(result["data"].head(k)["pattern"].eq(DORM_PATTERN).sum())

        rows.append(row)

    return pd.DataFrame(rows)


def incident_ranks(results, incident, budgets):
    """Check where game-incident participants appear in each user-level ranking."""

    participants = set(incident.get("participants", []))

    if not participants:
        raise ValueError("incident_truth.json must list the incident participants")

    rows = []

    for method, result in results.items():
        data = result["data"]
        ranks = data.loc[data["user_id"].isin(participants), "rank"]

        row = {
            "method": method,
            "participants": len(participants),
            "best_rank": int(ranks.min()),
            "median_rank": float(ranks.median()),
        }

        for k in budgets:
            row[f"in_top_{k}"] = int(ranks.le(k).sum())

        rows.append(row)

    return pd.DataFrame(rows)


def shared_ip_experiment(features, labels, baseline_data, config_path, budgets):
    """Re-score the rules with a raised shared-IP weight and compare with the baseline."""

    config = score_rules.load_config(config_path)
    experiment = join_labels(validate_scores(score_rules.score_features(features, config)), labels)

    rows = []

    for name, data in [("baseline", baseline_data), ("raised_shared_ip", experiment)]:
        dorm_ranks = data.loc[data["pattern"].eq(DORM_PATTERN), "rank"]

        row = {
            "configuration": name,
            "average_precision": calculate_average_precision(data),
            "first_dorm_rank": int(dorm_ranks.min()),
        }

        for k in budgets:
            top = evaluate_top_k(data, k)
            row[f"precision_at_{k}"] = top["precision"]
            row[f"dorm_in_top_{k}"] = int(top["accounts"]["pattern"].eq(DORM_PATTERN).sum())

        rows.append(row)

    return pd.DataFrame(rows), float(config["shared_ip"]["weight"])


def rank_model_scores(data, anomaly_scores):
    """Rank model scores with the same hash tie-break as score_model.py."""

    ranked = data[["user_id"]].copy()
    ranked["anomaly_score"] = anomaly_scores.to_numpy()
    ranked["tie_breaker"] = ranked["user_id"].map(lambda user_id: hashlib.sha256(str(user_id).encode("utf-8")).hexdigest())

    ranked = ranked.sort_values(["anomaly_score", "tie_breaker"], ascending=[False, True]).drop(columns="tie_breaker").reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    return ranked


def seed_stability(feature_data, labels, rule_data, k, seeds):
    """Re-fit the model with different forest seeds and measure what changes."""

    rule_ids = set(rule_data.head(k)["user_id"])
    abusive_patterns = sorted(labels.loc[labels["label"].eq(1), "pattern"].unique())

    rows = []

    for seed in seeds:
        anomaly_scores, *_ = score_model.score_accounts(feature_data, random_state=seed)
        data = join_labels(validate_scores(rank_model_scores(feature_data, anomaly_scores), "anomaly_score"), labels)

        top = data.head(k)
        model_ids = set(top["user_id"])

        row = {
            "seed": seed,
            f"precision_at_{k}": evaluate_top_k(data, k)["precision"],
            "average_precision": calculate_average_precision(data, "anomaly_score"),
            f"jaccard_with_rules_at_{k}": len(rule_ids & model_ids) / len(rule_ids | model_ids),
        }

        for pattern in abusive_patterns:
            row[f"{pattern}_in_top_{k}"] = int(top["pattern"].eq(pattern).sum())

        for budget in [120, 200]:
            row[f"dorm_in_top_{budget}"] = int(data.head(budget)["pattern"].eq(DORM_PATTERN).sum())

        rows.append(row)

    per_seed = pd.DataFrame(rows)
    spread = per_seed.drop(columns="seed").agg(["min", "median", "max"]).T
    spread.index.name = "measure"

    return per_seed, spread


def realistic_base_rate(results, base_rate, draws, ks, seed=0):
    """Subsample abusive accounts to a realistic base rate and re-measure precision."""

    if not 0 < base_rate < 1:
        raise ValueError("base_rate must be between 0 and 1")

    rng = np.random.default_rng(seed)
    rows = []

    for method, result in results.items():
        data = result["data"]
        score_column = result["score_column"]

        innocent = data[data["label"].eq(0)]
        abusive = data[data["label"].eq(1)]

        # Keep every innocent account; keep only enough abusers for the target rate.
        keep_abusive = max(1, round(base_rate * len(innocent) / (1 - base_rate)))

        precisions = {k: [] for k in ks}
        average_precisions = []

        for _ in range(draws):
            sample = abusive.iloc[rng.choice(len(abusive), size=keep_abusive, replace=False)]

            # Removing accounts does not change the order of the rest.
            subset = pd.concat([innocent, sample]).sort_values("rank")

            for k in ks:
                precisions[k].append(subset.head(k)["label"].mean())

            average_precisions.append(average_precision_score(subset["label"], subset[score_column]))

        row = {
            "method": method,
            "base_rate": keep_abusive / (len(innocent) + keep_abusive),
            "abusive_accounts_kept": keep_abusive,
            "median_average_precision": float(np.median(average_precisions)),
        }

        for k in ks:
            row[f"median_precision_at_{k}"] = float(np.median(precisions[k]))
            # With few abusers, even a perfect ranking cannot fill k slots.
            row[f"best_possible_precision_at_{k}"] = min(keep_abusive, k) / k

        rows.append(row)

    return pd.DataFrame(rows)


def markdown_table(table):
    """Format a table as markdown without needing the optional tabulate package."""

    def cell(value):
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    table = table.reset_index() if table.index.name else table
    header = "| " + " | ".join(str(column) for column in table.columns) + " |"
    divider = "|" + "|".join("---" for _ in table.columns) + "|"
    body = ["| " + " | ".join(cell(value) for value in row) + " |" for row in table.itertuples(index=False)]

    return "\n".join([header, divider, *body])


def save_reports(results, comparison, overlap, extra_tables, summary_lines, output_dir):
    """Save rules and model evaluations in a new report folder."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    comparison.to_csv(output_dir / "budget_comparison.csv", index=False)
    overlap.to_csv(output_dir / "ranking_overlap.csv", index=False)

    for name, table in extra_tables.items():
        if table is not None:
            table.to_csv(output_dir / f"{name}.csv", index=table.index.name is not None)

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

    # A readable summary the README can quote directly.
    with (output_dir / "summary.md").open("w", encoding="utf-8") as file:
        file.write("\n".join(summary_lines) + "\n")

    return output_dir


def banner(title, summary_lines, table=None, note=None):
    """Print a section and add the same section to the markdown summary."""

    print(f"\n{title}")
    print("=" * len(title))
    summary_lines.extend(["", f"## {title}", ""])

    if table is not None:
        print(table.to_string(index=table.index.name is not None, float_format=lambda value: f"{value:.3f}", na_rep="undefined"))
        summary_lines.append(markdown_table(table))

    if note:
        print(f"\n{note}")
        summary_lines.extend(["", note])


def main():
    """Evaluate rules and Isolation Forest using the same labelled accounts."""

    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"

    parser = argparse.ArgumentParser(description="Evaluate both detectors against the synthetic labels.")
    parser.add_argument("--config", type=Path, default=project_dir / "config" / "rules.yaml", help="configuration holding the review capacity")
    parser.add_argument("--experiment-config", type=Path, default=project_dir / "config" / "rules_shared_ip_raised.yaml", help="rules configuration for the shared-IP experiment")
    parser.add_argument("--seeds", type=int, default=10, help="number of forest seeds for the stability check (0 to skip)")
    parser.add_argument("--base-rate", type=float, default=0.005, help="realistic share of abusive accounts to re-measure at")
    parser.add_argument("--draws", type=int, default=200, help="random subsamples for the base-rate check")
    args = parser.parse_args()

    k = load_review_capacity(args.config)

    stages = load_verified_stages(data_dir)

    labels = load_labels(io.BytesIO(stages["labels"]))
    rule_data = join_labels(load_scores(io.BytesIO(stages["rule_scores"])), labels)
    model_data = join_labels(load_scores(io.BytesIO(stages["model_scores"]), score_column="anomaly_score"), labels)

    if set(rule_data["user_id"]) != set(model_data["user_id"]):
        raise ValueError("Rules and model results must contain the same accounts.")

    incident = json.loads(stages["incident"].decode("utf-8"))

    budgets = sorted(set(BUDGETS) | {k})

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
            "score_column": score_column,
            "average_precision": calculate_average_precision(data, score_column=score_column),
            "budgets": {},
            "scenarios": {},
        }

        for budget in budgets:
            evaluation = evaluate_at_budget(data, budget)
            results[method]["budgets"][budget] = evaluation
            results[method]["scenarios"][budget] = calculate_scenario_breakdowns(data, k=budget)

            comparison_rows.append({
                "method": method,
                "k": budget,
                **evaluation["outcomes"],
                **evaluation["metrics"],
            })

    # Compare which accounts appear in both review lists.
    for budget in budgets:
        rule_ids = set(evaluate_top_k(rule_data, budget)["accounts"]["user_id"])
        model_ids = set(evaluate_top_k(model_data, budget)["accounts"]["user_id"])

        shared_ids = rule_ids & model_ids
        combined_ids = rule_ids | model_ids

        overlap_rows.append({
            "k": budget,
            "shared_accounts": len(shared_ids),
            "rules_only": len(rule_ids - model_ids),
            "model_only": len(model_ids - rule_ids),
            "overlap_fraction": len(shared_ids) / budget,
            "jaccard_similarity": len(shared_ids) / len(combined_ids),
        })

    comparison = pd.DataFrame(comparison_rows).sort_values(["k", "method"]).reset_index(drop=True)
    overlap = pd.DataFrame(overlap_rows)

    summary_lines = [f"# SAD-Lite evaluation (K = {k})"]

    # --- Setup --------------------------------------------------------------
    abusive_count = int(labels["label"].sum())
    base_rate = abusive_count / len(labels)

    setup = pd.DataFrame([{
        "accounts": len(labels),
        "abusive_accounts": abusive_count,
        "base_rate": base_rate,
        "flag_nobody_accuracy": 1 - base_rate,
        "review_capacity_k": k,
    }])

    banner("Setup", summary_lines, setup,
           "Flagging nobody is this accurate, which is why accuracy is not reported.")

    # --- Headline numbers at K ------------------------------------------------
    headline = pd.DataFrame([
        {
            "method": method,
            f"precision_at_{k}": result["budgets"][k]["metrics"]["precision"],
            f"recall_at_{k}": result["budgets"][k]["metrics"]["recall"],
            "false_positives": result["budgets"][k]["outcomes"]["FP"],
            "average_precision": result["average_precision"],
        }
        for method, result in results.items()
    ])

    banner(f"Headline numbers at K = {k}", summary_lines, headline)

    curve = comparison[["method", "k", "precision", "recall", "FP"]]
    banner("Precision and recall as the budget grows", summary_lines, curve,
           "Precision falls and recall rises as K grows. Choosing where to sit is a business decision.")

    # --- Coverage and agreement ---------------------------------------------
    coverage = caught_by_pattern(results, k)
    banner(f"What each method catches in its top {k}", summary_lines, coverage)

    banner("Do the rankings agree?", summary_lines, overlap)

    ranges = rank_ranges(results)
    banner("Rank ranges of each abusive scenario", summary_lines, ranges)

    # --- False positives that matter -----------------------------------------
    lookalikes = lookalike_table(results)
    banner("Innocent lookalikes: dormitory accounts flagged", summary_lines, lookalikes,
           "Dormitory users are legitimate customers behind one shared IP. The model has no "
           "labels, so it cannot learn that a shared IP is usually innocent; the low shared-IP "
           "weight in the rules is that knowledge, written down.")

    # --- Game incident -----------------------------------------------------
    incident_table = incident_ranks(results, incident, [40, 120, 200])
    banner("Game-incident participants in the user-level rankings", summary_lines, incident_table,
           f"{incident['game_name']} terminated {incident['window_terminated']} of "
           f"{incident['window_sessions']} sessions in its incident window "
           f"({incident['window_termination_rate']:.1%}, against "
           f"{incident['baseline_termination_rate']:.1%} outside it). Each participant had one "
           f"bad session, so user-level detection should not single them out.")

    # --- Shared-IP experiment -----------------------------------------------
    feature_table = score_rules.validate_features(provenance.frame_from_bytes(stages["features"]))
    experiment = None

    if args.experiment_config.exists():
        experiment, raised_weight = shared_ip_experiment(feature_table, labels, rule_data, args.experiment_config, [k, 120, 200])
        banner(f"Experiment: shared-IP weight raised to {raised_weight:g}", summary_lines, experiment)
    else:
        print(f"\nSkipping the shared-IP experiment: {args.experiment_config} not found.")

    # --- Model stability across forest seeds -----------------------------------
    seed_runs = None
    seed_spread = None

    if args.seeds > 0:
        model_feature_data = score_model.load_features(stages["features"])
        seed_runs, seed_spread = seed_stability(model_feature_data, labels, rule_data, k, list(range(args.seeds)))
        banner(f"Model stability across {args.seeds} forest seeds", summary_lines, seed_spread,
               "Measures with a narrow range are findings; measures with a wide range depend on the seed.")

    # --- Realistic base rate ----------------------------------------------------
    realistic = realistic_base_rate(results, args.base_rate, args.draws, [10, k])
    banner(f"Re-measured at a {args.base_rate:.1%} base rate", summary_lines, realistic,
           "The same rankings with most abusers removed. Precision at K falls because there are "
           "fewer abusers to find, not because the detectors changed.")

    run_name = datetime.now(timezone.utc).strftime("comparison_%Y%m%d_%H%M%S_%f_UTC")

    extra_tables = {
        "caught_by_pattern": coverage,
        "rank_ranges": ranges,
        "lookalikes": lookalikes,
        "incident_ranks": incident_table,
        "shared_ip_experiment": experiment,
        "seed_stability_runs": seed_runs,
        "seed_stability_spread": seed_spread,
        "realistic_base_rate": realistic,
    }

    output_dir = save_reports(results, comparison, overlap, extra_tables, summary_lines, project_dir / "reports" / run_name)

    print(f"\nReports saved to: {output_dir}")
    print("Evaluation on synthetic data; the model was fitted and scored on the same accounts.")


if __name__ == "__main__":
    main()