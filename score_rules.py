# =============================================================================
# SAD-Lite — Rule-Based Account Scoring
# =============================================================================
#
# Reads data/features.csv and configurable thresholds from config/rules.yaml.
# Produces ranked accounts with scores, evidence, and alternative explanations.
#
# Initial rules cover:
# - Frequent anti-cheat/abuse terminations with sufficient history.
# - Low variation in BOTH input activity and session duration.
# - High daily device/country diversity.
# - Shared devices and payment identifiers, supported by account age.
# - Repeated payment failures as weak supporting context.
# - Shared IP addresses, disabled by default through a zero weight.
#
# Distribution statistics are exported for inspection. They do not
# automatically change thresholds or scores.
#
# Scores are review priorities, not probabilities or proof of abuse.
# This file does not read labels or incident truth and does not punish users.
# Evidence summaries support later investigation and mock enforcement.
#
# Initial thresholds are synthetic-project assumptions, not production rules.
# Equal scores use a deterministic hash tie-break, not account-number order.
# =============================================================================

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# Created as config/rules.yaml on the first run.
# After that, edit the YAML file to change these settings.
DEFAULT_RULES = {
    "termination": {
        "min_sessions": 10,
        "min_events": 3,
        "min_rate": 0.20,
        "weight": 30,
    },
    "automation": {
        "min_sessions": 10,
        "max_input_cv": 0.05,
        "max_duration_cv": 0.05,
        "weight": 30,
    },
    "diversity": {
        "min_sessions": 10,
        "min_daily_devices": 3,
        "min_daily_countries": 3,
        "single_weight": 10,
        "both_weight": 25,
    },
    "coordination": {
        "min_other_device_users": 2,
        "min_other_payment_users": 2,
        "weight": 10,
        "max_age_days": 14,
        "young_bonus": 15,
    },
    "payment": {
        "min_attempts": 5,
        "min_failures": 3,
        "min_rate": 0.50,
        "weight": 5,
    },
    "shared_ip": {
        "min_other_users": 10,
        "weight": 0,
    },
    "review": {
        "min_score": 25,
    },
}

REQUIRED = [
    "session_count", "termination_count", "termination_rate", "input_cv",
    "duration_cv", "max_devices_per_day", "max_countries_per_day",
    "max_other_users_on_device", "max_other_users_on_payment",
    "account_age_at_first_session_days", "payment_attempt_count",
    "payment_failure_count", "payment_failure_rate", "max_other_users_on_ip",
]


def validate_config(config):
    """Reject missing settings, invalid thresholds, and invalid weights."""

    if not isinstance(config, dict) or set(config) != set(DEFAULT_RULES):
        raise ValueError("Rule configuration must contain exactly the default rule sections")

    for section, defaults in DEFAULT_RULES.items():
        values = config[section]

        if not isinstance(values, dict) or set(values) != set(defaults):
            raise ValueError(f"Unexpected or missing settings in {section}")

        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{section}.{key} must be a finite nonnegative number")

            if key == "min_rate" and value > 1:
                raise ValueError(f"{section}.{key} must be at most 1")

    positive_counts = {
        "termination": ["min_sessions", "min_events"],
        "automation": ["min_sessions"],
        "diversity": ["min_sessions", "min_daily_devices", "min_daily_countries"],
        "coordination": ["min_other_device_users", "min_other_payment_users"],
        "payment": ["min_attempts", "min_failures"],
        "shared_ip": ["min_other_users"],
    }

    for section, keys in positive_counts.items():
        for key in keys:
            value = config[section][key]

            if value < 1 or value != int(value):
                raise ValueError(f"{section}.{key} must be a positive integer")

    if config["review"]["min_score"] <= 0:
        raise ValueError("review.min_score must be positive")

    if config["diversity"]["both_weight"] < config["diversity"]["single_weight"]:
        raise ValueError("both_weight must be at least single_weight")


def load_config(path):
    """Create the initial config if needed, then load and validate it."""

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(DEFAULT_RULES, file, sort_keys=False)

    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    validate_config(config)

    return config


def validate_features(features):
    """Validate the feature table before scoring."""

    features = features.copy()
    missing = {"user_id", *REQUIRED} - set(features.columns)

    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    if features.empty:
        raise ValueError("Feature table is empty")

    if {"label", "pattern"} & set(features.columns):
        raise ValueError("Remove ground-truth columns before scoring")

    if features["user_id"].isna().any() or features["user_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Missing account identifiers")

    features["user_id"] = features["user_id"].astype(str)

    if not features["user_id"].is_unique:
        raise ValueError("Duplicate account identifiers")

    for column in features.columns.drop("user_id"):
        features[column] = pd.to_numeric(features[column], errors="raise")

        if np.isinf(features[column]).any() or features[column].dropna().lt(0).any():
            raise ValueError(f"Invalid numeric values in {column}")

    allowed_missing = {
        "std_duration_min": features["session_count"] < 2,
        "std_input_rate": features["session_count"] < 2,
        "duration_cv": features["session_count"] < 2,
        "input_cv": features["session_count"] < 2,
        "payment_failure_rate": features["payment_attempt_count"].eq(0),
    }

    if "mean_input_rate" in features:
        allowed_missing["input_cv"] = allowed_missing["input_cv"] | features["mean_input_rate"].eq(0)

    for column in features.columns.drop("user_id"):
        allowed = allowed_missing.get(column, pd.Series(False, index=features.index))

        if (features[column].isna() & ~allowed).any():
            raise ValueError(f"Unexplained missing values in {column}")

    count_columns = [
        name for name in REQUIRED
        if name.endswith("_count")
        or (name.startswith(("max_", "payment_attempt_", "payment_failure_")) and not name.endswith("_rate"))
    ]

    for column in count_columns:
        if (features[column] % 1 != 0).any():
            raise ValueError(f"{column} must contain whole counts")

    if features["session_count"].lt(1).any():
        raise ValueError("Accounts must have at least one session")

    if (features["termination_count"] > features["session_count"]).any():
        raise ValueError("More terminations than sessions")

    if (features["payment_failure_count"] > features["payment_attempt_count"]).any() or (features["payment_attempt_count"] > features["session_count"]).any():
        raise ValueError("Inconsistent payment counts")

    expected = features["termination_count"] / features["session_count"]

    if not np.allclose(features["termination_rate"], expected, atol=1e-8, rtol=0):
        raise ValueError("Termination rates do not match counts")

    expected = features["payment_failure_count"] / features["payment_attempt_count"].replace(0, np.nan)

    if not np.allclose(features["payment_failure_rate"], expected, atol=1e-8, rtol=0, equal_nan=True):
        raise ValueError("Payment rates do not match counts")

    return features


def score_account(row, config):
    """Calculate one account's score and supporting evidence."""

    evidence = []

    # Nested helper: append evidence only when a rule contributes points.
    def record(rule, points, reason, caveat):
        if points > 0:
            evidence.append({
                "rule": rule,
                "points": float(points),
                "reason": reason,
                "caveat": caveat,
            })

    # 1. Repeated terminations: require sufficient history, count, and rate.
    rule = config["termination"]

    if row.session_count >= rule["min_sessions"] and row.termination_count >= rule["min_events"] and row.termination_rate >= rule["min_rate"]:
        reason = f"{row.termination_count} of {row.session_count} sessions ended in anti-cheat/abuse termination ({row.termination_rate:.1%})."
        caveat = "A game incident or false positive may affect innocent accounts; inspect the timeline and game."
        record("frequent_terminations", rule["weight"], reason, caveat)

    # 2. Automation: both consistency measurements must meet thresholds.
    rule = config["automation"]

    if row.session_count >= rule["min_sessions"] and pd.notna(row.input_cv) and pd.notna(row.duration_cv):
        if row.input_cv <= rule["max_input_cv"] and row.duration_cv <= rule["max_duration_cv"]:
            reason = f"Across {row.session_count} sessions, input CV={row.input_cv:.4f} and duration CV={row.duration_cv:.4f}."
            caveat = "Regular routines and session limits can also produce consistency; these are session averages."
            record("consistent_activity", rule["weight"], reason, caveat)

    # 3. Diversity: both_weight replaces single_weight; they are not added.
    rule = config["diversity"]

    if row.session_count >= rule["min_sessions"]:
        devices = row.max_devices_per_day >= rule["min_daily_devices"]
        countries = row.max_countries_per_day >= rule["min_daily_countries"]

        if devices or countries:
            points = rule["both_weight"] if devices and countries else rule["single_weight"]
            reason = f"Daily maxima: {row.max_devices_per_day} devices and {row.max_countries_per_day} countries."
            caveat = "The maxima may occur on different days; VPNs, travel, or device changes may explain them."
            record("daily_diversity", points, reason, caveat)

    # 4. Coordination: require both forms of sharing.
    # Account age adds points only when sharing already meets the conditions.
    rule = config["coordination"]

    if row.max_other_users_on_device >= rule["min_other_device_users"] and row.max_other_users_on_payment >= rule["min_other_payment_users"]:
        young = row.account_age_at_first_session_days <= rule["max_age_days"]
        points = rule["weight"] + (rule["young_bonus"] if young else 0)

        reason = (
            f"Device sharing with up to {row.max_other_users_on_device} other accounts; "
            f"payment sharing with up to {row.max_other_users_on_payment}; "
            f"age at first session={row.account_age_at_first_session_days:.1f} days."
        )
        caveat = "These may be different account groups; households can share both. This does not establish similar creation dates."
        record("shared_entities", points, reason, caveat)

    # 5. Payments: weak supporting context, never a missing-payment penalty.
    rule = config["payment"]

    if row.payment_attempt_count >= rule["min_attempts"] and row.payment_failure_count >= rule["min_failures"] and pd.notna(row.payment_failure_rate):
        if row.payment_failure_rate >= rule["min_rate"]:
            reason = f"{row.payment_failure_count} of {row.payment_attempt_count} payment attempts failed ({row.payment_failure_rate:.1%})."
            caveat = "Insufficient funds or processing faults do not establish cheating; this is weak supporting context."
            record("repeated_payment_failures", rule["weight"], reason, caveat)

    # 6. Shared IP: zero baseline weight; available for the planned experiment.
    rule = config["shared_ip"]

    if row.max_other_users_on_ip >= rule["min_other_users"]:
        reason = f"An IP was shared with up to {row.max_other_users_on_ip} other accounts."
        caveat = "Colleges, workplaces, and households commonly share IP addresses."
        record("shared_ip", rule["weight"], reason, caveat)

    score = sum(item["points"] for item in evidence)

    return {
        "user_id": row.user_id,
        "rule_score": score,
        "review_recommended": score >= config["review"]["min_score"],
        "triggered_rules": "; ".join(item["rule"] for item in evidence),
        "evidence_summary": " | ".join(item["reason"] for item in evidence) or "No enabled scoring rule triggered; this does not establish innocence.",
        "caveats": " | ".join(item["caveat"] for item in evidence),
        "evidence_json": json.dumps(evidence, ensure_ascii=False, allow_nan=False),
    }


def score_features(features, config):
    """Score all accounts and return a reproducible ranking."""

    validate_config(config)
    features = validate_features(features)

    scores = pd.DataFrame([score_account(row, config) for row in features.itertuples(index=False)])

    # Hash only breaks ties. It does not affect scores or rule decisions.
    # This avoids systematically favouring low, sequential account numbers.
    scores["_tie"] = scores["user_id"].map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())

    scores = scores.sort_values(["rule_score", "_tie"], ascending=[False, True]).drop(columns="_tie").reset_index(drop=True)
    scores.insert(0, "rank", np.arange(1, len(scores) + 1))

    return scores


def feature_distributions(features):
    """Describe numerical features without using labels or changing rules."""

    numeric = features.drop(columns="user_id")

    # count excludes missing values; std is sample standard deviation.
    # 50% is the median. Other percentiles show distribution shape.
    stats = numeric.describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
    stats.insert(1, "missing_count", numeric.isna().sum())
    stats.index.name = "feature"

    return stats


def main():
    root = Path(__file__).resolve().parent

    config = load_config(root / "config" / "rules.yaml")
    features = validate_features(pd.read_csv(root / "data" / "features.csv"))
    scores = score_features(features, config)

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    output = root / "data" / "rule_scores.csv"
    scores.to_csv(output, index=False)

    feature_distributions(features).to_csv(reports / "feature_distributions.csv")

    # Save the exact settings used to produce this ranking.
    with (reports / "rules_used.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    print(scores[["rank", "user_id", "rule_score", "review_recommended", "triggered_rules"]].head(15).to_string(index=False))
    print(f"\nScored {len(scores)} accounts; {int(scores.review_recommended.sum())} meet the review threshold.")
    print(f"Ranking saved to: {output}")
    print(f"Distribution statistics and configuration snapshot saved to: {reports}")
    print("Scores are review priorities, not probabilities or enforcement decisions.")


if __name__ == "__main__":
    main()