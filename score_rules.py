# =============================================================================
# SAD-Lite — Rule-Based Account Scoring
# =============================================================================
#
# Reads data/features.csv and configurable thresholds from config/rules.yaml.
# Produces ranked accounts with scores, evidence, and alternative explanations.
#
# HOW A RULE SCORES
# Every rule reads one feature and turns it into a signal between 0 and 1:
# - 0 at or before `trigger`, the value where the rule starts to care.
# - 1 at or beyond `saturate`, the value where the rule is fully convinced.
# - A straight line in between.
# For "lower is suspicious" rules (low variation, young accounts), `saturate`
# is below `trigger`; the same formula handles both directions.
#
# HOW THE SCORE IS BUILT
# score = sum(weight * signal) / sum(weight), so every score is between 0 and 1.
# No real account trips every rule, so the top of the scale is unreachable in
# practice. Only the ranking carries meaning; the score is not a probability.
#
# Rules:
# - Shared devices and shared payment identifiers (entity graph).
# - Frequent anti-cheat/abuse terminations with sufficient history.
# - Many countries or IP addresses within one day, and many devices overall.
# - Low variation in session duration, and separately in input activity.
# - Round-the-clock play across most hours of the day.
# - Accounts that were new at their first session.
# - Shared IP addresses, with a deliberately low weight: colleges, cafés,
#   households, and carrier-grade NAT share IPs far more often than rings do.
#   The low weight is that knowledge, written down.
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
#
# SOURCES
# The rule set, weights, triggers, and saturation points follow the SAD-Lite
# reference configuration (config/detection.yaml). Entity-graph triggers are
# one lower than the reference because our counts exclude the account itself.
# =============================================================================

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# Created as config/rules.yaml on the first run.
# After that, edit the YAML file to change these settings.
DEFAULT_RULES = {
    "shared_device": {
        "weight": 3.0,
        "trigger": 3,
        "saturate": 19,
    },
    "shared_payment": {
        "weight": 2.5,
        "trigger": 2,
        "saturate": 19,
    },
    "termination_rate": {
        "weight": 3.0,
        "trigger": 0.08,
        "saturate": 0.40,
        "min_sessions": 10,
        "min_events": 3,
    },
    "countries_per_day": {
        "weight": 2.5,
        "trigger": 1,
        "saturate": 4,
    },
    "duration_consistency": {
        "weight": 2.0,
        "trigger": 0.25,
        "saturate": 0.05,
        "min_sessions": 10,
    },
    "input_consistency": {
        "weight": 2.0,
        "trigger": 0.15,
        "saturate": 0.03,
        "min_sessions": 10,
    },
    "distinct_devices": {
        "weight": 1.5,
        "trigger": 3,
        "saturate": 8,
    },
    "ips_per_day": {
        "weight": 1.5,
        "trigger": 2,
        "saturate": 8,
    },
    "round_the_clock": {
        "weight": 1.5,
        "trigger": 18,
        "saturate": 24,
    },
    "new_account": {
        "weight": 1.5,
        "trigger": 21,
        "saturate": 1,
    },
    # Deliberately low weight; raising it is the shared-IP experiment.
    "shared_ip": {
        "weight": 0.4,
        "trigger": 14,
        "saturate": 79,
    },
    "review": {
        # One top-weight rule at full strength: 3.0 / 21.4 ≈ 0.14.
        "min_score": 0.14,
        # How many accounts one reviewer can examine in a day.
        "capacity": 40,
    },
}

# The feature each rule reads. Kept in code, not config, so a config edit
# cannot silently point a rule at a different measurement.
RULE_FEATURES = {
    "shared_device": "max_other_users_on_device",
    "shared_payment": "max_other_users_on_payment",
    "termination_rate": "termination_rate",
    "countries_per_day": "max_countries_per_day",
    "duration_consistency": "duration_cv",
    "input_consistency": "input_cv",
    "distinct_devices": "distinct_devices",
    "ips_per_day": "max_ips_per_day",
    "round_the_clock": "distinct_hours",
    "new_account": "account_age_at_first_session_days",
    "shared_ip": "max_other_users_on_ip",
}

RULE_NAMES = list(RULE_FEATURES)

# Rules measuring a rate between 0 and 1.
RATE_RULES = ["termination_rate"]

REQUIRED = [
    "session_count", "termination_count", "mean_input_rate",
    *RULE_FEATURES.values(),
]

# Features that must contain whole numbers.
COUNT_FEATURES = [
    "session_count", "termination_count", "distinct_devices", "distinct_hours",
    "max_ips_per_day", "max_countries_per_day", "max_other_users_on_device",
    "max_other_users_on_payment", "max_other_users_on_ip",
]

# How many reasons appear in the one-line evidence summary.
SUMMARY_REASONS = 4


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

    for name in RULE_NAMES:
        rule = config[name]

        # Equal values would turn the ramp into an on/off switch.
        if rule["trigger"] == rule["saturate"]:
            raise ValueError(f"{name}.trigger and {name}.saturate must differ")

        for key in ["min_sessions", "min_events"]:
            if key in rule and (rule[key] < 1 or rule[key] != int(rule[key])):
                raise ValueError(f"{name}.{key} must be a positive integer")

    for name in RATE_RULES:
        if not (config[name]["trigger"] <= 1 and config[name]["saturate"] <= 1):
            raise ValueError(f"{name} thresholds must be at most 1")

    if sum(config[name]["weight"] for name in RULE_NAMES) <= 0:
        raise ValueError("At least one rule must have a positive weight")

    if not 0 < config["review"]["min_score"] <= 1:
        raise ValueError("review.min_score must be in (0, 1]")

    capacity = config["review"]["capacity"]

    if capacity < 1 or capacity != int(capacity):
        raise ValueError("review.capacity must be a positive integer")


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
        "input_cv": (features["session_count"] < 2) | features["mean_input_rate"].eq(0),
        "termination_game_concentration": features["termination_count"] < 3,
    }

    for column in features.columns.drop("user_id"):
        allowed = allowed_missing.get(column, pd.Series(False, index=features.index))

        if (features[column].isna() & ~allowed).any():
            raise ValueError(f"Unexplained missing values in {column}")

    for column in COUNT_FEATURES:
        if (features[column] % 1 != 0).any():
            raise ValueError(f"{column} must contain whole counts")

    if features["session_count"].lt(1).any():
        raise ValueError("Accounts must have at least one session")

    if (features["termination_count"] > features["session_count"]).any():
        raise ValueError("More terminations than sessions")

    expected = features["termination_count"] / features["session_count"]

    if not np.allclose(features["termination_rate"], expected, atol=1e-8, rtol=0):
        raise ValueError("Termination rates do not match counts")

    if not features["distinct_hours"].between(1, 24).all():
        raise ValueError("distinct_hours must be between 1 and 24")

    return features


def rule_signal(value, trigger, saturate):
    """Map a feature value onto 0..1 between trigger and saturate, in either direction."""

    # An undefined feature cannot support a rule.
    if pd.isna(value):
        return 0.0

    signal = (value - trigger) / (saturate - trigger)

    return float(min(max(signal, 0.0), 1.0))


def score_account(row, config):
    """Calculate one account's score and supporting evidence."""

    evidence = []
    signals = {}

    total_weight = sum(config[name]["weight"] for name in RULE_NAMES)

    # Nested helper: compute a rule's signal and record evidence when it contributes.
    def record(rule, reason, caveat, enabled=True):
        settings = config[rule]
        value = getattr(row, RULE_FEATURES[rule])
        signal = rule_signal(value, settings["trigger"], settings["saturate"]) if enabled else 0.0
        points = settings["weight"] * signal

        signals[rule] = signal

        if points > 0:
            evidence.append({
                "rule": rule,
                "feature": RULE_FEATURES[rule],
                "value": float(value),
                "signal": round(signal, 4),
                "weight": float(settings["weight"]),
                "contribution": round(points / total_weight, 4),
                "reason": reason,
                "caveat": caveat,
            })

    # 1. Shared device: many other accounts using one of this account's devices.
    record(
        "shared_device",
        f"A device is shared with {row.max_other_users_on_device:.0f} other accounts.",
        "Family computers and gaming cafés also share devices; check payment and creation dates.",
    )

    # 2. Shared payment identifier.
    record(
        "shared_payment",
        f"A payment identifier is shared with {row.max_other_users_on_payment:.0f} other accounts.",
        "Households can share one card; the device and payment groups may differ.",
    )

    # 3. Repeated terminations: require sufficient history and count, not only a rate.
    rule = config["termination_rate"]
    record(
        "termination_rate",
        f"{row.termination_count:.0f} of {row.session_count:.0f} sessions ended in anti-cheat/abuse termination ({row.termination_rate:.1%}).",
        "A game incident or anti-cheat fault may affect innocent accounts; inspect the timeline and game.",
        enabled=row.session_count >= rule["min_sessions"] and row.termination_count >= rule["min_events"],
    )

    # 4. Countries within one day.
    record(
        "countries_per_day",
        f"Played from {row.max_countries_per_day:.0f} countries within a single day.",
        "VPNs and travel produce the same pattern.",
    )

    # 5–6. Consistency: each measured separately, each needing enough sessions.
    rule = config["duration_consistency"]
    record(
        "duration_consistency",
        f"Session lengths are unusually consistent across {row.session_count:.0f} sessions (CV={row.duration_cv:.3f}).",
        "Platform session limits and regular routines can also produce consistent lengths.",
        enabled=row.session_count >= rule["min_sessions"],
    )

    rule = config["input_consistency"]
    record(
        "input_consistency",
        f"Input activity is unusually consistent across {row.session_count:.0f} sessions (CV={row.input_cv:.3f}).",
        "These are session averages; similar gameplay can produce similar averages.",
        enabled=row.session_count >= rule["min_sessions"],
    )

    # 7. Devices overall.
    record(
        "distinct_devices",
        f"Used {row.distinct_devices:.0f} distinct devices.",
        "Device upgrades and several household devices also raise this count.",
    )

    # 8. IP addresses within one day.
    record(
        "ips_per_day",
        f"Used {row.max_ips_per_day:.0f} distinct IP addresses within a single day.",
        "Mobile networks and switching between home and work networks change IPs.",
    )

    # 9. Round-the-clock play.
    record(
        "round_the_clock",
        f"Started sessions in {row.distinct_hours:.0f} different hours of the day.",
        "Hours are UTC; shift workers and long sessions spread start times too.",
    )

    # 10. New account at first session.
    record(
        "new_account",
        f"The account was {row.account_age_at_first_session_days:.1f} days old at its first session.",
        "Every legitimate player also starts with a new account.",
    )

    # 11. Shared IP: low weight by design.
    record(
        "shared_ip",
        f"An IP address is shared with {row.max_other_users_on_ip:.0f} other accounts.",
        "Colleges, workplaces, cafés, households, and carrier-grade NAT commonly share IP addresses.",
    )

    score = sum(item["weight"] * item["signal"] for item in evidence) / total_weight

    # Strongest contributions first, so the summary leads with the main reason.
    evidence.sort(key=lambda item: item["contribution"], reverse=True)
    shown = evidence[:SUMMARY_REASONS]

    result = {
        "user_id": row.user_id,
        "rule_score": score,
        "review_recommended": score >= config["review"]["min_score"],
        "triggered_rules": "; ".join(item["rule"] for item in evidence),
        "evidence_summary": " | ".join(item["reason"] for item in shown) or "No enabled scoring rule triggered; this does not establish innocence.",
        "caveats": " | ".join(item["caveat"] for item in shown),
        "evidence_json": json.dumps(evidence, ensure_ascii=False, allow_nan=False),
    }

    # Per-rule signals, for investigation and for comparing configurations.
    for rule in RULE_NAMES:
        result[f"signal_{rule}"] = round(signals[rule], 4)

    return result


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

    parser = argparse.ArgumentParser(description="Score accounts with weighted, explainable rules.")
    parser.add_argument("--config", type=Path, default=root / "config" / "rules.yaml", help="rule configuration to use")
    parser.add_argument("--output", type=Path, default=root / "data" / "rule_scores.csv", help="where to save the ranking")
    parser.add_argument("--top", type=int, default=15, help="how many accounts to print")
    args = parser.parse_args()

    config = load_config(args.config)
    features = validate_features(pd.read_csv(root / "data" / "features.csv"))
    scores = score_features(features, config)

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False)

    feature_distributions(features).to_csv(reports / "feature_distributions.csv")

    # Save the exact settings used to produce this ranking.
    snapshot = reports / f"rules_used_{args.output.stem}.yaml"

    with snapshot.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    pd.set_option("display.max_colwidth", 110)
    print(scores[["rank", "user_id", "rule_score", "evidence_summary"]].head(args.top).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nScored {len(scores)} accounts with {args.config.name}; {int(scores.review_recommended.sum())} meet the review threshold.")
    print(f"Review capacity: {config['review']['capacity']} accounts.")
    print(f"Ranking saved to: {args.output}")
    print(f"Distribution statistics saved to: {reports / 'feature_distributions.csv'}")
    print(f"Configuration snapshot saved to: {snapshot}")
    print("Scores are review priorities, not probabilities or enforcement decisions.")


if __name__ == "__main__":
    main()