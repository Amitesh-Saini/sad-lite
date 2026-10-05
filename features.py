# =============================================================================
# SAD-Lite — Account Feature Extraction
# =============================================================================
#
# Reads data/sessions.csv and writes data/features.csv.
# Converts many session records into one summary row per account.
#
# Features cover activity counts, distinct identifiers, termination rates,
# duration/input variability, daily maxima, account age, shared entities,
# and payment failures.
#
# Definitions:
# - Days are UTC calendar days, assigned using session_start.
# - Account age is measured at the first observed session.
# - Shared-entity counts exclude the account itself.
# - Input mean is an unweighted average of session-level input rates.
# - Standard deviation uses the sample formula (ddof=1).
# - Payment failure rate is failures divided by payment attempts.
#
# Undefined values remain blank in the output:
# - Sample standard deviation and CV require at least two sessions.
# - Input CV is undefined when the mean input rate is zero.
# - Payment failure rate is undefined when there are no payment attempts.
# These require an explicit handling policy before model fitting.
#
# Does not read labels.csv or incident_truth.json.
# user_id identifies rows; it must not be used as a numerical model feature.
# Features describe behaviour and do not establish whether an account is abusive.
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd


def prepare_sessions(sessions):
    """Validate input and convert timestamps and numerical columns."""

    sessions = sessions.copy()

    required = [
        "session_uuid", "user_id", "account_created_at", "session_start",
        "session_end", "duration_min", "client_ip", "country", "device_id",
        "app_id", "input_events_per_min", "bytes_streamed", "payment_hash",
        "end_reason", "payment_status",
    ]

    missing = set(required) - set(sessions.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if sessions.empty:
        raise ValueError("The sessions file is empty")

    if {"label", "pattern"} & set(sessions.columns):
        raise ValueError("Ground-truth columns must not be in session data")

    if sessions[required].isna().any().any():
        raise ValueError("Required fields contain missing values")

    for column in required:
        if sessions[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Blank values in {column}")

    if sessions["session_uuid"].duplicated().any():
        raise ValueError("Duplicate session IDs")

    # Parse each value individually to support fractional-second timestamps.
    for column in ["session_start", "session_end", "account_created_at"]:
        sessions[column] = sessions[column].map(pd.Timestamp)

        if sessions[column].map(lambda value: value.tzinfo is None).any():
            raise ValueError(f"{column} must include timezone offsets")

        sessions[column] = pd.to_datetime(sessions[column], utc=True)

    for column in ["duration_min", "input_events_per_min", "bytes_streamed"]:
        sessions[column] = pd.to_numeric(sessions[column], errors="raise")

        if not np.isfinite(sessions[column]).all() or (sessions[column] < 0).any():
            raise ValueError(f"Invalid values in {column}")

    if (sessions["duration_min"] <= 0).any():
        raise ValueError("Duration must be positive")

    if (sessions["bytes_streamed"] % 1 != 0).any():
        raise ValueError("Streamed bytes must be integers")

    if (sessions["account_created_at"] > sessions["session_start"]).any() or (sessions["session_end"] <= sessions["session_start"]).any():
        raise ValueError("Invalid timestamp ordering")

    actual_duration = (sessions["session_end"] - sessions["session_start"]).dt.total_seconds() / 60

    # Allow up to 0.01 minute of rounding difference.
    if not np.allclose(actual_duration, sessions["duration_min"], atol=0.01, rtol=0):
        raise ValueError("Duration disagrees with timestamps")

    if sessions.groupby("user_id")["account_created_at"].nunique().ne(1).any():
        raise ValueError("An account has inconsistent creation timestamps")

    valid_endings = ["normal", "timeout", "crash", "terminated_anticheat", "terminated_abuse"]

    if not sessions["end_reason"].isin(valid_endings).all():
        raise ValueError("Unknown session end reason")

    if not sessions["payment_status"].isin(["not_attempted", "succeeded", "failed"]).all():
        raise ValueError("Unknown payment status")

    return sessions


def create_summary(sessions):
    """Use a dictionary to accumulate basic measurements for each account."""

    summaries = {}

    for session in sessions.itertuples(index=False):
        if session.user_id not in summaries:
            summaries[session.user_id] = {
                "count": 0,
                "devices": set(),
                "ips": set(),
                "countries": set(),
                "games": set(),
                "terminations": 0,
                "duration": 0.0,
                "input": 0.0,
                "bytes": 0,
            }

        item = summaries[session.user_id]

        item["count"] += 1
        item["devices"].add(session.device_id)
        item["ips"].add(session.client_ip)
        item["countries"].add(session.country)
        item["games"].add(session.app_id)

        item["terminations"] += session.end_reason in ("terminated_anticheat", "terminated_abuse")
        item["duration"] += session.duration_min
        item["input"] += session.input_events_per_min
        item["bytes"] += session.bytes_streamed

    rows = []

    for user_id, item in summaries.items():
        rows.append({
            "user_id": user_id,
            "session_count": item["count"],
            "distinct_devices": len(item["devices"]),
            "distinct_ips": len(item["ips"]),
            "distinct_countries": len(item["countries"]),
            "distinct_games": len(item["games"]),
            "termination_count": item["terminations"],
            "termination_rate": item["terminations"] / item["count"],
            "total_duration_min": item["duration"],
            "mean_duration_min": item["duration"] / item["count"],
            "mean_input_rate": item["input"] / item["count"],
            "total_bytes_streamed": item["bytes"],
        })

    return pd.DataFrame(rows)


def add_variability_features(sessions, summary):
    """Measure variation in duration and session-average input activity."""

    grouped = sessions.groupby("user_id")

    summary["std_duration_min"] = summary["user_id"].map(grouped["duration_min"].std(ddof=1))
    summary["std_input_rate"] = summary["user_id"].map(grouped["input_events_per_min"].std(ddof=1))

    summary["duration_cv"] = summary["std_duration_min"] / summary["mean_duration_min"]
    summary["input_cv"] = summary["std_input_rate"] / summary["mean_input_rate"].replace(0, np.nan)

    return summary


def add_daily_features(sessions, summary):
    """Calculate active days and maximum activity within one UTC start-date."""

    daily_sessions = sessions.assign(session_day=sessions["session_start"].dt.floor("D"))

    daily = daily_sessions.groupby(["user_id", "session_day"]).agg(
        daily_sessions=("session_uuid", "size"),
        daily_devices=("device_id", "nunique"),
        daily_countries=("country", "nunique"),
    )

    grouped = daily.groupby(level="user_id")

    summary["active_days"] = summary["user_id"].map(grouped.size())

    mappings = [
        ("daily_sessions", "max_sessions_per_day"),
        ("daily_devices", "max_devices_per_day"),
        ("daily_countries", "max_countries_per_day"),
    ]

    for source, target in mappings:
        summary[target] = summary["user_id"].map(grouped[source].max())

    return summary


def add_account_age(sessions, summary):
    """Measure account age in fractional days at its first observed session."""

    grouped = sessions.groupby("user_id")

    age = (grouped["session_start"].min() - grouped["account_created_at"].first()).dt.total_seconds() / 86400

    summary["account_age_at_first_session_days"] = summary["user_id"].map(age)

    return summary


def add_shared_entity_features(sessions, summary):
    """Find the largest number of other accounts sharing each entity type."""

    entities = [
        ("client_ip", "ip"),
        ("device_id", "device"),
        ("payment_hash", "payment"),
    ]

    for entity, suffix in entities:
        # Count distinct accounts using each identifier.
        counts = sessions.groupby(entity)["user_id"].nunique()

        # Subtract the current account.
        others = sessions[entity].map(counts) - 1

        # Choose the most widely shared identifier used by each account.
        maximum = others.groupby(sessions["user_id"]).max()

        summary[f"max_other_users_on_{suffix}"] = summary["user_id"].map(maximum)

    return summary


def add_payment_features(sessions, summary):
    """Count payment attempts and failures, then calculate failure rates."""

    attempted = sessions["payment_status"].isin(["succeeded", "failed"])
    failed = sessions["payment_status"].eq("failed")

    summary["payment_attempt_count"] = summary["user_id"].map(attempted.groupby(sessions["user_id"]).sum())
    summary["payment_failure_count"] = summary["user_id"].map(failed.groupby(sessions["user_id"]).sum())

    # Zero attempts means the rate is undefined, not a measured zero.
    summary["payment_failure_rate"] = summary["payment_failure_count"] / summary["payment_attempt_count"].replace(0, np.nan)

    return summary


def validate_summary(sessions, summary):
    """Reject inconsistent outputs and unexplained missing values."""

    if not summary["user_id"].is_unique or set(summary["user_id"]) != set(sessions["user_id"]):
        raise ValueError("Summary must contain exactly one row per account")

    if summary["session_count"].sum() != len(sessions):
        raise ValueError("Session counts do not match")

    numeric = summary.drop(columns="user_id")

    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Infinite feature values found")

    if (numeric < 0).any().any():
        raise ValueError("Negative feature values found")

    # Missing values are permitted only under these exact conditions.
    allowed_missing = {
        "std_duration_min": summary["session_count"] < 2,
        "std_input_rate": summary["session_count"] < 2,
        "duration_cv": summary["session_count"] < 2,
        "input_cv": (summary["session_count"] < 2) | summary["mean_input_rate"].eq(0),
        "payment_failure_rate": summary["payment_attempt_count"].eq(0),
    }

    for column in numeric.columns:
        expected = allowed_missing.get(column, pd.Series(False, index=summary.index))

        if not numeric[column].isna().equals(expected):
            raise ValueError(f"Unexpected missing-value pattern in {column}")

    for column in ["termination_rate", "payment_failure_rate"]:
        if not summary[column].dropna().between(0, 1).all():
            raise ValueError(f"{column} is outside [0, 1]")


def build_features(sessions):
    """Run validation and all feature calculations."""

    sessions = prepare_sessions(sessions)
    summary = create_summary(sessions)

    helpers = [
        add_variability_features,
        add_daily_features,
        add_account_age,
        add_shared_entity_features,
        add_payment_features,
    ]

    for helper in helpers:
        summary = helper(sessions, summary)

    validate_summary(sessions, summary)

    return summary.sort_values("user_id").reset_index(drop=True)


def main():
    # Resolve paths relative to this file, regardless of terminal location.
    data_dir = Path(__file__).resolve().parent / "data"

    sessions = pd.read_csv(data_dir / "sessions.csv")
    summary = build_features(sessions)

    output_path = data_dir / "features.csv"
    summary.to_csv(output_path, index=False, na_rep="")

    print(summary.head().to_string(index=False))
    print(f"Created {len(summary)} account summaries with {len(summary.columns) - 1} numeric features")
    print(f"Saved to: {output_path}")

    missing = summary.isna().sum()

    if missing.any():
        print("\nDocumented undefined values (blank in CSV):")
        print(missing[missing > 0].to_string())


if __name__ == "__main__":
    main()