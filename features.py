# =============================================================================
# SAD-Lite — Account Feature Extraction
# =============================================================================
#
# Reads data/sessions.csv and writes data/features.csv.
# Converts many session records into one summary row per account.
#
# Features are grouped into families (see FEATURE_FAMILIES):
# - Volume: how much an account plays.
# - Diversity: how many distinct IPs, devices, countries, and games it uses,
#   overall and within a single day.
# - Regularity: how human-shaped its behaviour is (variation in duration and
#   input activity, and the spread of hours it plays).
# - Outcomes: how its sessions end, and whether terminations concentrate on
#   one game.
# - Account age: how old the account was when it first played.
# - Entity graph: how many other accounts share its IPs, devices, and
#   payment identifiers.
#
# Definitions:
# - Days and hours are UTC, assigned using session_start.
# - Account age is measured at the first observed session.
# - Shared-entity counts exclude the account itself.
# - Input mean is an unweighted average of session-level input rates.
# - Standard deviation uses the sample formula (ddof=1).
# - Termination game concentration is the share of an account's terminated
#   sessions that occurred on its most-terminated game.
#
# Undefined values remain blank in the output:
# - Sample standard deviation and CV require at least two sessions.
# - Input CV is undefined when the mean input rate is zero.
# - Termination game concentration is undefined with fewer than three
#   terminations, because one or two cannot show a concentration.
# These require an explicit handling policy before model fitting.
#
# Does not read labels.csv or incident_truth.json.
# user_id identifies rows; it must not be used as a numerical model feature.
# Features describe behaviour and do not establish whether an account is abusive.
# Hours are UTC, not each player's local time; in this synthetic data all
# session times are generated in UTC.
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd


TERMINATED_ENDINGS = ("terminated_anticheat", "terminated_abuse")

# Concentration needs several terminations to mean anything:
# one termination is always "100% on one game".
MIN_TERMINATIONS_FOR_CONCENTRATION = 3

# Output column order, grouped by family.
FEATURE_FAMILIES = {
    "volume": [
        "session_count",
        "active_days",
        "sessions_per_active_day",
        "max_sessions_per_day",
        "total_duration_min",
        "total_bytes_streamed",
    ],
    "diversity": [
        "distinct_devices",
        "distinct_ips",
        "distinct_countries",
        "distinct_games",
        "max_devices_per_day",
        "max_ips_per_day",
        "max_countries_per_day",
    ],
    "regularity": [
        "mean_duration_min",
        "std_duration_min",
        "duration_cv",
        "mean_input_rate",
        "std_input_rate",
        "input_cv",
        "distinct_hours",
    ],
    "outcomes": [
        "termination_count",
        "termination_rate",
        "termination_game_concentration",
    ],
    "account_age": [
        "account_age_at_first_session_days",
    ],
    "entity_graph": [
        "max_other_users_on_ip",
        "max_other_users_on_device",
        "max_other_users_on_payment",
    ],
}

FEATURE_COLUMNS = [column for columns in FEATURE_FAMILIES.values() for column in columns]


def prepare_sessions(sessions):
    """Validate input and convert timestamps and numerical columns."""

    sessions = sessions.copy()

    required = [
        "session_uuid", "user_id", "account_created_at", "session_start",
        "session_end", "duration_min", "client_ip", "country", "device_id",
        "app_id", "input_events_per_min", "bytes_streamed", "payment_hash",
        "end_reason",
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

        item["terminations"] += session.end_reason in TERMINATED_ENDINGS
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


def add_daily_features(sessions, summary):
    """Calculate active days and maximum activity within one UTC start-date."""

    daily_sessions = sessions.assign(session_day=sessions["session_start"].dt.floor("D"))

    daily = daily_sessions.groupby(["user_id", "session_day"]).agg(
        daily_sessions=("session_uuid", "size"),
        daily_devices=("device_id", "nunique"),
        daily_ips=("client_ip", "nunique"),
        daily_countries=("country", "nunique"),
    )

    grouped = daily.groupby(level="user_id")

    summary["active_days"] = summary["user_id"].map(grouped.size())

    # Average sessions on the days the account actually played.
    summary["sessions_per_active_day"] = summary["session_count"] / summary["active_days"]

    # A burst in one day is hidden by a 30-day distinct count;
    # nine IPs over a month differs from nine IPs in one day.
    mappings = [
        ("daily_sessions", "max_sessions_per_day"),
        ("daily_devices", "max_devices_per_day"),
        ("daily_ips", "max_ips_per_day"),
        ("daily_countries", "max_countries_per_day"),
    ]

    for source, target in mappings:
        summary[target] = summary["user_id"].map(grouped[source].max())

    return summary


def add_variability_features(sessions, summary):
    """Measure variation in duration and session-average input activity."""

    grouped = sessions.groupby("user_id")

    summary["std_duration_min"] = summary["user_id"].map(grouped["duration_min"].std(ddof=1))
    summary["std_input_rate"] = summary["user_id"].map(grouped["input_events_per_min"].std(ddof=1))

    summary["duration_cv"] = summary["std_duration_min"] / summary["mean_duration_min"]
    summary["input_cv"] = summary["std_input_rate"] / summary["mean_input_rate"].replace(0, np.nan)

    return summary


def add_time_of_day_features(sessions, summary):
    """Count the distinct UTC hours in which each account starts sessions."""

    # People play at some times of day; a value near 24 means round-the-clock play.
    hours = sessions["session_start"].dt.hour

    summary["distinct_hours"] = summary["user_id"].map(hours.groupby(sessions["user_id"]).nunique())

    return summary


def add_termination_features(sessions, summary):
    """Measure how concentrated each account's terminations are on one game."""

    terminated = sessions[sessions["end_reason"].isin(TERMINATED_ENDINGS)]

    # Terminations on the account's most-terminated game, over all its terminations.
    per_game = terminated.groupby(["user_id", "app_id"]).size()
    top_game = per_game.groupby(level="user_id").max()
    total = terminated.groupby("user_id").size()

    # Too few terminations leaves concentration undefined, so it stays blank.
    concentration = (top_game / total)[total >= MIN_TERMINATIONS_FOR_CONCENTRATION]

    summary["termination_game_concentration"] = summary["user_id"].map(concentration)

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


def validate_summary(sessions, summary):
    """Reject inconsistent outputs and unexplained missing values."""

    if list(summary.columns) != ["user_id"] + FEATURE_COLUMNS:
        raise ValueError("Summary columns do not match FEATURE_FAMILIES")

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
        "termination_game_concentration": summary["termination_count"] < MIN_TERMINATIONS_FOR_CONCENTRATION,
    }

    for column in numeric.columns:
        expected = allowed_missing.get(column, pd.Series(False, index=summary.index))

        if not numeric[column].isna().equals(expected):
            raise ValueError(f"Unexpected missing-value pattern in {column}")

    for column in ["termination_rate", "termination_game_concentration"]:
        if not summary[column].dropna().between(0, 1).all():
            raise ValueError(f"{column} is outside [0, 1]")

    if not summary["distinct_hours"].between(1, 24).all():
        raise ValueError("distinct_hours is outside [1, 24]")

    if (summary["max_ips_per_day"] > summary["distinct_ips"]).any():
        raise ValueError("max_ips_per_day cannot exceed distinct_ips")


def build_features(sessions):
    """Run validation and all feature calculations."""

    sessions = prepare_sessions(sessions)
    summary = create_summary(sessions)

    helpers = [
        add_daily_features,
        add_variability_features,
        add_time_of_day_features,
        add_termination_features,
        add_account_age,
        add_shared_entity_features,
    ]

    for helper in helpers:
        summary = helper(sessions, summary)

    # Arrange columns by family so the output reads in the same order as the docs.
    summary = summary[["user_id"] + FEATURE_COLUMNS]

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

    print("\nFeatures by family:")
    for family, columns in FEATURE_FAMILIES.items():
        print(f"  {family}: {', '.join(columns)}")

    missing = summary.isna().sum()

    if missing.any():
        print("\nDocumented undefined values (blank in CSV):")
        print(missing[missing > 0].to_string())


if __name__ == "__main__":
    main()