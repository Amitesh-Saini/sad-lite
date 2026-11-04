"""
score_model.py

Rank accounts by unusual behaviour using an Isolation Forest model.

This file provides a machine-learning approach that we can compare with
our rule-based detector. Instead of assigning points using explicit rules,
the model looks for unusual combinations of account features.

How Isolation Forest works:
- Pick a random feature, then a random split point between that feature's
  minimum and maximum. Repeat until each account sits alone or a depth
  limit is reached.
- Unusual accounts sit away from the crowd, so random splits separate them
  in few steps. Ordinary accounts sit in dense regions and take many.
- Averaging the isolation depth over many random trees gives the score.
Reference: Liu, Ting & Zhou, "Isolation Forest", ICDM 2008.

What this file does:
- Loads and validates the account summaries from features.csv.
- Selects behaviour features and handles documented missing values.
- Fits Isolation Forest using a fixed random seed for reproducibility.
- Gives each account an anomaly score, with higher scores meaning
  more unusual behaviour.
- Saves ranked accounts to model_scores.csv.
- Saves the settings and feature choices used for the run.

Feature choices:
- Rates, diversity, regularity, outcomes, account age, and the entity
  graph, including accounts sharing an IP address.
- Raw volume (session counts, total time, total bytes) and average levels
  (mean duration, mean input rate) are excluded. They mostly measure how
  much or how intensely someone plays, which varies widely among honest
  players, rather than how strange their behaviour is.
- Features are not standardised. Isolation Forest chooses each split
  uniformly between a feature's minimum and maximum, so rescaling a
  feature does not change which accounts are isolated.

User IDs identify results but are not model inputs. Ground-truth labels,
scenario names, and rule scores are excluded from the model inputs.
The model never sees labels, so it cannot learn that a shared IP is
usually innocent: nobody told it, and it has no labels to learn it from.

Evaluation happens separately in evaluate.py, where we compare the
model's ranking with the known labels and our rule-based results.

An unusual account is not necessarily abusive. Scores are review
priorities, not probabilities or automatic punishment decisions.

Sources: the feature selection follows the SAD-Lite reference
implementation (sadlite/score_model.py), without its standardisation step.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

# Features that may be undefined, and why (see features.py).
OPTIONAL_COLUMNS = ["input_cv", "duration_cv", "termination_game_concentration"]

# Concentration is only defined with at least this many terminations.
MIN_TERMINATIONS_FOR_CONCENTRATION = 3


def load_features(file_path):
    """Load and validate one feature row per account."""

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Feature file not found: {file_path}")

    count_columns = [
        "session_count", "active_days", "max_sessions_per_day",
        "distinct_devices", "distinct_ips", "distinct_countries",
        "distinct_games", "max_devices_per_day", "max_ips_per_day",
        "max_countries_per_day", "distinct_hours", "termination_count",
        "max_other_users_on_ip", "max_other_users_on_device",
        "max_other_users_on_payment",
    ]

    measurement_columns = [
        "sessions_per_active_day", "total_duration_min", "total_bytes_streamed",
        "mean_duration_min", "std_duration_min", "duration_cv",
        "mean_input_rate", "std_input_rate", "input_cv",
        "termination_rate", "termination_game_concentration",
        "account_age_at_first_session_days",
    ]

    numeric_columns = count_columns + measurement_columns
    expected_columns = ["user_id"] + numeric_columns

    data = pd.read_csv(file_path, dtype={"user_id": "string"})

    if data.empty:
        raise ValueError("Feature table contains no accounts.")

    # Only accept the expected feature schema.
    missing_columns = set(expected_columns) - set(data.columns)
    extra_columns = set(data.columns) - set(expected_columns)

    if missing_columns:
        raise ValueError(f"Missing feature columns: {sorted(missing_columns)}")

    if extra_columns:
        raise ValueError(f"Unexpected columns in feature table: {sorted(extra_columns)}")

    # Keep IDs unchanged so they still match the other files.
    if data["user_id"].isna().any():
        raise ValueError("user_id contains missing values.")

    if data["user_id"].str.strip().eq("").any():
        raise ValueError("user_id contains blank values.")

    if data["user_id"].ne(data["user_id"].str.strip()).any():
        raise ValueError("user_id contains leading or trailing spaces.")

    if data["user_id"].duplicated().any():
        raise ValueError("Each user_id must have exactly one feature row.")

    # Reject invalid text instead of silently turning it into NaN.
    for column in numeric_columns:
        try:
            data[column] = pd.to_numeric(data[column], errors="raise")
        except (ValueError, TypeError) as error:
            raise ValueError(f"{column} must contain numeric values.") from error

        if np.isinf(data[column].to_numpy(dtype=float)).any():
            raise ValueError(f"{column} contains infinite values.")

        if data[column].lt(0).any():
            raise ValueError(f"{column} contains negative values.")

    # These features can be undefined under specific conditions.
    optional_columns = {"std_duration_min", "std_input_rate", *OPTIONAL_COLUMNS}

    for column in numeric_columns:
        if column not in optional_columns and data[column].isna().any():
            raise ValueError(f"{column} contains unexpected missing values.")

    for column in count_columns + ["total_bytes_streamed"]:
        if data[column].mod(1).ne(0).any():
            raise ValueError(f"{column} must contain whole numbers.")

    positive_columns = [
        "session_count", "active_days", "sessions_per_active_day",
        "max_sessions_per_day", "distinct_devices", "distinct_ips",
        "distinct_countries", "distinct_games", "max_devices_per_day",
        "max_ips_per_day", "max_countries_per_day", "distinct_hours",
        "total_duration_min", "mean_duration_min",
    ]

    for column in positive_columns:
        if data[column].le(0).any():
            raise ValueError(f"{column} must be greater than zero.")

    # Counts within an account cannot exceed its session count.
    bounded_columns = [
        "active_days", "max_sessions_per_day", "distinct_devices",
        "distinct_ips", "distinct_countries", "distinct_games",
        "distinct_hours", "termination_count",
    ]

    for column in bounded_columns:
        if data[column].gt(data["session_count"]).any():
            raise ValueError(f"{column} cannot exceed session_count.")

    if data["distinct_hours"].gt(24).any():
        raise ValueError("distinct_hours cannot exceed 24.")

    daily_limits = {
        "max_devices_per_day": "distinct_devices",
        "max_ips_per_day": "distinct_ips",
        "max_countries_per_day": "distinct_countries",
    }

    for daily_column, total_column in daily_limits.items():
        if data[daily_column].gt(data[total_column]).any():
            raise ValueError(f"{daily_column} cannot exceed {total_column}.")

        if data[daily_column].gt(data["max_sessions_per_day"]).any():
            raise ValueError(f"{daily_column} cannot exceed max_sessions_per_day.")

    for column in ["max_other_users_on_ip", "max_other_users_on_device", "max_other_users_on_payment"]:
        if data[column].gt(len(data) - 1).any():
            raise ValueError(f"{column} exceeds the number of other accounts.")

    # Check derived values against their underlying counts.
    expected_termination_rate = data["termination_count"] / data["session_count"]
    expected_mean_duration = data["total_duration_min"] / data["session_count"]
    expected_sessions_per_day = data["session_count"] / data["active_days"]

    derived_values = {
        "termination_rate": expected_termination_rate,
        "mean_duration_min": expected_mean_duration,
        "sessions_per_active_day": expected_sessions_per_day,
    }

    for column, expected in derived_values.items():
        if not np.allclose(data[column], expected, rtol=1e-5, atol=1e-8, equal_nan=True):
            raise ValueError(f"{column} does not match its underlying values.")

    # Sample standard deviation is undefined for a single session.
    single_session = data["session_count"].eq(1)

    for column in ["std_duration_min", "std_input_rate"]:
        if not data[column].isna().equals(single_session):
            raise ValueError(f"{column} must be missing only for single-session accounts.")

    expected_duration_cv = data["std_duration_min"] / data["mean_duration_min"]
    expected_input_cv = data["std_input_rate"] / data["mean_input_rate"].replace(0, np.nan)

    for column, expected in {"duration_cv": expected_duration_cv, "input_cv": expected_input_cv}.items():
        if not np.allclose(data[column], expected, rtol=1e-5, atol=1e-8, equal_nan=True):
            raise ValueError(f"{column} does not match standard deviation divided by mean.")

    # Concentration is undefined only when there are too few terminations.
    too_few_terminations = data["termination_count"].lt(MIN_TERMINATIONS_FOR_CONCENTRATION)

    if not data["termination_game_concentration"].isna().equals(too_few_terminations):
        raise ValueError(
            f"termination_game_concentration must be missing only below "
            f"{MIN_TERMINATIONS_FOR_CONCENTRATION} terminations."
        )

    if not data["termination_game_concentration"].dropna().between(0, 1).all():
        raise ValueError("termination_game_concentration must be between 0 and 1.")

    return data[expected_columns].copy()

def select_model_features(data):
    """Select behaviour features without IDs, labels, rule scores, or raw volume."""

    feature_columns = [
        # Rate rather than volume.
        "sessions_per_active_day",
        # Diversity.
        "distinct_ips",
        "distinct_devices",
        "distinct_countries",
        "max_ips_per_day",
        "max_devices_per_day",
        "max_countries_per_day",
        # Regularity.
        "duration_cv",
        "input_cv",
        "distinct_hours",
        # Outcomes.
        "termination_rate",
        "termination_game_concentration",
        # Account age.
        "account_age_at_first_session_days",
        # Entity graph, including the IP-sharing feature behind the dorm problem.
        "max_other_users_on_device",
        "max_other_users_on_payment",
        "max_other_users_on_ip",
    ]

    missing_columns = set(feature_columns) - set(data.columns)

    if missing_columns:
        raise ValueError(f"Missing model features: {sorted(missing_columns)}")

    # Explicit selection prevents unrelated columns entering the model.
    model_features = data[feature_columns].copy()

    return model_features


def handle_missing_values(model_features, fill_values=None):
    """Fill documented missing values and preserve indicators of missingness."""

    if model_features.empty:
        raise ValueError("Cannot prepare an empty feature table.")

    prepared = model_features.copy()

    for column in OPTIONAL_COLUMNS:
        if column not in prepared.columns:
            raise ValueError(f"Missing required feature: {column}")

    # All other features must already have valid values.
    for column in prepared.columns:
        if not pd.api.types.is_numeric_dtype(prepared[column]):
            raise ValueError(f"{column} must contain numeric values.")

        if np.isinf(prepared[column].to_numpy(dtype=float)).any():
            raise ValueError(f"{column} contains infinite values.")

        if column not in OPTIONAL_COLUMNS and prepared[column].isna().any():
            raise ValueError(f"{column} contains unexpected missing values.")

    # Learn replacement values only when preparing training data.
    # The median of the defined values is a neutral, typical value;
    # the indicator column keeps the fact that the value was undefined.
    if fill_values is None:
        fill_values = {}

        for column in OPTIONAL_COLUMNS:
            available_values = prepared[column].dropna()

            if available_values.empty:
                raise ValueError(f"Cannot calculate a median: all values in {column} are missing.")

            fill_values[column] = float(available_values.median())
    else:
        fill_values = dict(fill_values)

        if set(fill_values) != set(OPTIONAL_COLUMNS):
            raise ValueError(f"fill_values must contain exactly: {OPTIONAL_COLUMNS}")

    # Check supplied replacements before applying them.
    for column, value in fill_values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"Replacement for {column} must be numeric.")

        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Replacement for {column} must be finite and nonnegative.")

        fill_values[column] = float(value)

    # Always add all indicators so the model's columns stay consistent.
    # An indicator that is always 0 (for example when every account has
    # enough sessions for a CV) cannot be split on, so it does no harm.
    for column in OPTIONAL_COLUMNS:
        prepared[f"{column}_missing"] = prepared[column].isna().astype(int)
        prepared[column] = prepared[column].fillna(fill_values[column])

    if not np.isfinite(prepared.to_numpy(dtype=float)).all():
        raise ValueError("Prepared features still contain missing or infinite values.")

    return prepared, fill_values


def fit_isolation_forest(prepared_features, random_state=42, n_estimators=300):
    """Fit Isolation Forest using the prepared numeric features."""

    if len(prepared_features) < 2:
        raise ValueError("At least two accounts are needed to fit the model.")

    if prepared_features.shape[1] == 0:
        raise ValueError("The model needs at least one feature.")

    if not np.isfinite(prepared_features.to_numpy(dtype=float)).all():
        raise ValueError("Model features must not contain missing or infinite values.")

    # max_samples="auto" uses min(256, number of accounts) per tree, as in the original paper.
    model = IsolationForest(n_estimators=n_estimators, max_samples="auto", contamination="auto", max_features=1.0, bootstrap=False, n_jobs=1, random_state=random_state, verbose=0, warm_start=False)

    # Build the trees using account features, without ground-truth labels.
    model.fit(prepared_features)

    return model


def calculate_anomaly_scores(model, prepared_features):
    """Calculate anomaly scores, where higher means more unusual."""

    if prepared_features.empty:
        raise ValueError("Cannot score an empty feature table.")

    if not np.isfinite(prepared_features.to_numpy(dtype=float)).all():
        raise ValueError("Model features must not contain missing or infinite values.")

    # Reverse the sign so more unusual accounts receive higher scores.
    scores = -model.score_samples(prepared_features)

    if not np.isfinite(scores).all():
        raise ValueError("The model produced invalid anomaly scores.")

    return pd.Series(scores, index=prepared_features.index, name="anomaly_score")

def rank_and_save_scores(data, anomaly_scores, output_path):
    """Rank accounts by anomaly score and save the results to CSV."""

    if data.empty:
        raise ValueError("Cannot rank an empty account table.")

    if "user_id" not in data.columns:
        raise ValueError("Account table must contain user_id.")

    # Matching indexes prevent scores being assigned to the wrong accounts.
    if not data.index.is_unique or not anomaly_scores.index.is_unique:
        raise ValueError("Account and score indexes must be unique.")

    if not data.index.equals(anomaly_scores.index):
        raise ValueError("Account rows and anomaly scores must have matching indexes and order.")

    if data["user_id"].isna().any() or data["user_id"].duplicated().any():
        raise ValueError("User IDs must be present and unique.")

    if not np.isfinite(anomaly_scores.to_numpy(dtype=float)).all():
        raise ValueError("Anomaly scores must be finite numbers.")

    ranked = data[["user_id"]].copy()
    ranked["anomaly_score"] = anomaly_scores

    # The hash only breaks ties; it is not evidence of suspicious behaviour.
    ranked["tie_breaker"] = ranked["user_id"].map(lambda user_id: hashlib.sha256(user_id.encode("utf-8")).hexdigest())

    ranked = ranked.sort_values(["anomaly_score", "tie_breaker", "user_id"], ascending=[False, True, True]).reset_index(drop=True)
    ranked = ranked.drop(columns="tie_breaker")
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_path, index=False)

    return ranked

def score_accounts(data, random_state=42, n_estimators=300):
    """Prepare features, fit the model, and return scores with the settings used."""

    model_features = select_model_features(data)
    prepared_features, fill_values = handle_missing_values(model_features)

    model = fit_isolation_forest(prepared_features, random_state=random_state, n_estimators=n_estimators)
    anomaly_scores = calculate_anomaly_scores(model, prepared_features)

    return anomaly_scores, model, model_features, prepared_features, fill_values

def display_results(ranked, output_path, top_n=15):
    """Display the highest-ranked accounts and the output location."""

    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 0:
        raise ValueError("top_n must be a nonnegative integer.")

    shown = min(top_n, len(ranked))

    if shown:
        print(f"\nTop {shown} most unusual accounts:")
        print(ranked.head(shown).to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    print(f"\nScored {len(ranked)} accounts.")
    print(f"Ranking saved to: {output_path}")
    print("Higher anomaly scores mean more unusual behaviour.")
    print("Scores are not probabilities of abuse or enforcement decisions.")

def main():
    """Load account features, fit Isolation Forest, and save ranked scores."""

    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    reports_dir = project_dir / "reports"

    parser = argparse.ArgumentParser(description="Rank accounts with an Isolation Forest.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the forest; record it in any write-up")
    parser.add_argument("--output", type=Path, default=data_dir / "model_scores.csv", help="where to save the ranking")
    parser.add_argument("--top", type=int, default=15, help="how many accounts to print")
    args = parser.parse_args()

    features_path = data_dir / "features.csv"
    settings_path = reports_dir / f"model_settings_{args.output.stem}.json"

    # Load features and prepare numeric model inputs.
    data = load_features(features_path)

    # Fit the model and score the same accounts for our initial comparison.
    anomaly_scores, model, model_features, prepared_features, fill_values = score_accounts(data, random_state=args.seed)
    ranked = rank_and_save_scores(data, anomaly_scores, args.output)

    # Record the settings and preprocessing choices used for this run.
    settings = {
        "model_type": "IsolationForest",
        "model_parameters": model.get_params(),
        "sklearn_version": sklearn.__version__,
        "random_seed": args.seed,
        "account_count": len(data),
        "selected_features": model_features.columns.tolist(),
        "prepared_features": prepared_features.columns.tolist(),
        "missing_value_replacements": fill_values,
        "standardised": False,
        "score_definition": "-model.score_samples(prepared_features)",
        "higher_scores_mean": "More unusual behaviour",
        "evaluation_scope": "Model fitted and scored on the same synthetic accounts.",
    }

    reports_dir.mkdir(parents=True, exist_ok=True)

    with settings_path.open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=4, allow_nan=False)

    display_results(ranked, args.output, top_n=args.top)

    print(f"Model settings saved to: {settings_path}")
    print("Initial comparison: fitted and scored on the same synthetic dataset.")


if __name__ == "__main__":
    main()