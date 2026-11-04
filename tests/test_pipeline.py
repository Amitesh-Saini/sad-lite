# =============================================================================
# SAD-Lite — Pipeline Tests
# =============================================================================
#
# PURPOSE
# Encode the claims this project makes, so that changing a weight, a feature,
# or a generator parameter cannot quietly make one of them untrue.
# These tests are not here to prove the code is correct. They check that the
# statements in the README, the resume, and interview answers still hold.
#
# WHY THIS IS NEEDED
# Every quoted number comes from a pipeline with many tunable parts. Without
# tests, a small edit can break a claim ("the rules catch the whole smurf
# ring", "the incident is invisible at user level") and nothing reports it.
#
# WHAT IT TESTS
# - Data generation: determinism, unique IDs, valid timestamps, labels for
#   every account, the dormitory and innocent lookalikes labelled innocent.
# - Identifiers: no identifier reveals which account type generated it,
#   either by its format or by its numerical order.
# - Features: one row per account, only documented missing values, labels
#   never present, and each abuse pattern visible in the feature it targets.
# - Scoring: bounded scores, a reason for every flagged account, and both
#   detectors far better than random at the review capacity.
# - Project claims: what each detector catches at K = 40, the dormitory
#   held out by the rules, and the shared-IP experiment admitting it.
# - Game incident: the incident itself does not push participants into the
#   user-level top 120 (see THE INCIDENT TEST below).
# - Stability: the model's quoted findings stay within fixed bounds across
#   several forest seeds.
#
# THE INCIDENT TEST
# Incident participants are ordinary users, and some ordinary users are
# legitimately unusual for other reasons. So "no participant is ever in the
# top 120" is the wrong guarantee: it could fail because of a participant's
# unrelated behaviour. The guarantee tested is that the INCIDENT ITSELF does
# not raise suspicion. The dataset is built twice with the same seed, with and
# without the incident (every other session is identical), and the test
# counts participants that enter the top 120 only when the incident is
# present. Incident membership is never used to remove accounts from rankings.
#
# HOW IT RUNS
# Every dataset, feature table, and score is built in memory once, at import.
# Nothing in data/ or reports/ is read or overwritten.
#
# Run:
#     python -m pytest tests/ -v
#     python tests/test_pipeline.py      # works without pytest installed
#
# SOURCES
# Structure and most tests follow the SAD-Lite reference test suite
# (tests/test_pipeline.py). The identifier, lookalike, controlled-incident,
# shared-IP experiment, and seed-stability tests are this project's own.
# The disarm and agent tests are added when disarm_watch.py and agent_sim.py
# are written.
# =============================================================================

import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluate  # noqa: E402
import features  # noqa: E402
import generate_users  # noqa: E402
import score_model  # noqa: E402
import score_rules  # noqa: E402


SEED = 42

# Review capacity used for every claim, read from the same config the pipeline uses.
RULES_CONFIG = score_rules.load_config(ROOT / "config" / "rules.yaml")
EXPERIMENT_CONFIG = score_rules.load_config(ROOT / "config" / "rules_shared_ip_raised.yaml")
K = int(RULES_CONFIG["review"]["capacity"])

# Bounds for the model's findings across forest seeds. Each one is a claim
# made about the model; observed values at seed 42 are in the comments.
# If a bound breaks, the matching claim must be re-checked before quoting it.
STABILITY_SEEDS = range(5)
STABILITY_BOUNDS = {
    # Observed 1.000 at every seed.
    f"precision_at_{K}": (0.95, 1.00),
    # Observed about 0.87–0.92.
    "average_precision": (0.80, 0.97),
    # "The two detectors largely disagree": observed 0.23–0.25.
    f"jaccard_with_rules_at_{K}": (0.15, 0.35),
    # "The model finds the resold accounts the rules miss": observed 24–25 of 25.
    f"account_resale_in_top_{K}": (22, 25),
    # "The model misses the bot farm": observed 0.
    f"bot_farm_in_top_{K}": (0, 2),
    # "The model finds about half the smurf ring": observed 15–16 of 30.
    f"smurf_ring_in_top_{K}": (12, 20),
    # "The model holds the dorm out until large budgets": observed 1–3.
    "dorm_in_top_200": (0, 5),
}

# An identifier read as a number must not rank abusers better than this
# multiple of the base rate (random ranking scores about the base rate).
IDENTIFIER_AP_LIMIT = 2.0

# Fewer than this many participants may enter the top 120 because of the incident.
# Observed: 0 for both detectors. Reintroducing the reference generator's bug
# (fresh identifiers for incident sessions) gives 6, so 3 catches it with margin.
INCIDENT_BUDGET = 120
MAX_INCIDENT_ENTRANTS = 3


# --------------------------------------------------------------------------
# Fixtures, built once
# --------------------------------------------------------------------------

def build_world(include_incident=True):
    """Generate a dataset and its feature table in memory."""

    sessions, labels, incident = generate_users.build_dataset(seed=SEED, include_incident=include_incident)

    session_table = pd.DataFrame(sessions)
    label_table = pd.DataFrame(labels)
    feature_table = features.build_features(session_table)

    return session_table, label_table, incident, feature_table


def score_with_rules(feature_table, config=RULES_CONFIG):
    """Rank accounts with the rules, exactly as score_rules.py does."""

    return score_rules.score_features(feature_table, config)


def score_with_model(feature_table, seed=SEED):
    """Rank accounts with the model, through score_model.py's own validation."""

    # load_features validates a CSV file, so write the table to a temporary one.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "features.csv"
        feature_table.to_csv(path, index=False, na_rep="")
        model_input = score_model.load_features(path)

    anomaly_scores, *_ = score_model.score_accounts(model_input, random_state=seed)

    return evaluate.rank_model_scores(model_input, anomaly_scores), model_input


def with_labels(scores, label_table, score_column):
    """Validate a ranking and join the ground truth, as evaluate.py does."""

    return evaluate.join_labels(evaluate.validate_scores(scores, score_column), label_table)


_sessions, _labels, _incident, _features = build_world()

_rule_scores = score_with_rules(_features)
_model_scores, _model_input = score_with_model(_features)

_rules = with_labels(_rule_scores, _labels, "rule_score")
_model = with_labels(_model_scores, _labels, "anomaly_score")

# The same seed without the incident, for the controlled incident test.
_sessions_clean, _labels_clean, _, _features_clean = build_world(include_incident=False)


def _top(ranked, pattern, k=K):
    """Count accounts of one scenario in the top k of a ranking."""

    return int(ranked.head(k)["pattern"].eq(pattern).sum())


def _total(pattern):
    """Count accounts of one scenario in the labels."""

    return int(_labels["pattern"].eq(pattern).sum())


# --------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------

def test_generation_is_deterministic():
    again, _, _ = generate_users.build_dataset(seed=SEED)
    again = pd.DataFrame(again)

    assert len(again) == len(_sessions)
    assert again["session_uuid"].tolist() == _sessions["session_uuid"].tolist()
    assert again["session_start"].tolist() == _sessions["session_start"].tolist()


def test_session_uuids_are_unique():
    assert _sessions["session_uuid"].is_unique


def test_no_session_precedes_account_creation():
    """A session before the account existed is a generator bug, and an obvious tell."""

    starts = pd.to_datetime(_sessions["session_start"], format="ISO8601", utc=True)
    created = pd.to_datetime(_sessions["account_created_at"], format="ISO8601", utc=True)

    assert (starts >= created).all()


def test_sessions_end_after_they_start():
    starts = pd.to_datetime(_sessions["session_start"], format="ISO8601", utc=True)
    ends = pd.to_datetime(_sessions["session_end"], format="ISO8601", utc=True)

    assert (ends > starts).all()


def test_every_user_has_a_label():
    assert set(_sessions["user_id"]) == set(_labels["user_id"])


def test_population_matches_the_documented_counts():
    """1,980 accounts, 120 abusive: the numbers every precision figure depends on."""

    assert len(_labels) == 1980
    assert int(_labels["label"].sum()) == 120
    assert _labels["pattern"].value_counts().to_dict() == generate_users.ACCOUNT_COUNTS


def test_the_dorm_is_labelled_innocent():
    dorm = _labels[_labels["pattern"] == evaluate.DORM_PATTERN]

    assert len(dorm) == generate_users.ACCOUNT_COUNTS[evaluate.DORM_PATTERN]
    assert dorm["label"].sum() == 0, "the hard negatives must be labelled 0"


def test_dorm_really_does_share_one_ip():
    dorm_users = set(_labels.loc[_labels["pattern"] == evaluate.DORM_PATTERN, "user_id"])

    assert _sessions.loc[_sessions["user_id"].isin(dorm_users), "client_ip"].nunique() == 1


def test_lookalikes_are_labelled_innocent():
    """Lookalikes resemble abuse on purpose; their label must still be innocent."""

    lookalikes = _labels[_labels["pattern"].str.startswith("normal_")]

    assert set(lookalikes["pattern"]) == set(generate_users.LOOKALIKE_COUNTS)
    assert lookalikes["label"].sum() == 0


# --------------------------------------------------------------------------
# Identifiers
# --------------------------------------------------------------------------

def test_identifiers_share_one_format_across_account_types():
    """No identifier format may encode its account type (no "bot_120" user IDs)."""

    formats = {
        "user_id": r"^user_\d{5}$",
        "device_id": r"^device_[0-9a-f]{10}$",
        "payment_hash": r"^payment_[0-9a-f]{10}$",
        "client_ip": r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
        "session_uuid": r"^session_\d{5}_\d{5}$",
    }

    for column, pattern in formats.items():
        mismatched = _sessions.loc[~_sessions[column].str.match(pattern), column]

        assert mismatched.empty, f"{column} values outside the shared format: {mismatched.head(3).tolist()}"


def test_identifier_order_does_not_reveal_abuse():
    """Read as numbers, identifiers must rank abusers no better than chance.

    Catches leaks a format check cannot, such as abusive accounts being given
    consecutive user numbers or IP addresses from one range.
    """

    def ip_number(address):
        first, second, third, fourth = (int(part) for part in address.split("."))
        return ((first * 256 + second) * 256 + third) * 256 + fourth

    first_sessions = _sessions.sort_values("session_start").groupby("user_id").first()
    labels = _labels.set_index("user_id").loc[first_sessions.index, "label"]
    base_rate = labels.mean()

    as_numbers = {
        "user_id": first_sessions.index.str[len("user_"):].astype(int),
        "client_ip": first_sessions["client_ip"].map(ip_number),
        "device_id": first_sessions["device_id"].str[len("device_"):].map(lambda value: int(value, 16)),
        "payment_hash": first_sessions["payment_hash"].str[len("payment_"):].map(lambda value: int(value, 16)),
    }

    for column, values in as_numbers.items():
        values = np.asarray(values, dtype=float)

        # Check both directions: a leak could put abusers at either end.
        best = max(average_precision_score(labels, values), average_precision_score(labels, -values))

        assert best < base_rate * IDENTIFIER_AP_LIMIT, (
            f"{column} order ranks abusers with AP {best:.3f} against a base rate of {base_rate:.3f}"
        )


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def test_one_row_per_user():
    assert len(_features) == _labels["user_id"].nunique()
    assert _features["user_id"].is_unique


def test_only_documented_values_are_missing():
    """Undefined values are allowed only where features.py documents them."""

    allowed = {"std_duration_min", "std_input_rate", "duration_cv", "input_cv", "termination_game_concentration"}
    missing = set(_features.columns[_features.isna().any()])

    assert missing <= allowed, f"undocumented missing values in {sorted(missing - allowed)}"


def test_labels_never_leak_into_features():
    """The one mistake that invalidates everything downstream."""

    leaked = {"label", "pattern"} & set(_features.columns)

    assert not leaked, f"ground truth leaked into features: {leaked}"


def test_smurf_ring_shows_up_as_shared_device():
    smurfs = _features.merge(_labels).query("pattern == 'smurf_ring'")

    # 30 accounts on one device: each sees the other 29.
    assert smurfs["max_other_users_on_device"].min() >= 19


def test_bots_have_low_variance():
    table = _features.merge(_labels)
    bot_cv = table.loc[table["pattern"] == "bot_farm", "duration_cv"].mean()
    normal_cv = table.loc[table["pattern"] == "normal", "duration_cv"].mean()

    assert bot_cv < normal_cv / 3


def test_cheaters_have_high_termination_rate():
    table = _features.merge(_labels)

    assert table.loc[table["pattern"] == "cheater", "termination_rate"].mean() > 0.3


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def test_rule_scores_are_bounded():
    assert _rule_scores["rule_score"].between(0, 1).all()


def test_model_scores_are_bounded():
    assert _model_scores["anomaly_score"].between(0, 1).all()


def test_every_flagged_user_has_a_reason():
    top = _rule_scores.head(50)

    assert not top["evidence_summary"].str.startswith("No enabled scoring rule").any()


def test_rules_beat_random_by_a_mile():
    precision = evaluate.evaluate_top_k(_rules, K)["precision"]
    base_rate = _rules["label"].mean()

    assert precision > base_rate * 5, f"precision@{K} {precision:.2f} vs base rate {base_rate:.3f}"


def test_model_beats_random():
    precision = evaluate.evaluate_top_k(_model, K)["precision"]

    assert precision > _model["label"].mean() * 5


def test_average_precision_is_sane():
    for ranked, column in [(_rules, "rule_score"), (_model, "anomaly_score")]:
        assert 0.0 <= evaluate.calculate_average_precision(ranked, column) <= 1.0


# --------------------------------------------------------------------------
# Project claims at the review capacity
# --------------------------------------------------------------------------

def test_rules_catch_the_ring_and_some_bots_but_no_resale():
    """The rules' half of "two detectors, different abuse"."""

    assert _top(_rules, "smurf_ring") == _total("smurf_ring")
    assert _top(_rules, "bot_farm") >= 8
    assert _top(_rules, "account_resale") == 0


def test_neither_detector_catches_cheaters_at_capacity():
    assert _top(_rules, "cheater") == 0
    assert _top(_model, "cheater") == 0


def test_rules_hold_the_dorm_out():
    """The low shared-IP weight keeps legitimate dorm users away from reviewers."""

    dorm_flagged = _top(_rules, evaluate.DORM_PATTERN, k=200)

    assert dorm_flagged <= 2, f"{dorm_flagged} dorm accounts in the rules' top 200"


def test_raising_shared_ip_weight_lets_the_dorm_in():
    """The Day 5 experiment: the low weight is what holds the dorm out."""

    experiment = with_labels(score_with_rules(_features, EXPERIMENT_CONFIG), _labels, "rule_score")

    baseline_dorm = _top(_rules, evaluate.DORM_PATTERN, k=200)
    raised_dorm = _top(experiment, evaluate.DORM_PATTERN, k=200)

    assert raised_dorm >= baseline_dorm + 30, f"dorm in top 200: {baseline_dorm} -> {raised_dorm}"


# --------------------------------------------------------------------------
# Game incident
# --------------------------------------------------------------------------

def test_the_incident_adds_only_its_own_sessions():
    """The controlled comparison is only fair if nothing else differs."""

    added = len(_sessions) - len(_sessions_clean)

    assert added == _incident["surge_sessions"]
    assert _labels_clean.equals(_labels)

    # Every session in the clean dataset also exists, unchanged, in the full one.
    key = ["user_id", "session_start", "app_id", "end_reason", "client_ip", "device_id"]
    clean = set(map(tuple, _sessions_clean[key].to_numpy()))
    full = set(map(tuple, _sessions[key].to_numpy()))

    assert clean <= full


def test_the_incident_is_invisible_at_user_level():
    """The whole reason the game-level watcher exists.

    Counts participants that are in the top 120 WITH the incident but not
    WITHOUT it, so a participant who is unusual for unrelated reasons does
    not count against the incident.
    """

    participants = set(_incident["participants"])

    clean_rules = score_with_rules(_features_clean)
    clean_model, _ = score_with_model(_features_clean)

    for name, with_incident, without_incident in [
        ("rules", _rule_scores, clean_rules),
        ("model", _model_scores, clean_model),
    ]:
        top_with = set(with_incident.head(INCIDENT_BUDGET)["user_id"])
        top_without = set(without_incident.head(INCIDENT_BUDGET)["user_id"])

        entrants = (participants & top_with) - top_without

        assert len(entrants) < MAX_INCIDENT_ENTRANTS, (
            f"{name}: {len(entrants)} participants entered the top {INCIDENT_BUDGET} "
            f"because of the incident; game-level and user-level views should disagree here"
        )


# --------------------------------------------------------------------------
# Stability across forest seeds
# --------------------------------------------------------------------------

def test_model_findings_stay_within_seed_bounds():
    """The model's quoted findings must not depend on one lucky forest."""

    _, spread = evaluate.seed_stability(_model_input, _labels, _rules, K, list(STABILITY_SEEDS))

    for measure, (low, high) in STABILITY_BOUNDS.items():
        observed_low = spread.loc[measure, "min"]
        observed_high = spread.loc[measure, "max"]

        assert low <= observed_low and observed_high <= high, (
            f"{measure} ranged {observed_low:g}–{observed_high:g} across seeds, outside {low:g}–{high:g}"
        )


# --------------------------------------------------------------------------
# Plain-Python runner, so this works before pytest is installed
# --------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(name, test) for name, test in sorted(globals().items()) if name.startswith("test_") and callable(test)]
    failures = 0

    for name, test in tests:
        try:
            test()
            print(f"  PASS  {name}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {error}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)