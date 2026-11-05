# =============================================================================
# SAD-Lite — Subject Investigation
# =============================================================================
#
# PURPOSE
# Turn one identifier into a case file a reviewer can read, check, and defend.
# Given a user ID, session ID, device ID, payment identifier, or IP address,
# assemble what the session log shows about it, follow its connections a
# bounded distance, place each observation in context, and offer hypotheses
# with their alternatives. The tool presents evidence. It decides nothing.
#
# WHY THIS IS NEEDED
# A detector puts an account at the top of a list; it does not tell a person
# what to do. A reviewer needs the facts behind the score, the innocent
# explanations that fit the same facts, and enough references to check every
# claim themselves, both now and when an appeal is reviewed later.
#
# WHAT IT READS (and nothing else)
# Only these published stages, each verified by provenance.py:
# - generate/sessions.csv: required. If it cannot be verified, the tool
#   writes a diagnostic explaining why, not a case report.
# - features/features.csv: optional, for population percentiles.
# - rules/rule_scores.csv and model/model_scores.csv: optional detector scores.
# Each optional stage is used only if it verifies AND was built from the
# verified upstream file. Otherwise it is excluded from every observation and
# hypothesis and listed, with the reason, in the provenance section.
# Every input must have exactly its expected columns, so a copied label
# column (label, pattern, is_bot, ...) is rejected rather than read.
# The tool never opens data/truth/: labels and incident ground truth are for
# evaluation only. A test confirms the report is identical with truth removed.
# Provenance detects accidental mixing; it does not detect deliberate
# rewriting of data/ (see provenance.py, TRUST MODEL).
#
# HOW AN INVESTIGATION RUNS
# 1. Resolve the subject. A session widens to its account. A device, payment,
#    or IP stays an entity: it may belong to many accounts, and choosing one
#    would choose the conclusion.
# 2. Collect facts from the subject's own sessions: volume, variability,
#    diversity, terminations, overlaps, rapid country changes.
# 3. Follow connections one step: the subject's devices and payments first,
#    then IPs. Strong identifiers come first so that a busy shared address
#    cannot fill the search before a shared device is examined. Each
#    identifier type has its own listing allowance. Widely shared identifiers
#    are summarised over all their accounts; only the allowance is listed.
#    The report always states where the search stopped and what it left out.
# 4. Add context: percentiles among all accounts, and whether each
#    termination happened while that game's termination rate was elevated,
#    computed from the session log itself.
# 5. Build an event timeline from recorded fields only.
# 6. Offer hypotheses, each with supporting observations, competing innocent
#    explanations, the check made against each, a confidence, and its basis.
#
# CONFIDENCE
# Confidence is a qualitative judgement with a stated basis, never a
# probability. It depends on whether each competing innocent explanation was
# contradicted by a specific check in the data, or merely remains possible.
# Evidence is grouped into families (entity graph, timing, geography,
# outcomes, regularity, account age) to organise the reasoning. Families are
# not statistically independent: one household can produce shared payments,
# devices, and IPs at once. Each hypothesis therefore names its most plausible
# common-cause innocent explanation explicitly.
# - Strong: the main innocent explanations were each contradicted by a check.
# - Moderate: some, but not all, were contradicted.
# - Weak: the pattern is present, but innocent explanations remain open.
# - Insufficient: too few observations to judge.
#
# TRACEABILITY AND REPRODUCIBILITY
# Every observation cites its supporting session IDs (up to a limit, with the
# total count), states how it was calculated, and names its evidence family.
# Population claims cite the features file hash and population size.
# The report records the subject, the input hashes, the settings, this
# script's version and source hash, and library versions. Sorting, rounding,
# and formatting are fixed, so the same inputs, settings, and code produce
# the same report. The generation time is kept in a separate metadata block,
# outside the reproducible report.
#
# UNTRUSTED TEXT
# Every string in the session log (game names, identifiers, countries) comes
# from clients in a real system. In the markdown report, control characters
# are replaced with visible escapes so text cannot hide or overwrite lines in
# a terminal. That does not stop readable text that tries to instruct an AI
# reader; the JSON marks these fields as data, and any agent consuming reports
# must treat them as evidence only and act through restricted tools.
#
# OUTPUT
# reports/cases/<case_id>.md for people and reports/cases/<case_id>.json for
# software. Nothing in data/ is modified.
#
# Run:
#     python investigate.py --user-id user_00042
#     python investigate.py --session-id session_00042_00003
#     python investigate.py --device-id device_3fa94c1e0b
#     python investigate.py --payment-id payment_8c01d2e9aa
#     python investigate.py --ip 10.12.34.56
#
# SOURCES
# The idea of a pivoting case file follows the SAD-Lite reference
# implementation (sadlite/investigate.py). Provenance checks, pivot ordering
# and allowances, the confidence scheme, the incident check, and
# sanitisation are this project's own.
# =============================================================================

import argparse
import hashlib
import json
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import features
import generate_users
import provenance
import score_rules


INVESTIGATOR_VERSION = "1.0.0"
REPORT_SCHEMA_VERSION = 1

# The only stages and files this tool may read.
INPUT_STAGES = {
    "generate": ["sessions.csv"],
    "features": ["features.csv"],
    "rules": ["rule_scores.csv", "rules_config.yaml"],
    "model": ["model_scores.csv", "model_settings.json"],
}

# Exact columns each input must have. Anything else, such as a copied label
# column, makes the input unacceptable.
EXPECTED_COLUMNS = {
    "sessions.csv": list(generate_users.SESSION_FIELDS),
    "features.csv": ["user_id"] + list(features.FEATURE_COLUMNS),
    "rule_scores.csv": [
        "rank", "user_id", "rule_score", "review_recommended", "triggered_rules",
        "evidence_summary", "caveats", "evidence_json",
    ] + [f"signal_{rule}" for rule in score_rules.RULE_NAMES],
    "model_scores.csv": ["rank", "user_id", "anomaly_score"],
}

# Investigation settings. All are recorded in every report.
SETTINGS = {
    # Strong identifiers before weak ones.
    "pivot_order": ["device_id", "payment_hash", "client_ip"],
    # Search depth: the subject's identifiers and the accounts on them.
    # Linked accounts' other identifiers are not followed.
    "search_depth": 1,
    # How many of the subject's identifiers of each type are examined.
    "identifiers_examined_per_type": {"device_id": 10, "payment_hash": 10, "client_ip": 10},
    # How many linked accounts are listed per identifier; all are summarised.
    "accounts_listed_per_identifier": {"device_id": 10, "payment_hash": 10, "client_ip": 5},
    # Session IDs cited per observation; the total is always stated.
    "session_ids_cited": 10,
    "timeline_events_shown": 60,
    # A country change between sessions starting less than this many hours apart.
    "rapid_change_hours": 6,
    # Game-wide check: a termination counts as during an elevated period if
    # the game's termination rate in a window centred on it is at least
    # elevated_rate_ratio times the game's whole-dataset rate, over at least
    # elevated_min_sessions sessions.
    "incident_window_hours": 6,
    "elevated_rate_ratio": 5.0,
    "elevated_min_sessions": 20,
    # Reporting conventions, not mathematical requirements.
    "min_sessions_for_variability": 2,
    "sessions_for_reliable_variability": 10,
    # Hypothesis triggers, matching the rules' trigger values where they overlap.
    "device_link_min_other_accounts": 3,
    "payment_link_min_other_accounts": 2,
    # Assumptions used to test the household explanation.
    "household_max_accounts": 5,
    "coordinated_creation_days": 7,
    "automation_max_duration_cv": 0.10,
    "automation_max_input_cv": 0.05,
    "round_the_clock_hours": 18,
    "cheating_min_terminations": 3,
    "cheating_min_rate": 0.15,
    "shared_network_min_other_accounts": 10,
    "vpn_exit_min_other_accounts": 5,
}

# Features reported as population percentiles, with their family.
PERCENTILE_FEATURES = {
    "session_count": "volume",
    "distinct_devices": "diversity",
    "distinct_ips": "diversity",
    "max_ips_per_day": "diversity",
    "max_countries_per_day": "geography",
    "duration_cv": "regularity",
    "input_cv": "regularity",
    "distinct_hours": "timing",
    "termination_rate": "outcomes",
    "account_age_at_first_session_days": "account_age",
    "max_other_users_on_device": "entity_graph",
    "max_other_users_on_payment": "entity_graph",
    "max_other_users_on_ip": "entity_graph",
}

IDENTIFIER_NAMES = {"device_id": "device", "payment_hash": "payment identifier", "client_ip": "IP address"}

# Fields whose values come from the session log, and so from clients.
UNTRUSTED_FIELDS = ["session_uuid", "user_id", "client_ip", "country", "device_id", "app_id", "game_name", "payment_hash"]

TERMINATED_ENDINGS = ("terminated_anticheat", "terminated_abuse")


# ------------------------------------------------------------
# Untrusted text and formatting.
# ------------------------------------------------------------

def clean_text(value):
    """Replace control and format characters with visible escapes."""

    text = "" if value is None else str(value)

    # Category C covers control, format (e.g. bidirectional overrides), and unassigned.
    return "".join(
        f"\\u{ord(character):04x}" if unicodedata.category(character).startswith("C") else character
        for character in text
    )


def md(value):
    """Prepare untrusted text for a markdown table or line."""

    return clean_text(value).replace("|", "\\|").replace("`", "'")


def plural(count, singular, plural_form=None):
    """'1 device', '3 devices'."""

    return f"{count} {singular if count == 1 else (plural_form or singular + 's')}"


def stamp(value):
    """Format a timestamp the same way everywhere."""

    if value is None or pd.isna(value):
        return "unavailable"

    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M UTC")


def number(value, digits=3):
    """Round a number for JSON, keeping None for undefined values."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    return round(float(value), digits)


def cite(session_ids, limit=SETTINGS["session_ids_cited"]):
    """Cite up to `limit` session IDs, in a stable order, with the total."""

    ordered = sorted(set(session_ids))

    return {"session_ids": ordered[:limit], "total_sessions": len(ordered)}


# ------------------------------------------------------------
# Loading verified inputs.
# ------------------------------------------------------------

def check_columns(frame, file_name):
    """Require exactly the expected columns, in order."""

    if list(frame.columns) != EXPECTED_COLUMNS[file_name]:
        unexpected = sorted(set(frame.columns) - set(EXPECTED_COLUMNS[file_name]))
        missing = sorted(set(EXPECTED_COLUMNS[file_name]) - set(frame.columns))
        raise provenance.ProvenanceError(
            f"{file_name} does not have the expected columns (unexpected: {unexpected}, missing: {missing})"
        )


def stage_record(stage, manifest, contents, status, reason=None):
    """Describe one stage for the provenance section."""

    record = {"stage": stage, "status": status, "reason": reason}

    if manifest is not None:
        record["files"] = {name: provenance.sha256_bytes(data) for name, data in sorted(contents.items())}
        record["inputs"] = manifest.get("inputs", {})
        record["code"] = manifest.get("code")
        record["software"] = manifest.get("software")

    return record


def load_inputs(data_dir):
    """Load and verify every input stage, deciding which may be used."""

    data_dir = Path(data_dir)
    records = []

    # Sessions are required; without them there is nothing verified to report.
    try:
        manifest, contents = provenance.load_stage(data_dir, "generate", INPUT_STAGES["generate"])
        sessions = provenance.frame_from_bytes(contents["sessions.csv"], dtype=str, keep_default_na=False)
        check_columns(sessions, "sessions.csv")
    except provenance.ProvenanceError as error:
        records.append(stage_record("generate", None, {}, "failed", str(error)))
        return {"sessions": None, "provenance": records}

    sessions_digest = provenance.sha256_bytes(contents["sessions.csv"])
    records.append(stage_record("generate", manifest, contents, "verified"))

    inputs = {"sessions": sessions, "sessions_sha256": sessions_digest, "features": None, "rules": None, "model": None}

    # Features: verified, and built from these sessions.
    features_digest = None

    try:
        manifest, contents = provenance.load_stage(data_dir, "features", INPUT_STAGES["features"])
        provenance.check_input(manifest, "sessions", "generate", "sessions.csv", sessions_digest)
        feature_table = provenance.frame_from_bytes(contents["features.csv"], dtype={"user_id": str})
        check_columns(feature_table, "features.csv")
        features_digest = provenance.sha256_bytes(contents["features.csv"])
        inputs["features"] = feature_table
        inputs["features_sha256"] = features_digest
        records.append(stage_record("features", manifest, contents, "verified"))
    except provenance.ProvenanceError as error:
        records.append(stage_record("features", None, {}, "excluded", str(error)))

    # Scores: verified, and built from the verified features.
    for stage, file_name in [("rules", "rule_scores.csv"), ("model", "model_scores.csv")]:
        try:
            if features_digest is None:
                raise provenance.ProvenanceError("features stage is not verified, so the scores' origin cannot be checked")

            manifest, contents = provenance.load_stage(data_dir, stage, INPUT_STAGES[stage])
            provenance.check_input(manifest, "features", "features", "features.csv", features_digest)
            scores = provenance.frame_from_bytes(contents[file_name], dtype={"user_id": str})
            check_columns(scores, file_name)
            inputs[stage] = scores
            records.append(stage_record(stage, manifest, contents, "verified"))
        except provenance.ProvenanceError as error:
            records.append(stage_record(stage, None, {}, "excluded", str(error)))

    inputs["provenance"] = records

    return inputs


def prepare_sessions(raw):
    """Convert the verified session strings into typed columns."""

    sessions = raw.copy()

    for column, parsed in [("session_start", "start"), ("session_end", "end"), ("account_created_at", "created")]:
        # An empty or unparsable timestamp becomes unavailable, never a guess.
        sessions[parsed] = pd.to_datetime(sessions[column].replace("", None), format="ISO8601", utc=True, errors="coerce")

    sessions["duration"] = pd.to_numeric(sessions["duration_min"], errors="coerce")
    sessions["input_rate"] = pd.to_numeric(sessions["input_events_per_min"], errors="coerce")
    sessions["terminated"] = sessions["end_reason"].isin(TERMINATED_ENDINGS)
    sessions["country_known"] = sessions["country"].str.strip().ne("")

    return sessions.sort_values(["start", "session_uuid"], kind="mergesort").reset_index(drop=True)


# ------------------------------------------------------------
# Observations about one account.
# ------------------------------------------------------------

class Observations:
    """Collects observations and gives each a stable ID."""

    def __init__(self):
        self.items = []

    def add(self, family, kind, statement, calculation, session_ids=(), value=None, details=None):
        observation = {
            "id": f"obs-{len(self.items) + 1:02d}",
            "family": family,
            "kind": kind,
            "statement": statement,
            "calculation": calculation,
            "value": value,
            **cite(session_ids),
        }

        if details is not None:
            observation["details"] = details

        self.items.append(observation)

        return observation["id"]


def game_windows(sessions):
    """Index each game's session starts and terminations for window checks."""

    index = {}

    for app_id, group in sessions.groupby("app_id", sort=True):
        group = group.sort_values("start", kind="mergesort")
        starts = group["start"].to_numpy(dtype="datetime64[ns]")
        terminated = group["terminated"].to_numpy()

        index[app_id] = {
            "starts": starts,
            "cumulative": np.concatenate([[0], np.cumsum(terminated)]),
            "overall_rate": float(terminated.mean()) if len(terminated) else 0.0,
        }

    return index


def termination_context(session, windows):
    """Compare a game's termination rate around one session with its overall rate."""

    game = windows[session["app_id"]]
    half = np.timedelta64(int(SETTINGS["incident_window_hours"] * 30), "m")
    centre = np.datetime64(session["start"].tz_convert(None), "ns")

    low = np.searchsorted(game["starts"], centre - half, side="left")
    high = np.searchsorted(game["starts"], centre + half, side="left")

    window_sessions = int(high - low)
    window_terminated = int(game["cumulative"][high] - game["cumulative"][low])
    window_rate = window_terminated / window_sessions if window_sessions else 0.0

    elevated = (
        window_sessions >= SETTINGS["elevated_min_sessions"]
        and window_rate >= SETTINGS["elevated_rate_ratio"] * max(game["overall_rate"], 1e-9)
    )

    return {
        "session_id": session["session_uuid"],
        "app_id": session["app_id"],
        "window_sessions": window_sessions,
        "window_terminated": window_terminated,
        "window_rate": number(window_rate),
        "game_overall_rate": number(game["overall_rate"]),
        "elevated": bool(elevated),
    }


def overlapping_pairs(first, second, same_frame=False):
    """Find pairs of sessions whose time ranges overlap (start_A < end_B and start_B < end_A)."""

    pairs = []
    rows_a = first[["session_uuid", "start", "end", "device_id", "user_id"]].to_dict("records")
    rows_b = second[["session_uuid", "start", "end", "device_id", "user_id"]].to_dict("records")

    for i, a in enumerate(rows_a):
        candidates = rows_b[i + 1:] if same_frame else rows_b

        for b in candidates:
            if pd.isna(a["end"]) or pd.isna(b["end"]):
                continue

            if a["start"] < b["end"] and b["start"] < a["end"]:
                pairs.append((a, b))

    return pairs


def account_observations(account, sessions, windows, observations):
    """Record the facts about one account's own sessions."""

    own = sessions[sessions["user_id"] == account]
    ids = own["session_uuid"].tolist()
    facts = {"session_count": len(own)}

    # --- Account age ------------------------------------------------------
    created_values = own["created"].dropna().unique()

    if len(created_values) == 1:
        facts["created"] = pd.Timestamp(created_values[0])
        age_days = (own["start"].min() - facts["created"]).total_seconds() / 86400
        facts["age_at_first_session_days"] = age_days
        observations.add(
            "account_age", "account_age",
            f"Account created {stamp(facts['created'])} (as recorded); {age_days:.1f} days old at its first session.",
            "first session_start minus the recorded account_created_at",
            ids[:1], number(age_days),
        )
    else:
        facts["created"] = None
        facts["age_at_first_session_days"] = None
        observations.add(
            "account_age", "account_age",
            "Account creation time unavailable or inconsistent; account age is not reported.",
            "account_created_at must have exactly one recorded value", ids[:1],
        )

    # --- Volume -----------------------------------------------------------
    hours = float(own["duration"].sum()) / 60
    active_days = own["start"].dt.floor("D").nunique()
    facts["active_days"] = active_days

    observations.add(
        "volume", "volume",
        f"{len(own)} sessions on {active_days} days between {stamp(own['start'].min())} and "
        f"{stamp(own['start'].max())}, {hours:.1f} hours in total.",
        "count of sessions; sum of duration_min / 60; distinct UTC start dates",
        ids, {"sessions": len(own), "hours": number(hours, 1), "active_days": active_days},
    )

    # --- Regularity ---------------------------------------------------------
    for column, label, key in [("duration", "Session-duration", "duration_cv"), ("input_rate", "Input-rate", "input_cv")]:
        values = own[column].dropna()

        if len(values) < SETTINGS["min_sessions_for_variability"] or values.mean() == 0:
            facts[key] = None
            observations.add(
                "regularity", key, f"{label} variability: insufficient observations ({len(values)} sessions).",
                "coefficient of variation needs at least two sessions and a nonzero mean", ids,
            )
            continue

        cv = float(values.std(ddof=1) / values.mean())
        facts[key] = cv
        caution = ""

        if len(values) < SETTINGS["sessions_for_reliable_variability"]:
            caution = f" Based on only {len(values)} sessions, below the {SETTINGS['sessions_for_reliable_variability']} this report treats as reliable."

        observations.add(
            "regularity", key, f"{label} variability (CV) is {cv:.3f} across {len(values)} sessions.{caution}",
            "sample standard deviation / mean (ddof=1)", ids, number(cv),
        )

    hours_played = own["start"].dt.hour.nunique()
    facts["distinct_hours"] = hours_played

    observations.add(
        "timing", "distinct_hours", f"Sessions started in {hours_played} different UTC hours of the day.",
        "distinct UTC hour of session_start", ids, hours_played,
    )

    # --- Diversity and geography ----------------------------------------------
    known = own[own["country_known"]]
    unknown_location = len(own) - len(known)
    facts["countries"] = sorted(known["country"].unique())
    facts["devices"] = own["device_id"].nunique()

    location_note = ""

    if unknown_location:
        location_note = f" Location unavailable for {unknown_location} sessions; those are not counted as any country."

    observations.add(
        "diversity", "diversity",
        f"{plural(facts['devices'], 'device')}, {plural(own['client_ip'].nunique(), 'IP address', 'IP addresses')}, "
        f"{plural(len(facts['countries']), 'country', 'countries')} ({', '.join(md(country) for country in facts['countries'])}).{location_note}",
        "distinct device_id, client_ip, and country (blank country = unavailable)",
        ids, {"devices": facts["devices"], "ips": own["client_ip"].nunique(),
              "countries": len(facts["countries"]), "location_unavailable": unknown_location},
    )

    # Country changes between consecutive sessions that start close together.
    rapid = []
    ordered = known.sort_values(["start", "session_uuid"], kind="mergesort").to_dict("records")

    for previous, current in zip(ordered, ordered[1:]):
        gap_hours = (current["start"] - previous["start"]).total_seconds() / 3600

        if current["country"] != previous["country"] and gap_hours < SETTINGS["rapid_change_hours"]:
            rapid.append({
                "from_session": previous["session_uuid"], "to_session": current["session_uuid"],
                "from_country": previous["country"], "to_country": current["country"],
                "gap_hours": number(gap_hours, 2),
            })

    facts["rapid_changes"] = rapid

    if rapid:
        observations.add(
            "geography", "rapid_country_change",
            f"{len(rapid)} country changes between sessions starting less than {SETTINGS['rapid_change_hours']} hours apart.",
            "consecutive sessions (by start) with different known countries and start gap below the limit",
            [item["from_session"] for item in rapid] + [item["to_session"] for item in rapid],
            len(rapid), details=rapid[:SETTINGS["session_ids_cited"]],
        )

    # Overlapping sessions on this account.
    pairs = overlapping_pairs(own, own, same_frame=True)
    different_devices = [(a, b) for a, b in pairs if a["device_id"] != b["device_id"]]
    facts["overlaps"] = pairs
    facts["overlaps_different_devices"] = different_devices

    if pairs:
        observations.add(
            "timing", "account_overlap",
            f"{len(pairs)} pairs of this account's sessions overlap in time, {len(different_devices)} of them on different devices.",
            "session pairs with start_A < end_B and start_B < end_A",
            [a["session_uuid"] for a, _ in pairs] + [b["session_uuid"] for _, b in pairs],
            len(pairs),
            details=[{"first": a["session_uuid"], "second": b["session_uuid"], "different_devices": a["device_id"] != b["device_id"]}
                     for a, b in pairs[:SETTINGS["session_ids_cited"]]],
        )

    # --- Outcomes ---------------------------------------------------------------
    terminated = own[own["terminated"]]
    rate = len(terminated) / len(own) if len(own) else 0.0
    facts["terminations"] = len(terminated)
    facts["termination_rate"] = rate

    contexts = [termination_context(row, windows) for row in terminated.to_dict("records")]
    facts["termination_contexts"] = contexts
    facts["terminated_games"] = terminated["game_name"].value_counts().sort_index().to_dict()

    if len(terminated):
        games = ", ".join(f"{md(game)} ({count})" for game, count in sorted(facts["terminated_games"].items()))
        observations.add(
            "outcomes", "terminations",
            f"{len(terminated)} of {len(own)} sessions ({rate:.1%}) ended in anti-cheat or abuse termination, on: {games}.",
            "end_reason in terminated_anticheat / terminated_abuse",
            terminated["session_uuid"].tolist(), {"terminations": len(terminated), "rate": number(rate)},
        )

        elevated = [item for item in contexts if item["elevated"]]
        observations.add(
            "outcomes", "termination_context",
            f"{len(elevated)} of {len(terminated)} terminations happened while that game's termination rate was elevated "
            f"(at least {SETTINGS['elevated_rate_ratio']:g}× its whole-dataset rate over at least "
            f"{SETTINGS['elevated_min_sessions']} sessions in a {SETTINGS['incident_window_hours']}-hour window).",
            "per termination: the game's terminations / sessions starting within the centred window, from the session log",
            [item["session_id"] for item in contexts], len(elevated), details=contexts[:SETTINGS["session_ids_cited"]],
        )
    else:
        observations.add(
            "outcomes", "terminations", f"No terminations in {len(own)} sessions.",
            "end_reason in terminated_anticheat / terminated_abuse", ids, {"terminations": 0, "rate": 0.0},
        )

    return own, facts


def percentile_observations(account, feature_table, features_digest, observations):
    """Place the account's features among all accounts, citing the population."""

    row = feature_table[feature_table["user_id"] == account]

    if row.empty:
        return {}

    row = row.iloc[0]
    percentiles = {}

    for column, family in PERCENTILE_FEATURES.items():
        values = pd.to_numeric(feature_table[column], errors="coerce")
        defined = values.dropna()
        value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]

        if pd.isna(value) or defined.empty:
            percentiles[column] = {"value": None, "percentile": None, "population": len(defined)}
            continue

        # Mid-rank percentile: accounts below, plus half of those tied.
        below = int((defined < value).sum())
        tied = int((defined == value).sum())
        percentile = 100 * (below + 0.5 * tied) / len(defined)

        percentiles[column] = {"value": number(value), "percentile": number(percentile, 1), "population": len(defined), "family": family}

    observations.add(
        "context", "percentiles",
        f"Feature percentiles among {len(feature_table)} accounts (see the Context table).",
        "mid-rank percentile: (accounts below + half of ties) / accounts with a defined value",
        (), details={"features_sha256": features_digest, "population": len(feature_table), "percentiles": percentiles},
    )

    return percentiles


def score_summary(account, inputs):
    """Report each verified detector's score for the account, with its origin."""

    summary = {}

    for stage, column in [("rules", "rule_score"), ("model", "anomaly_score")]:
        table = inputs.get(stage)

        if table is None:
            summary[stage] = None
            continue

        row = table[table["user_id"] == account]

        if row.empty:
            summary[stage] = None
            continue

        row = row.iloc[0]
        entry = {"score": number(float(row[column]), 4), "rank": int(row["rank"]), "of": len(table)}

        if stage == "rules":
            entry["triggered_rules"] = clean_text(row["triggered_rules"]) if isinstance(row["triggered_rules"], str) else ""
            entry["reasons"] = clean_text(row["evidence_summary"])

        summary[stage] = entry

    return summary


# ------------------------------------------------------------
# Connections.
# ------------------------------------------------------------

def identifier_summary(identifier_type, value, sessions, subject_accounts, inputs):
    """Summarise every account on one identifier, listing only the allowance."""

    on_identifier = sessions[sessions[identifier_type] == value]
    members = sorted(on_identifier["user_id"].unique())
    others = [account for account in members if account not in subject_accounts]

    # Creation-date spread across all accounts on this identifier.
    created = sessions[sessions["user_id"].isin(members)].groupby("user_id")["created"].first().dropna()
    spread_days = (created.max() - created.min()).total_seconds() / 86400 if len(created) > 1 else None

    # How strongly each other account is linked: which identifier types it shares with the subject.
    subject_rows = sessions[sessions["user_id"].isin(subject_accounts)]
    subject_values = {kind: set(subject_rows[kind]) for kind in SETTINGS["pivot_order"]}

    member_rows = sessions[sessions["user_id"].isin(others)]
    links = []

    # For an identifier investigated on its own there is no subject account, so
    # links are measured between the accounts on it: a value of another type
    # used by two or more of them.
    if not subject_accounts:
        for kind in SETTINGS["pivot_order"]:
            if kind == identifier_type:
                continue

            users_per_value = member_rows.groupby(kind)["user_id"].nunique()
            subject_values[kind] = set(users_per_value[users_per_value >= 2].index)

    for account, group in member_rows.groupby("user_id", sort=True):
        shared = [kind for kind in SETTINGS["pivot_order"] if set(group[kind]) & subject_values[kind]]
        links.append({"user_id": account, "shared_types": shared, "sessions_on_identifier": int((on_identifier["user_id"] == account).sum())})

    # Strongest links first, then account ID, so the listing is stable.
    # For an identifier subject, "shared_types" lists the other identifier types
    # an account shares with the other accounts on it.
    links.sort(key=lambda item: (-len(item["shared_types"]), item["user_id"]))

    allowance = SETTINGS["accounts_listed_per_identifier"][identifier_type]
    listed = links[:allowance]

    for item in listed:
        account_created = created.get(item["user_id"])
        item["created"] = stamp(account_created) if account_created is not None else "unavailable"

        for stage in ["rules", "model"]:
            table = inputs.get(stage)
            match = None if table is None else table[table["user_id"] == item["user_id"]]
            item[f"{stage}_rank"] = int(match.iloc[0]["rank"]) if match is not None and not match.empty else None

    # Accounts linked by something other than this identifier.
    multi_link = sum(1 for item in links if any(kind != identifier_type for kind in item["shared_types"]))

    return {
        "identifier_type": identifier_type,
        "value": value,
        "accounts": len(members),
        "other_accounts": len(others),
        "sessions": int(len(on_identifier)),
        "first_use": stamp(on_identifier["start"].min()),
        "last_use": stamp(on_identifier["start"].max()),
        "creation_spread_days": number(spread_days, 1),
        "others_sharing_more_than_this_identifier": multi_link,
        "listed": listed,
        "not_listed": max(0, len(links) - allowance),
        "session_ids": cite(on_identifier["session_uuid"])["session_ids"],
    }


def follow_connections(subject_accounts, sessions, inputs):
    """Follow the subject's identifiers one step, strongest identifier types first."""

    subject_rows = sessions[sessions["user_id"].isin(subject_accounts)]
    connections = []
    stopped = []

    for identifier_type in SETTINGS["pivot_order"]:
        # Examine identifiers in the order the subject first used them.
        values = subject_rows.drop_duplicates(identifier_type)[identifier_type].tolist()
        limit = SETTINGS["identifiers_examined_per_type"][identifier_type]

        for value in values[:limit]:
            connections.append(identifier_summary(identifier_type, value, sessions, subject_accounts, inputs))

        if len(values) > limit:
            stopped.append(f"{len(values) - limit} further {IDENTIFIER_NAMES[identifier_type]}s of the subject were not examined (limit {limit}).")

    for item in connections:
        if item["not_listed"]:
            stopped.append(
                f"{IDENTIFIER_NAMES[item['identifier_type']]} {md(item['value'])}: {item['other_accounts']} other accounts, "
                f"{len(item['listed'])} listed, {item['not_listed']} summarised but not listed."
            )

    stopped.append("Search depth 1: linked accounts' own other identifiers were not followed.")

    return connections, stopped


def strongest(connections, identifier_type):
    """The identifier of one type shared with the most other accounts."""

    of_type = [item for item in connections if item["identifier_type"] == identifier_type]

    return max(of_type, key=lambda item: (item["other_accounts"], item["value"]), default=None)


# ------------------------------------------------------------
# Timeline.
# ------------------------------------------------------------

def build_timeline(own, facts):
    """Events from recorded fields only, in time order."""

    events = []

    if facts.get("created") is not None:
        events.append((facts["created"], 0, "account created (as recorded)", None))

    first = own.iloc[0]
    last = own.iloc[-1]
    events.append((first["start"], 1, f"first session: {md(first['game_name'])}", first["session_uuid"]))

    seen = {"device_id": set(), "client_ip": set(), "country": set()}

    for row in own.to_dict("records"):
        for column, label in [("device_id", "device"), ("client_ip", "IP"), ("country", "country")]:
            value = row[column]

            if column == "country" and not row["country_known"]:
                continue

            if value not in seen[column]:
                seen[column].add(value)

                if len(seen[column]) > 1:
                    events.append((row["start"], 2, f"first use of {label} {md(value)}", row["session_uuid"]))

        if row["terminated"]:
            events.append((row["start"], 3, f"{row['end_reason']} on {md(row['game_name'])}", row["session_uuid"]))

    for item in facts["rapid_changes"]:
        session = own[own["session_uuid"] == item["to_session"]].iloc[0]
        events.append((session["start"], 4, f"country {md(item['from_country'])} → {md(item['to_country'])} after {item['gap_hours']} h", item["to_session"]))

    for a, b in facts["overlaps"]:
        later = max(a, b, key=lambda row: (row["start"], row["session_uuid"]))
        events.append((later["start"], 5, f"overlaps session {md(min(a['session_uuid'], b['session_uuid']))}", later["session_uuid"]))

    events.append((last["start"], 6, "last session in the data", last["session_uuid"]))

    events.sort(key=lambda event: (event[0], event[1], event[3] or ""))

    limit = SETTINGS["timeline_events_shown"]
    shown = [{"time": stamp(time), "event": text, "session_id": session_id} for time, _, text, session_id in events[:limit]]

    return {"events": shown, "total_events": len(events), "not_shown": max(0, len(events) - limit)}


# ------------------------------------------------------------
# Hypotheses.
# ------------------------------------------------------------

def judge(alternatives):
    """Turn the result of each check against an innocent explanation into a confidence."""

    contradicted = sum(1 for item in alternatives if item["result"] == "contradicted")

    if contradicted == len(alternatives):
        return "Strong", "every competing innocent explanation was contradicted by a check in the data"

    if contradicted:
        return "Moderate", f"{contradicted} of {len(alternatives)} competing innocent explanations were contradicted; the rest remain possible"

    return "Weak", "the pattern is present, but no competing innocent explanation was contradicted"


def find_observation(observations, kind):
    return next((item["id"] for item in observations.items if item["kind"] == kind), None)


def account_hypotheses(facts, connections, observations, percentiles):
    """Offer hypotheses with alternatives, checks, confidence, and basis."""

    hypotheses = []
    device = strongest(connections, "device_id")
    payment = strongest(connections, "payment_hash")
    ip = strongest(connections, "client_ip")

    # --- Coordinated multi-accounting -------------------------------------
    if (device and device["other_accounts"] >= SETTINGS["device_link_min_other_accounts"]
            and payment and payment["other_accounts"] >= SETTINGS["payment_link_min_other_accounts"]):

        group_size = max(device["accounts"], payment["accounts"])
        spread = device["creation_spread_days"]

        alternatives = [
            {
                "explanation": "A household or family sharing a console and a payment card.",
                "check": f"More than {SETTINGS['household_max_accounts']} accounts on the shared identifiers (an assumption about household size)",
                "result": "contradicted" if group_size > SETTINGS["household_max_accounts"] else "not contradicted",
                "observed": f"{group_size} accounts",
            },
            {
                "explanation": "Unrelated people who joined at different times and happen to share equipment.",
                "check": f"Accounts on the shared device created within {SETTINGS['coordinated_creation_days']} days of each other",
                "result": "contradicted" if spread is not None and spread <= SETTINGS["coordinated_creation_days"] else "not contradicted",
                "observed": f"creation dates span {spread} days" if spread is not None else "creation dates unavailable",
            },
        ]

        confidence, basis = judge(alternatives)

        hypotheses.append({
            "name": "Coordinated multi-accounting",
            "statement": (
                f"One device is shared with {device['other_accounts']} other accounts and one payment identifier with "
                f"{payment['other_accounts']}. No single link is conclusive; the combination with the checks below is the case."
            ),
            "families": ["entity_graph", "account_age"],
            "supporting_observations": [item for item in [find_observation(observations, "account_age")] if item],
            "common_cause_alternative": alternatives[0]["explanation"],
            "alternatives": alternatives,
            "confidence": confidence,
            "confidence_basis": basis,
            "next_check": "Review whether the linked accounts play at the same times from the shared device, and compare their billing details.",
        })

    # --- Account sharing or resale ------------------------------------------
    countries = len(facts["countries"])
    overlaps_devices = len(facts["overlaps_different_devices"])

    if countries >= 2 and (facts["rapid_changes"] or facts["devices"] >= 3 or overlaps_devices):
        new_country_sessions = {item["to_session"] for item in facts["rapid_changes"]}
        own_devices = facts["devices"]

        vpn_like = ip is not None and ip["other_accounts"] >= SETTINGS["vpn_exit_min_other_accounts"] and ip["others_sharing_more_than_this_identifier"] == 0

        alternatives = [
            {
                "explanation": "One person travelling with their own device.",
                "check": "Sessions on different devices overlap in time, or three or more devices are used",
                "result": "contradicted" if overlaps_devices or own_devices >= 3 else "not contradicted",
                "observed": f"{overlaps_devices} overlapping pairs on different devices; {own_devices} devices",
            },
            {
                "explanation": "One person using a VPN that exits in another country.",
                "check": "The account's most-shared IP is not a shared exit (shared with few unlinked accounts)",
                "result": "not contradicted" if vpn_like else "contradicted",
                "observed": (
                    f"most-shared IP has {ip['other_accounts']} other accounts, {ip['others_sharing_more_than_this_identifier']} also linked otherwise"
                    if ip else "no IP information"
                ),
            },
        ]

        confidence, basis = judge(alternatives)

        hypotheses.append({
            "name": "Account sharing or resale",
            "statement": (
                f"The account was used from {countries} countries, with {len(facts['rapid_changes'])} rapid country changes, "
                f"{own_devices} devices, and {overlaps_devices} overlapping pairs on different devices. "
                f"A shared household or travel remains plausible unless the checks below rule them out."
            ),
            "families": ["geography", "diversity", "timing"],
            "supporting_observations": [item for item in [find_observation(observations, "diversity"),
                                                          find_observation(observations, "rapid_country_change"),
                                                          find_observation(observations, "account_overlap")] if item],
            "common_cause_alternative": alternatives[0]["explanation"],
            "alternatives": alternatives,
            "confidence": confidence,
            "confidence_basis": basis,
            "next_check": "Check whether the new devices first appear in the same period as the new countries, and whether the account's owner reported travel.",
        })

    # --- Automation -----------------------------------------------------------
    duration_cv = facts.get("duration_cv")
    input_cv = facts.get("input_cv")
    reliable = facts["session_count"] >= SETTINGS["sessions_for_reliable_variability"]

    duration_low = duration_cv is not None and duration_cv <= SETTINGS["automation_max_duration_cv"]
    input_low = input_cv is not None and input_cv <= SETTINGS["automation_max_input_cv"]

    if duration_low or input_low:
        alternatives = [
            {
                "explanation": "A routine player or a platform session limit producing consistent session lengths.",
                "check": "Input activity is also unusually consistent (a routine fixes length, not moment-to-moment play)",
                "result": "contradicted" if duration_low and input_low else "not contradicted",
                "observed": f"duration CV {number(duration_cv)}, input CV {number(input_cv)}",
            },
            {
                "explanation": "A heavy human player, such as a shift worker, playing at unusual hours.",
                "check": f"Play spans at least {SETTINGS['round_the_clock_hours']} hours of the day while staying consistent",
                "result": "contradicted" if facts["distinct_hours"] >= SETTINGS["round_the_clock_hours"] and duration_low and input_low else "not contradicted",
                "observed": f"{facts['distinct_hours']} distinct hours",
            },
        ]

        confidence, basis = judge(alternatives)

        if not reliable:
            confidence, basis = "Insufficient", f"only {facts['session_count']} sessions; variability is not reliable below {SETTINGS['sessions_for_reliable_variability']}"

        hypotheses.append({
            "name": "Automation",
            "statement": "Session length and/or input activity are far more consistent than human play usually is.",
            "families": ["regularity", "timing"],
            "supporting_observations": [item for item in [find_observation(observations, "duration_cv"),
                                                          find_observation(observations, "input_cv"),
                                                          find_observation(observations, "distinct_hours")] if item],
            "common_cause_alternative": alternatives[0]["explanation"],
            "alternatives": alternatives,
            "confidence": confidence,
            "confidence_basis": basis,
            "next_check": "Inspect per-input timing inside sessions; session averages cannot show it.",
        })

    # --- Cheating, and the game-incident explanation ----------------------------
    contexts = facts["termination_contexts"]
    elevated = sum(1 for item in contexts if item["elevated"])

    if facts["terminations"]:
        game_names = ", ".join(md(game) for game in sorted(facts["terminated_games"]))

        if elevated:
            share = elevated / facts["terminations"]
            hypotheses.append({
                "name": "Caught in a game-wide termination spike",
                "statement": (
                    f"{elevated} of {facts['terminations']} terminations happened while the game's termination rate was elevated. "
                    f"That points to a game-level event (a cheat release, a bad patch, or an anti-cheat fault), not to this account."
                ),
                "families": ["outcomes"],
                "supporting_observations": [item for item in [find_observation(observations, "termination_context")] if item],
                "common_cause_alternative": "The account cheated during the same period as many others.",
                "alternatives": [{
                    "explanation": "The account cheated during the same period as many others.",
                    "check": "The account also has terminations outside elevated periods",
                    "result": "contradicted" if elevated == facts["terminations"] else "not contradicted",
                    "observed": f"{facts['terminations'] - elevated} terminations outside elevated periods",
                }],
                "confidence": "Strong" if share == 1 else "Moderate",
                "confidence_basis": "all terminations fall in elevated game-wide periods" if share == 1 else "some terminations fall outside elevated periods",
                "next_check": "Check the game-level monitor's alert for this window.",
            })

        outside = facts["terminations"] - elevated
        outside_rate = outside / facts["session_count"]

        if outside >= SETTINGS["cheating_min_terminations"] and outside_rate >= SETTINGS["cheating_min_rate"]:
            concentrated = len(facts["terminated_games"]) <= 2
            high = percentiles.get("termination_rate", {}).get("percentile")

            alternatives = [
                {
                    "explanation": "Anti-cheat false positives hitting an honest player repeatedly.",
                    "check": "Terminations are concentrated on one or two games (false positives tend to follow whatever the player plays)",
                    "result": "contradicted" if concentrated else "not contradicted",
                    "observed": f"terminations on: {game_names}",
                },
                {
                    "explanation": "A game-wide fault or incident.",
                    "check": "Terminations happened outside elevated game-wide periods",
                    "result": "contradicted",
                    "observed": f"{outside} of {facts['terminations']} terminations outside elevated periods",
                },
            ]

            confidence, basis = judge(alternatives)

            hypotheses.append({
                "name": "Cheating",
                "statement": (
                    f"{outside} terminations outside any game-wide spike ({outside_rate:.1%} of sessions)"
                    + (f", above {high}% of accounts." if high is not None else ".")
                ),
                "families": ["outcomes"],
                "supporting_observations": [item for item in [find_observation(observations, "terminations"),
                                                              find_observation(observations, "termination_context")] if item],
                "common_cause_alternative": alternatives[0]["explanation"],
                "alternatives": alternatives,
                "confidence": confidence,
                "confidence_basis": basis,
                "next_check": "Ask the anti-cheat team what triggered each termination and whether the detections were confirmed.",
            })

    # --- Shared network (an innocent explanation in its own right) -----------------
    # Offered only when most accounts on the address share nothing else with this
    # account; otherwise the coordinated-accounts hypothesis covers the pattern.
    if (ip is not None and ip["other_accounts"] >= SETTINGS["shared_network_min_other_accounts"]
            and ip["others_sharing_more_than_this_identifier"] <= ip["other_accounts"] / 2):
        hypotheses.append({
            "name": "Shared network",
            "statement": (
                f"IP address {md(ip['value'])} was used by {ip['accounts']} accounts between {ip['first_use']} and {ip['last_use']}; "
                f"{ip['others_sharing_more_than_this_identifier']} of the others also share a device or payment with this account. "
                f"Few other links fit a dormitory, office, café, carrier-grade NAT, or a VPN exit."
            ),
            "families": ["entity_graph"],
            "supporting_observations": [],
            "common_cause_alternative": "Not applicable: this is itself the innocent explanation for a shared IP.",
            "alternatives": [{
                "explanation": "A ring of accounts behind one address.",
                "check": "The other accounts on the IP share devices or payments with this account",
                "result": "contradicted" if ip["others_sharing_more_than_this_identifier"] == 0 else "not contradicted",
                "observed": f"{ip['others_sharing_more_than_this_identifier']} of {ip['other_accounts']} share more than the IP",
            }],
            "confidence": "Strong" if ip["others_sharing_more_than_this_identifier"] == 0 else "Weak",
            "confidence_basis": "no other account on the IP shares a device or payment with this account" if ip["others_sharing_more_than_this_identifier"] == 0 else "some accounts on the IP share more than the address",
            "next_check": "Check whether the address belongs to a known institution, carrier, or VPN provider.",
        })

    if not hypotheses:
        hypotheses.append({
            "name": "Nothing stands out",
            "statement": "No hypothesis condition was met. Closing a case as 'no action' is a valid result.",
            "families": [],
            "supporting_observations": [],
            "common_cause_alternative": None,
            "alternatives": [],
            "confidence": "Not applicable",
            "confidence_basis": "no pattern to assess",
            "next_check": "None required unless new evidence arrives.",
        })

    return hypotheses


def entity_hypotheses(summary):
    """Hypotheses for a device, payment, or IP subject."""

    name = IDENTIFIER_NAMES[summary["identifier_type"]]
    accounts = summary["accounts"]
    spread = summary["creation_spread_days"]

    if accounts < 2:
        return [{
            "name": "Single account",
            "statement": f"Only one account used this {name}.",
            "families": ["entity_graph"], "supporting_observations": [], "common_cause_alternative": None,
            "alternatives": [], "confidence": "Not applicable", "confidence_basis": "no shared use to assess",
            "next_check": "Investigate the account itself.",
        }]

    creation_check = {
        "explanation": "Unrelated people who joined at different times.",
        "check": f"Accounts created within {SETTINGS['coordinated_creation_days']} days of each other",
        "result": "contradicted" if spread is not None and spread <= SETTINGS["coordinated_creation_days"] else "not contradicted",
        "observed": f"creation dates span {spread} days" if spread is not None else "creation dates unavailable",
    }

    if summary["identifier_type"] == "client_ip":
        # For an address, the common innocent cause is translation: many unrelated
        # people behind one public IP.
        linked = summary["others_sharing_more_than_this_identifier"]
        alternatives = [
            {
                "explanation": "Network address translation (dormitory, office, café, carrier-grade NAT, VPN exit).",
                "check": "Accounts on the address are also linked by devices or payments",
                "result": "contradicted" if linked > summary["other_accounts"] / 2 else "not contradicted",
                "observed": f"{linked} of {summary['accounts']} accounts on it linked by more than the address",
            },
            creation_check,
        ]
    else:
        alternatives = [
            {
                "explanation": "A household or family sharing equipment or a payment card.",
                "check": f"More than {SETTINGS['household_max_accounts']} accounts (an assumption about household size)",
                "result": "contradicted" if accounts > SETTINGS["household_max_accounts"] else "not contradicted",
                "observed": f"{accounts} accounts",
            },
            creation_check,
        ]

    confidence, basis = judge(alternatives)

    hypotheses = [{
        "name": "Coordinated accounts",
        "statement": f"This {name} was used by {accounts} accounts between {summary['first_use']} and {summary['last_use']}.",
        "families": ["entity_graph", "account_age"],
        "supporting_observations": [],
        "common_cause_alternative": alternatives[0]["explanation"],
        "alternatives": alternatives,
        "confidence": confidence,
        "confidence_basis": basis,
        "next_check": "Investigate the most strongly linked accounts individually.",
    }]

    # For an address that links almost nothing else, state the innocent reading directly.
    if summary["identifier_type"] == "client_ip" and summary["others_sharing_more_than_this_identifier"] <= accounts / 2:
        linked = summary["others_sharing_more_than_this_identifier"]
        hypotheses.append({
            "name": "Shared network",
            "statement": (
                f"IP address {md(summary['value'])} was used by {accounts} accounts between {summary['first_use']} and "
                f"{summary['last_use']}; {linked} of them share a device or payment with another account on it. "
                f"That fits a dormitory, office, café, carrier-grade NAT, or VPN exit."
            ),
            "families": ["entity_graph"],
            "supporting_observations": [],
            "common_cause_alternative": "Not applicable: this is itself the innocent explanation for a shared IP.",
            "alternatives": [{
                "explanation": "A ring of accounts behind one address.",
                "check": "Accounts on the address share devices or payments with each other",
                "result": "contradicted" if linked == 0 else "not contradicted",
                "observed": f"{linked} of {accounts} accounts linked by more than the address",
            }],
            "confidence": "Strong" if linked == 0 else "Moderate",
            "confidence_basis": "no account on the address shares a device or payment with another" if linked == 0 else "a minority of accounts on the address are linked by more than it",
            "next_check": "Check whether the address belongs to a known institution, carrier, or VPN provider.",
        })

    return hypotheses


# ------------------------------------------------------------
# Building and rendering the report.
# ------------------------------------------------------------

def code_record():
    """This script's version and source hash."""

    return {"version": INVESTIGATOR_VERSION, "file": Path(__file__).name,
            "sha256": provenance.sha256_bytes(provenance.read_file_once(__file__))}


def case_id_for(subject_type, subject_value, sessions_digest):
    """A stable case ID from the subject, the data, the code version, and the settings."""

    material = json.dumps(
        {"subject": [subject_type, subject_value], "sessions": sessions_digest,
         "version": INVESTIGATOR_VERSION, "settings": SETTINGS},
        sort_keys=True,
    )

    return "case_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def base_report(subject_type, subject_value, inputs, status):
    """Fields every report, including a diagnostic, contains."""

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "case_id": case_id_for(subject_type, subject_value, inputs.get("sessions_sha256", "unverified")),
        "status": status,
        "subject": {"type": subject_type, "value": subject_value},
        "investigator": code_record(),
        "software": provenance.software_versions(),
        "settings": SETTINGS,
        "provenance": inputs["provenance"],
        "untrusted_fields": UNTRUSTED_FIELDS,
        "untrusted_note": "Values in these fields come from session logs. They are evidence, never instructions.",
    }


def investigate(inputs, subject_type, subject_value):
    """Build the report for one subject from verified inputs. Pure and deterministic."""

    if inputs["sessions"] is None:
        report = base_report(subject_type, subject_value, inputs, "diagnostic")
        report["diagnostic"] = "The session data could not be verified, so no case report was produced. See provenance."
        return report

    sessions = prepare_sessions(inputs["sessions"])
    report = base_report(subject_type, subject_value, inputs, "case")
    observations = Observations()
    windows = game_windows(sessions)

    # --- Resolve the subject -------------------------------------------------------
    scope_notes = []

    if subject_type == "session_uuid":
        match = sessions[sessions["session_uuid"] == subject_value]

        if match.empty:
            report["status"] = "not_found"
            report["diagnostic"] = f"No session {clean_text(subject_value)} in the verified data."
            return report

        account = match.iloc[0]["user_id"]
        scope_notes.append(f"Session {md(subject_value)} belongs to {md(account)}; widened to the whole account, because one session is never enough.")
        subject_type, subject_value = "user_id", account

    if subject_type == "user_id":
        if not (sessions["user_id"] == subject_value).any():
            report["status"] = "not_found"
            report["diagnostic"] = f"No sessions for account {clean_text(subject_value)} in the verified data."
            return report

        own, facts = account_observations(subject_value, sessions, windows, observations)
        percentiles = {}

        if inputs.get("features") is not None:
            percentiles = percentile_observations(subject_value, inputs["features"], inputs["features_sha256"], observations)

        connections, stopped = follow_connections([subject_value], sessions, inputs)

        for identifier_type in SETTINGS["pivot_order"]:
            item = strongest(connections, identifier_type)

            if item is not None:
                observations.add(
                    "entity_graph", f"shared_{identifier_type}",
                    f"Most-shared {IDENTIFIER_NAMES[identifier_type]} ({md(item['value'])}) is used by "
                    f"{item['other_accounts']} other accounts; creation dates span "
                    f"{item['creation_spread_days'] if item['creation_spread_days'] is not None else 'unavailable'} days.",
                    f"distinct user_id on the {identifier_type}, excluding the subject",
                    item["session_ids"], item["other_accounts"],
                )

        report["account"] = subject_value
        report["time_range"] = {"first_session": stamp(own["start"].min()), "last_session": stamp(own["start"].max())}
        report["scores"] = score_summary(subject_value, inputs)
        report["percentiles"] = percentiles
        report["connections"] = connections
        report["timeline"] = build_timeline(own, facts)
        report["hypotheses"] = account_hypotheses(facts, connections, observations, percentiles)

    elif subject_type in SETTINGS["pivot_order"]:
        if not (sessions[subject_type] == subject_value).any():
            report["status"] = "not_found"
            report["diagnostic"] = f"No sessions with {subject_type} {clean_text(subject_value)} in the verified data."
            return report

        summary = identifier_summary(subject_type, subject_value, sessions, [], inputs)
        allowance = SETTINGS["accounts_listed_per_identifier"][subject_type]
        stopped = [f"{summary['accounts']} accounts used this {IDENTIFIER_NAMES[subject_type]}; {len(summary['listed'])} listed, "
                   f"{summary['not_listed']} summarised but not listed (allowance {allowance}).",
                   "Search depth 1: the listed accounts' own identifiers were not followed; investigate them individually."]

        on_identifier = sessions[sessions[subject_type] == subject_value]
        observations.add(
            "entity_graph", "identifier_use",
            f"{IDENTIFIER_NAMES[subject_type].capitalize()} {md(subject_value)} was used by {summary['accounts']} accounts in "
            f"{summary['sessions']} sessions between {summary['first_use']} and {summary['last_use']}.",
            f"distinct user_id and session count where {subject_type} matches",
            on_identifier["session_uuid"], summary["accounts"],
        )

        report["time_range"] = {"first_session": summary["first_use"], "last_session": summary["last_use"]}
        report["connections"] = [summary]
        report["hypotheses"] = entity_hypotheses(summary)

    else:
        raise ValueError(f"Unsupported subject type: {subject_type}")

    report["scope"] = {"notes": scope_notes, "search_stopped": stopped}
    report["observations"] = observations.items
    report["limitations"] = [
        "Synthetic data: results describe this generated world, not real players.",
        "Input activity is a per-session average; button-level timing is not available.",
        "Hours are UTC, not each player's local time.",
        "Provenance detects accidental mixing of runs, not deliberate rewriting of data/.",
        "Detector scores are review priorities; an anomaly score is not a probability of abuse.",
    ]

    return report


def render_markdown(report):
    """Render the report for a person. Contains no generation time, so it is reproducible."""

    lines = [f"# Case {report['case_id']}", ""]
    subject = report["subject"]
    lines.append(f"**Subject:** {subject['type']} `{md(subject['value'])}`  ")
    lines.append(f"**Status:** {report['status']}  ")
    lines.append(f"**Investigator:** version {report['investigator']['version']}, source sha256 {report['investigator']['sha256'][:16]}…")

    if report.get("time_range"):
        lines.append(f"**Investigated range:** {report['time_range']['first_session']} to {report['time_range']['last_session']}")

    lines += ["", "## Provenance", "", "| stage | status | detail |", "|---|---|---|"]

    for record in report["provenance"]:
        detail = record["reason"] or ", ".join(f"{name} {digest[:12]}…" for name, digest in record.get("files", {}).items())
        lines.append(f"| {record['stage']} | {record['status']} | {md(detail)} |")

    lines += ["", "Excluded stages are not used in any observation or hypothesis below."]

    if report["status"] != "case":
        lines += ["", "## Diagnostic", "", md(report.get("diagnostic", ""))]
        return "\n".join(lines) + "\n"

    if report.get("scope"):
        lines += ["", "## Scope and where the search stopped", ""]
        lines += [f"- {note}" for note in report["scope"]["notes"] + report["scope"]["search_stopped"]]

    lines += ["", "## Facts", "", "Each fact cites its sessions; totals are given when only some are listed.", ""]

    for item in report["observations"]:
        if item["kind"] == "percentiles":
            lines.append(f"- **{item['id']}** [context] {item['statement']}")
            lines.append(f"  - calculation: {item['calculation']}")
            continue

        shown = ", ".join(md(session_id) for session_id in item["session_ids"])
        more = f" (+{item['total_sessions'] - len(item['session_ids'])} more)" if item["total_sessions"] > len(item["session_ids"]) else ""
        lines.append(f"- **{item['id']}** [{item['family']}] {item['statement']}")
        lines.append(f"  - calculation: {item['calculation']}")

        if shown:
            lines.append(f"  - sessions: {shown}{more}")

    scores = report.get("scores")

    if scores is not None:
        lines += ["", "## Detector scores", ""]

        for stage, label in [("rules", "Rule score"), ("model", "Isolation Forest anomaly score")]:
            entry = scores.get(stage)

            if entry is None:
                lines.append(f"- {label}: not available (stage excluded or account not scored).")
                continue

            lines.append(f"- {label}: {entry['score']} (rank {entry['rank']} of {entry['of']}), from the verified `{stage}` stage.")

            if stage == "rules" and entry.get("reasons"):
                lines.append(f"  - reasons: {md(entry['reasons'])}")

        lines.append("- Scores are review priorities. A high anomaly score means unusual behaviour; it is not a probability of abuse.")

    if report.get("percentiles"):
        population = next(item for item in report["observations"] if item["kind"] == "percentiles")["details"]
        lines += ["", "## Context: compared with all accounts", "",
                  f"Population: {population['population']} accounts in features.csv (sha256 {population['features_sha256'][:12]}…). "
                  "Mid-rank percentile: share of accounts below, plus half of ties.", "",
                  "| feature | value | percentile |", "|---|---|---|"]

        for column, entry in report["percentiles"].items():
            value = "undefined" if entry["value"] is None else entry["value"]
            percentile = "—" if entry["percentile"] is None else f"{entry['percentile']}"
            lines.append(f"| {column} | {value} | {percentile} |")

    lines += ["", "## Connections", ""]

    for item in report.get("connections", []):
        name = IDENTIFIER_NAMES[item["identifier_type"]]
        lines.append(f"### {name} `{md(item['value'])}`")
        lines.append(
            f"{item['accounts']} accounts, {item['sessions']} sessions, {item['first_use']} to {item['last_use']}; "
            f"creation dates span {item['creation_spread_days'] if item['creation_spread_days'] is not None else 'unavailable'} days; "
            f"{item['others_sharing_more_than_this_identifier']} other accounts are linked by more than this {name}."
        )

        if item["listed"]:
            lines += ["", "| account | shared with subject | sessions here | created | rules rank | model rank |", "|---|---|---|---|---|---|"]

            for row in item["listed"]:
                shared = ", ".join(row["shared_types"]) or "—"
                lines.append(
                    f"| {md(row['user_id'])} | {shared} | {row['sessions_on_identifier']} | {row['created']} | "
                    f"{row['rules_rank'] if row['rules_rank'] is not None else 'n/a'} | {row['model_rank'] if row['model_rank'] is not None else 'n/a'} |"
                )

        if item["not_listed"]:
            lines.append(f"\n{item['not_listed']} further accounts summarised above but not listed.")

        lines.append("")

    if report.get("timeline"):
        timeline = report["timeline"]
        lines += ["## Timeline", "", "Events from recorded fields only.", ""]

        for event in timeline["events"]:
            reference = f" ({md(event['session_id'])})" if event["session_id"] else ""
            lines.append(f"- {event['time']}: {event['event']}{reference}")

        if timeline["not_shown"]:
            lines.append(f"- … {timeline['not_shown']} further events not shown (limit {SETTINGS['timeline_events_shown']}).")

    lines += ["", "## Hypotheses", "", "Facts are above. These are possible explanations, each with the checks made against its alternatives.", ""]

    for hypothesis in report["hypotheses"]:
        lines.append(f"### {hypothesis['name']} — confidence: {hypothesis['confidence']}")
        lines.append(hypothesis["statement"])
        lines.append(f"\n*Basis:* {hypothesis['confidence_basis']}.")

        if hypothesis["alternatives"]:
            lines += ["", "| alternative explanation | check | observed | result |", "|---|---|---|---|"]

            for alternative in hypothesis["alternatives"]:
                lines.append(f"| {alternative['explanation']} | {alternative['check']} | {alternative['observed']} | {alternative['result']} |")

        if hypothesis["supporting_observations"]:
            lines.append(f"\nSupporting observations: {', '.join(hypothesis['supporting_observations'])}.")

        lines.append(f"\n*Next check:* {hypothesis['next_check']}\n")

    lines += ["## What this report does not show", ""]
    lines += [f"- {item}" for item in report["limitations"]]
    lines += ["", "Next step is a human decision. This report presents evidence; it does not decide."]

    return "\n".join(lines) + "\n"


def write_report(report, output_dir):
    """Write the markdown and JSON reports. The timestamp lives only in JSON metadata."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown = render_markdown(report)
    document = {
        "metadata": {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "report": report,
    }

    markdown_path = output_dir / f"{report['case_id']}.md"
    json_path = output_dir / f"{report['case_id']}.json"

    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True, default=str), encoding="utf-8")

    return markdown, markdown_path, json_path


def main():
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Investigate one subject and write a case file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--user-id")
    group.add_argument("--session-id")
    group.add_argument("--device-id")
    group.add_argument("--payment-id")
    group.add_argument("--ip")
    parser.add_argument("--data-dir", type=Path, default=root / "data", help="directory holding the published stages")
    parser.add_argument("--output-dir", type=Path, default=root / "reports" / "cases", help="where to write the case files")
    args = parser.parse_args()

    subjects = [
        ("user_id", args.user_id), ("session_uuid", args.session_id), ("device_id", args.device_id),
        ("payment_hash", args.payment_id), ("client_ip", args.ip),
    ]
    subject_type, subject_value = next((kind, value) for kind, value in subjects if value is not None)

    inputs = load_inputs(args.data_dir)
    report = investigate(inputs, subject_type, subject_value)
    markdown, markdown_path, json_path = write_report(report, args.output_dir)

    print(markdown)
    print(f"Case files: {markdown_path} and {json_path}")

    # A diagnostic or a missing subject is not a successful investigation.
    if report["status"] != "case":
        sys.exit(2)


if __name__ == "__main__":
    main()