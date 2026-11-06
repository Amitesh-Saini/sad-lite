# =============================================================================
# SAD-Lite — Investigation and Provenance Tests
# =============================================================================
#
# PURPOSE
# Check the guarantees investigate.py and provenance.py claim, by trying to
# break each one: mixing runs, interrupting stages, swapping inputs mid-read,
# smuggling ground truth in, escaping the data directory, hiding evidence
# behind a busy shared address, and injecting terminal control characters.
#
# WHY THIS IS NEEDED
# A case report feeds an enforcement decision. If it can silently combine two
# runs, read the answer key, or lose the strongest evidence, the decision built
# on it is unsound, and nothing else in the pipeline would notice.
#
# WHAT IT TESTS
# - Independence from ground truth: identical reports with truth changed or
#   removed, and truth files never opened.
# - Provenance: scores from another run excluded; incomplete stages rejected;
#   changed sessions produce a diagnostic; forbidden columns, path traversal,
#   and symlinks rejected; a file replaced mid-run never mixes into a report.
# - The documented limit: a consistent rewrite of data and manifest is NOT
#   detected, and the test says so rather than pretending otherwise.
# - Investigation: the pivot-suppression attack fails; observations can be
#   re-derived from the sessions they cite; widely shared identifiers stay
#   bounded with the stopping point disclosed; control characters are
#   neutralised; repeated runs give identical reports.
#
# HOW IT RUNS
# Two complete pipelines (seeds 42 and 7) are published into a temporary
# directory once. Tests that modify data work on copies. Nothing in the
# project's data/ or reports/ is read or changed.
#
# Run:
#     python -m pytest tests/ -v
#     python tests/test_investigate.py      # works without pytest installed
# =============================================================================

import json
import shutil
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import investigate  # noqa: E402
import provenance  # noqa: E402
from pipeline_fixture import publish_pipeline  # noqa: E402


SEED = 42
OTHER_SEED = 7


# --------------------------------------------------------------------------
# Fixtures: complete published pipelines, built once (tests/pipeline_fixture.py)
# --------------------------------------------------------------------------

_workspace = Path(tempfile.mkdtemp(prefix="sadlite-investigate-tests-"))
_main = _workspace / "seed_42"
_other = _workspace / "seed_7"

_labels, _ = publish_pipeline(_main, SEED)
publish_pipeline(_other, OTHER_SEED)


def account_of(pattern):
    """The first account of a scenario: used only to choose subjects, never by the investigator."""

    return _labels.loc[_labels["pattern"] == pattern, "user_id"].iloc[0]


SMURF = account_of("smurf_ring")
RESALE = account_of("account_resale")
CHEATER = account_of("cheater")
DORM = account_of("dorm_shared_ip")


def copy_of(source):
    """A fresh copy of a published pipeline for a test that modifies data."""

    target = Path(tempfile.mkdtemp(prefix="case-", dir=_workspace)) / "data"
    shutil.copytree(source, target)
    return target


def run(data_dir, subject_type="user_id", subject_value=SMURF):
    """Investigate from a data directory; return the report and its markdown."""

    report = investigate.investigate(investigate.load_inputs(data_dir), subject_type, subject_value)
    return report, investigate.render_markdown(report)


def rewrite_manifest_hash(stage_dir, file_name):
    """Make a manifest match a modified file: possible for anyone who can write data/."""

    manifest_path = stage_dir / provenance.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = (stage_dir / file_name).read_bytes()
    manifest["files"][file_name] = {"sha256": provenance.sha256_bytes(data), "bytes": len(data)}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def stage_status(report, stage):
    return next(record for record in report["provenance"] if record["stage"] == stage)


def sessions_frame(data_dir):
    _, contents = provenance.load_stage(data_dir, "generate", ["sessions.csv"])
    return investigate.prepare_sessions(provenance.frame_from_bytes(contents["sessions.csv"], dtype=str, keep_default_na=False))


_baseline_report, _baseline_markdown = run(_main)


# --------------------------------------------------------------------------
# Independence from ground truth
# --------------------------------------------------------------------------

def test_report_is_identical_without_truth():
    data_dir = copy_of(_main)
    shutil.rmtree(data_dir / "truth")

    report, markdown = run(data_dir)

    assert markdown == _baseline_markdown
    assert report == _baseline_report


def test_report_is_identical_when_truth_changes():
    data_dir = copy_of(_main)
    labels_path = data_dir / "truth" / "labels.csv"
    flipped = pd.read_csv(labels_path)
    flipped["label"] = 1 - flipped["label"]
    flipped.to_csv(labels_path, index=False)

    _, markdown = run(data_dir)

    assert markdown == _baseline_markdown


def test_investigator_never_opens_truth():
    opened = []
    original = provenance.read_file_once

    def recording(path, *args, **kwargs):
        opened.append(Path(path).resolve())
        return original(path, *args, **kwargs)

    provenance.read_file_once = recording

    try:
        run(_main)
    finally:
        provenance.read_file_once = original

    truth_dir = (_main / "truth").resolve()

    assert opened, "the investigator should read its inputs through provenance"
    assert not [path for path in opened if truth_dir in path.parents], "truth files were opened"


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def test_scores_from_another_run_are_excluded():
    data_dir = copy_of(_main)
    shutil.rmtree(data_dir / "rules")
    shutil.copytree(_other / "rules", data_dir / "rules")

    report, markdown = run(data_dir)

    assert stage_status(report, "rules")["status"] == "excluded"
    assert "different features" in stage_status(report, "rules")["reason"]
    assert report["scores"]["rules"] is None
    assert "Rule score: not available" in markdown


def test_incomplete_stage_is_rejected():
    data_dir = copy_of(_main)
    (data_dir / "rules" / provenance.MANIFEST_NAME).unlink()

    report, _ = run(data_dir)

    assert stage_status(report, "rules")["status"] == "excluded"
    assert "incomplete" in stage_status(report, "rules")["reason"]


def test_changed_sessions_produce_a_diagnostic_not_a_case():
    data_dir = copy_of(_main)
    sessions_path = data_dir / "generate" / "sessions.csv"
    sessions_path.write_bytes(sessions_path.read_bytes().replace(b"normal", b"timeout", 1))

    report, markdown = run(data_dir)

    assert report["status"] == "diagnostic"
    assert "observations" not in report
    assert "## Diagnostic" in markdown


def test_forbidden_column_is_rejected_even_with_a_consistent_manifest():
    data_dir = copy_of(_main)
    features_path = data_dir / "features" / "features.csv"
    table = pd.read_csv(features_path)
    table["is_bot"] = 0
    table.to_csv(features_path, index=False)
    rewrite_manifest_hash(data_dir / "features", "features.csv")

    report, _ = run(data_dir)

    assert stage_status(report, "features")["status"] == "excluded"
    assert "is_bot" in stage_status(report, "features")["reason"]
    assert report["percentiles"] == {}


def test_path_traversal_in_a_manifest_is_rejected():
    data_dir = copy_of(_main)
    manifest_path = data_dir / "features" / provenance.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["../truth/labels.csv"] = manifest["files"].pop("features.csv")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report, _ = run(data_dir)

    assert stage_status(report, "features")["status"] == "excluded"
    assert "Unsafe file name" in stage_status(report, "features")["reason"]


def test_symlinked_input_is_rejected():
    data_dir = copy_of(_main)
    features_path = data_dir / "features" / "features.csv"
    elsewhere = data_dir.parent / "features_copy.csv"
    shutil.move(features_path, elsewhere)
    features_path.symlink_to(elsewhere)

    report, _ = run(data_dir)

    assert stage_status(report, "features")["status"] == "excluded"
    assert "symlink" in stage_status(report, "features")["reason"]


def test_file_replaced_mid_run_never_mixes_into_a_report():
    """Pause after sessions.csv is read and hashed, swap in another run's file, resume."""

    data_dir = copy_of(_main)
    sessions_path = (data_dir / "generate" / "sessions.csv").resolve()
    replacement = (_other / "generate" / "sessions.csv").read_bytes()
    original = provenance.read_file_once

    def swap_after_reading(path, *args, **kwargs):
        data = original(path, *args, **kwargs)

        if Path(path).resolve() == sessions_path:
            sessions_path.write_bytes(replacement)

        return data

    provenance.read_file_once = swap_after_reading

    try:
        report, markdown = run(data_dir)
    finally:
        provenance.read_file_once = original

    # The report describes exactly the bytes that were hashed: the original run.
    assert report["provenance"][0]["files"]["sessions.csv"] == _baseline_report["provenance"][0]["files"]["sessions.csv"]
    assert markdown == _baseline_markdown


def test_consistent_rewrite_of_data_and_manifest_is_outside_the_guarantee():
    """Documents the trust model's limit; it does not claim to catch this.

    Anyone who can write data/ can change a score and rewrite the manifest to
    match. Provenance protects against accidental mixing, not this. Detecting
    it would need a signature checked against a key outside the writer's reach.
    """

    data_dir = copy_of(_main)
    scores_path = data_dir / "rules" / "rule_scores.csv"
    table = pd.read_csv(scores_path, dtype={"user_id": str})
    table.loc[table["user_id"] == SMURF, "triggered_rules"] = "tampered"
    table.to_csv(scores_path, index=False)
    rewrite_manifest_hash(data_dir / "rules", "rule_scores.csv")

    report, _ = run(data_dir)

    assert stage_status(report, "rules")["status"] == "verified"
    assert report["scores"]["rules"]["triggered_rules"] == "tampered"


# --------------------------------------------------------------------------
# Investigation
# --------------------------------------------------------------------------

def test_pivot_suppression_attack_fails():
    """A ring routed through a busy shared address must still be exposed.

    If IPs were expanded first under one shared limit, the address's innocent
    accounts would fill the search before the ring's shared device was reached.
    """

    sessions = sessions_frame(_main)
    raw = provenance.frame_from_bytes(provenance.load_stage(_main, "generate", ["sessions.csv"])[1]["sessions.csv"],
                                      dtype=str, keep_default_na=False)

    # The busiest address used by the VPN lookalikes: a shared exit.
    vpn_users = set(_labels.loc[_labels["pattern"] == "normal_vpn", "user_id"])
    exit_ip = sessions[sessions["user_id"].isin(vpn_users)]["client_ip"].value_counts().index[0]
    ring = set(_labels.loc[_labels["pattern"] == "smurf_ring", "user_id"])
    raw.loc[raw["user_id"].isin(ring), "client_ip"] = exit_ip

    inputs = {"sessions": raw, "sessions_sha256": "poc", "features": None, "rules": None, "model": None, "provenance": []}
    report = investigate.investigate(inputs, "user_id", SMURF)

    first = report["connections"][0]
    ip_entry = next(item for item in report["connections"] if item["identifier_type"] == "client_ip")

    assert first["identifier_type"] == "device_id", "strong identifiers must be examined first"
    assert first["other_accounts"] == 29
    assert ip_entry["accounts"] > 30, "the attack needs the ring to share a busy address"
    assert any(item["name"] == "Coordinated multi-accounting" and item["confidence"] == "Strong" for item in report["hypotheses"])


def test_observations_can_be_rederived_from_cited_sessions():
    sessions = sessions_frame(_main).set_index("session_uuid")

    for account in [SMURF, RESALE, CHEATER]:
        report, _ = run(_main, "user_id", account)

        for item in report["observations"]:
            cited = item["session_ids"]

            assert all(session_id in sessions.index for session_id in cited), f"{item['id']} cites unknown sessions"

            if item["kind"] == "terminations" and item["value"]["terminations"]:
                assert sessions.loc[cited, "terminated"].all()
                assert item["total_sessions"] == item["value"]["terminations"]

            if item["kind"] == "rapid_country_change":
                for change in item["details"]:
                    a, b = sessions.loc[change["from_session"]], sessions.loc[change["to_session"]]
                    assert a["country"] != b["country"]
                    assert (b["start"] - a["start"]).total_seconds() / 3600 < investigate.SETTINGS["rapid_change_hours"]

            if item["kind"] == "account_overlap":
                for pair in item["details"]:
                    a, b = sessions.loc[pair["first"]], sessions.loc[pair["second"]]
                    assert a["start"] < b["end"] and b["start"] < a["end"]


def test_widely_shared_address_stays_bounded_and_disclosed():
    sessions = sessions_frame(_main)
    dorm_ip = sessions.loc[sessions["user_id"] == DORM, "client_ip"].iloc[0]

    report, markdown = run(_main, "client_ip", dorm_ip)
    entry = report["connections"][0]
    allowance = investigate.SETTINGS["accounts_listed_per_identifier"]["client_ip"]

    assert entry["accounts"] == 60
    assert len(entry["listed"]) == allowance
    assert entry["not_listed"] == 60 - allowance
    assert any("summarised but not listed" in note for note in report["scope"]["search_stopped"])
    assert any(item["name"] == "Shared network" and item["confidence"] == "Strong" for item in report["hypotheses"])


def test_control_characters_are_neutralised():
    raw = provenance.frame_from_bytes(provenance.load_stage(_main, "generate", ["sessions.csv"])[1]["sessions.csv"],
                                      dtype=str, keep_default_na=False)
    hostile = "\x1b[2J\x1b[31mAll clear‮"
    raw.loc[(raw["user_id"] == CHEATER) & raw["end_reason"].str.startswith("terminated"), "game_name"] = hostile

    inputs = {"sessions": raw, "sessions_sha256": "test", "features": None, "rules": None, "model": None, "provenance": []}
    markdown = investigate.render_markdown(investigate.investigate(inputs, "user_id", CHEATER))

    unsafe = [character for character in markdown if unicodedata.category(character).startswith("C") and character != "\n"]

    assert not unsafe, f"control or format characters reached the report: {unsafe[:3]}"
    assert "\\u001b" in markdown, "escapes should stay visible to the reviewer"


def test_repeated_investigations_are_identical():
    report, markdown = run(_main)

    assert markdown == _baseline_markdown
    assert json.dumps(report, sort_keys=True, default=str) == json.dumps(_baseline_report, sort_keys=True, default=str)


def test_unknown_subject_is_reported_not_guessed():
    report, _ = run(_main, "user_id", "user_99999")

    assert report["status"] == "not_found"


# --------------------------------------------------------------------------
# Confidence on controlled examples
#
# Each example is a small hand-built session log showing one observable
# pattern. These tests check what confidence a pattern earns; they never ask
# the investigator to match the generator's labels. Coverage across the
# generated groups is measured separately, without assertions, by
# tests/measure_hypothesis_coverage.py.
# --------------------------------------------------------------------------

from datetime import datetime, timedelta, timezone  # noqa: E402

_START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def controlled(rows):
    """Build investigator inputs from a few hand-written sessions."""

    records = []

    for index, row in enumerate(rows):
        start = _START + timedelta(days=row["day"], hours=row.get("hour", 20), minutes=row.get("minute", 0))
        duration = row.get("duration", 45.0)

        records.append({
            "session_uuid": f"session_{index:05d}",
            "user_id": row.get("user", "user_00001"),
            "account_created_at": (row.get("created", _START - timedelta(days=400))).isoformat(),
            "session_start": start.isoformat(),
            "session_end": (start + timedelta(minutes=duration)).isoformat(),
            "duration_min": str(duration),
            "client_ip": row.get("ip", "10.0.0.1"),
            "country": row.get("country", "CA"),
            "device_id": row.get("device", "device_0000000001"),
            "app_id": row.get("app", "app_1001"),
            "game_name": row.get("game", "Neon Drift"),
            "rig_type": "basic",
            "input_events_per_min": str(row.get("input", 120.0)),
            "bytes_streamed": "1000",
            "payment_hash": row.get("payment", "payment_0000000001"),
            "end_reason": row.get("end", "normal"),
        })

    raw = pd.DataFrame(records).astype(str)
    return {"sessions": raw, "sessions_sha256": "controlled", "features": None, "rules": None, "model": None, "provenance": []}


def confidence_of(inputs, name, account="user_00001"):
    report = investigate.investigate(inputs, "user_id", account)
    return next((item["confidence"] for item in report["hypotheses"] if item["name"] == name), None)


def varied(index, low, high):
    """Deterministic human-looking variation."""

    return low + (index * 37 % 101) / 100 * (high - low)


def test_traveller_on_own_device_is_at_most_weak():
    rows = [{"day": day, "duration": varied(day, 20, 120), "input": varied(day, 90, 160)} for day in range(30)]

    # A four-day trip on the same laptop, with a departure-day session abroad.
    for row in rows:
        if 10 <= row["day"] <= 13:
            row.update(country="FR", ip="10.9.9.9")

    rows.append({"day": 10, "hour": 9, "duration": 30})

    assert confidence_of(controlled(rows), "Account sharing or resale") in (None, "Weak")


def test_vpn_user_alternating_countries_is_at_most_weak():
    rows = []

    for day in range(30):
        rows.append({"day": day, "hour": 18, "duration": varied(day, 20, 90)})

        if day % 3 == 0:
            rows.append({"day": day, "hour": 21, "duration": varied(day + 1, 20, 90), "country": "DE", "ip": "10.8.8.8"})

    assert confidence_of(controlled(rows), "Account sharing or resale") in (None, "Weak")


def test_several_sharing_patterns_reach_moderate_but_not_strong():
    rows = [{"day": day, "duration": varied(day, 20, 120)} for day in range(14)]

    # Three further devices first observed within two days, records overlapping
    # on different devices, and three countries in one day.
    for offset, (device, country, ip) in enumerate([
        ("device_00000000aa", "US", "10.1.1.1"), ("device_00000000bb", "BR", "10.2.2.2"), ("device_00000000cc", "JP", "10.3.3.3"),
    ]):
        rows.append({"day": 15, "hour": 10 + offset, "minute": 0, "duration": 120, "device": device, "country": country, "ip": ip})
        rows.append({"day": 16, "hour": 2 + offset, "duration": 60, "device": device, "country": country, "ip": ip})

    assert confidence_of(controlled(rows), "Account sharing or resale") == "Moderate"


def test_bot_like_consistency_is_capped_at_moderate():
    # Start times shift by an hour each day, so play covers all 24 hours and
    # every alternative check is contradicted: only the cap holds it at Moderate.
    rows = [
        {"day": day, "hour": (slot * 4 + day) % 24, "minute": 5, "duration": 120 + (slot % 2) * 0.5, "input": 200 + (day % 2) * 0.5}
        for day in range(20) for slot in range(6)
    ]

    report = investigate.investigate(controlled(rows), "user_id", "user_00001")
    automation = next(item for item in report["hypotheses"] if item["name"] == "Automation")

    assert all(item["result"] == "contradicted" for item in automation["alternatives"]), "the example must reach the cap"
    assert automation["confidence"] == "Moderate"
    assert "capped at Moderate" in automation["confidence_basis"]


def test_frequent_repeated_terminations_are_capped_at_moderate():
    rows = [{"day": day, "duration": varied(day, 20, 90), "end": "terminated_anticheat" if day % 2 == 0 else "normal"}
            for day in range(30)]

    # The false-positive explanation can never be ruled out from session data,
    # so the strongest repeated-termination pattern stops at Moderate.
    assert confidence_of(controlled(rows), "Cheating") == "Moderate"


def test_a_few_unlucky_terminations_stay_weak():
    rows = [{"day": day, "duration": varied(day, 20, 90), "end": "terminated_anticheat" if day in (3, 11, 19, 24) else "normal"}
            for day in range(25)]

    assert confidence_of(controlled(rows), "Cheating") == "Weak"


def test_household_sharing_a_console_and_card_is_weak():
    rows = []

    for member in range(4):
        created = _START - timedelta(days=200 + 300 * member)

        for day in range(20):
            rows.append({"day": day, "hour": 12 + member * 2, "user": f"user_0000{member + 1}", "created": created,
                         "device": "device_00000000ff", "duration": varied(day + member, 20, 90)})

    assert confidence_of(controlled(rows), "Coordinated multi-accounting") == "Weak"


def test_ring_created_together_on_shared_equipment_is_strong():
    rows = []

    for member in range(8):
        created = _START + timedelta(days=2, hours=member * 3)

        for day in range(5, 20):
            rows.append({"day": day, "hour": member % 12 + 8, "user": f"user_0000{member + 1}", "created": created,
                         "duration": varied(day + member, 20, 90)})

    assert confidence_of(controlled(rows), "Coordinated multi-accounting") == "Strong"


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
            print(f"  FAIL  {name}: {error!r}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    shutil.rmtree(_workspace, ignore_errors=True)
    sys.exit(1 if failures else 0)