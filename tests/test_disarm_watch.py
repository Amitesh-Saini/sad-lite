# =============================================================================
# SAD-Lite — Disarm Watch Tests
# =============================================================================
#
# PURPOSE
# Check that the game-level watcher finds the incident it should, raises no
# proposal it should not, and that its Git proposal changes exactly one flag
# without touching anyone's working copy.
#
# WHAT IT TESTS
# - Detection: the injected incident is found and proposal-eligible; clean
#   data raises no proposal-eligible alert; incidents placed across window
#   and day boundaries are still found.
# - Window rules: tiny samples never alert; the baseline floor is applied
#   and reported; an insufficient baseline cannot support a proposal;
#   sessions in a flagged window are excluded from later baselines.
# - Inputs: grouping is by app_id, not display name; games in the data
#   without configuration are reported as unmonitored.
# - Proposals: refused outside a Git repository, with uncommitted changes to
#   games.yaml, after the committed configuration changes, for an edited
#   alert, for an insufficient-baseline alert, and for an already-disabled
#   game. A dry run writes nothing. An applied proposal creates one branch
#   with a one-line diff, leaves HEAD, the checked-out branch, and the working
#   tree untouched, and an exact retry is recognised. A conflicting existing
#   branch is refused. The single-flag proof rejects any other change.
#
# HOW IT RUNS
# Session stages are published into a temporary directory; Git tests use
# throwaway repositories. Nothing in the project's own folders is touched.
#
# Run:
#     python -m pytest tests/ -v
#     python tests/test_disarm_watch.py
# =============================================================================

import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import disarm_watch  # noqa: E402
import game_config  # noqa: E402
import game_monitor  # noqa: E402
import generate_users  # noqa: E402
import provenance  # noqa: E402


CONFIG_BYTES = (ROOT / "config" / "games.yaml").read_bytes()
SETTINGS = game_config.validate(game_config.parse(CONFIG_BYTES))["app_1001"]["settings"]
INCIDENT_APP = generate_users.INCIDENT_APP_ID

_workspace = Path(tempfile.mkdtemp(prefix="sadlite-disarm-tests-"))


def publish_sessions(name, **options):
    """Publish only the generate stage for one dataset; return (data_dir, incident)."""

    sessions, _, incident = generate_users.build_dataset(seed=42, **options)
    data = generate_users.rows_to_csv_bytes(sessions, generate_users.SESSION_FIELDS)
    data_dir = _workspace / name / "data"
    provenance.publish_stage(data_dir, "generate", {"sessions.csv": data}, {}, {"test": name}, generate_users.__file__)

    return data_dir, incident


def scan_of(data_dir, config_bytes=CONFIG_BYTES):
    sessions_bytes, dataset_id = disarm_watch.load_sessions(data_dir)
    return disarm_watch.scan(sessions_bytes, dataset_id, config_bytes)


def eligible(report):
    return [alert for alert in report["alerts"] if alert["status"] == "proposal_eligible"]


def overlaps_incident(alert, incident):
    return (alert["app_id"] == incident["app_id"]
            and pd.Timestamp(alert["episode"]["start"]) < pd.Timestamp(incident["incident_end"])
            and pd.Timestamp(alert["episode"]["end"]) > pd.Timestamp(incident["incident_start"]))


_incident_dir, _incident = publish_sessions("incident")
_clean_dir, _ = publish_sessions("clean", include_incident=False)
_incident_report = scan_of(_incident_dir)
_incident_alert = eligible(_incident_report)[0]


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

def test_the_incident_is_detected_and_proposal_eligible():
    proposals = eligible(_incident_report)

    assert len(proposals) == 1, f"expected one proposal-eligible alert, got {len(proposals)}"
    assert overlaps_incident(proposals[0], _incident)
    assert proposals[0]["peak_window"]["anticheat_terminations"] >= 200


def test_clean_data_raises_no_proposal():
    assert eligible(scan_of(_clean_dir)) == []


def test_incidents_across_window_and_day_boundaries_are_detected():
    # Starting mid-hour, and starting late enough to run past midnight.
    for name, start_hours in [("half-hour", 22 * 24 + 18.5), ("past-midnight", 22 * 24 + 21)]:
        data_dir, incident = publish_sessions(name, incident_start_hours=start_hours)
        proposals = [alert for alert in eligible(scan_of(data_dir)) if overlaps_incident(alert, incident)]

        assert proposals, f"{name} incident was not detected"


# --------------------------------------------------------------------------
# Window rules (controlled examples)
# --------------------------------------------------------------------------

def controlled_sessions(rows):
    """rows: (hours from start, terminated) for one game."""

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    frame = pd.DataFrame({
        "session_uuid": [f"session_{index:05d}" for index in range(len(rows))],
        "app_id": INCIDENT_APP,
        "session_end": [(start + timedelta(hours=hours)).isoformat() for hours, _ in rows],
        "end_reason": ["terminated_anticheat" if terminated else "normal" for _, terminated in rows],
    })
    return game_monitor.prepare_sessions(frame)


def test_one_termination_in_two_sessions_never_alerts():
    # A quiet game: after a ten-hour gap, two sessions in one window, one terminated (50%).
    history = [(hours / 4, False) for hours in range(4 * 24 * 8)]
    rows = history + [(24 * 8 + 10, True), (24 * 8 + 11, False)]

    windows, episodes = game_monitor.evaluate_game(controlled_sessions(rows), INCIDENT_APP, SETTINGS)
    window = next(item for item in windows if item["sessions"] == 2 and item["terminations"] == 1)

    assert not window["breached"] and not window["conditions"]["sessions"]
    assert episodes == []


def test_baseline_floor_is_applied_and_reported():
    history = [(hours / 4, False) for hours in range(4 * 24 * 8)]
    windows, _ = game_monitor.evaluate_game(controlled_sessions(history), INCIDENT_APP, SETTINGS)
    late = [item for item in windows if item["baseline_sufficient"]]

    assert late and all(item["observed_baseline"] == 0.0 for item in late)
    assert all(item["floor_applied"] and item["effective_baseline"] == SETTINGS["baseline_floor_rate"] for item in late)


def test_insufficient_baseline_cannot_support_a_proposal():
    # A burst on the first day, before enough history exists.
    rows = [(1 + index / 100, index % 2 == 0) for index in range(60)]
    _, episodes = game_monitor.evaluate_game(controlled_sessions(rows), INCIDENT_APP, SETTINGS)

    assert episodes and not any(episode["proposal_eligible"] for episode in episodes)


def test_flagged_sessions_are_excluded_from_later_baselines():
    sessions = game_monitor.prepare_sessions(provenance.frame_from_bytes(
        disarm_watch.load_sessions(_incident_dir)[0], dtype=str, keep_default_na=False))
    windows, episodes = game_monitor.evaluate_game(sessions, INCIDENT_APP, SETTINGS)
    after = episodes[-1]["end"] + timedelta(hours=12)
    later = next(item for item in windows if item["window_end"] == after)

    # The incident's ~290 terminations would dominate this baseline if included.
    assert later["baseline_terminations"] < 60


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def test_grouping_is_by_app_id_not_display_name():
    sessions_bytes, dataset_id = disarm_watch.load_sessions(_incident_dir)
    frame = provenance.frame_from_bytes(sessions_bytes, dtype=str, keep_default_na=False)
    frame.loc[frame.index % 2 == 0, "game_name"] = "Renamed Display Label"
    renamed = frame.to_csv(index=False).encode("utf-8")

    report = disarm_watch.scan(renamed, dataset_id, CONFIG_BYTES)

    assert report["alerts"] == _incident_report["alerts"]


def test_games_without_configuration_are_reported_as_unmonitored():
    text = CONFIG_BYTES.decode("utf-8")
    # No trailing newline in the block: the last game may end the file without one.
    block = '  - app_id: "app_1012"\n    name: "Molten Yard"\n    enforcement_enabled: true'
    assert text.count(block) == 1, "fixture text not found in config/games.yaml"
    report = scan_of(_incident_dir, text.replace(block, "").encode("utf-8"))

    assert report["unmonitored_app_ids"] == ["app_1012"]


# --------------------------------------------------------------------------
# Proposals
# --------------------------------------------------------------------------

def git(repo, *arguments):
    return subprocess.run(["git", *arguments], cwd=repo, capture_output=True, check=True).stdout.decode().strip()


def make_repo(config_bytes=CONFIG_BYTES):
    repo = Path(tempfile.mkdtemp(prefix="repo-", dir=_workspace))
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.invalid")
    (repo / "config").mkdir()
    (repo / "config" / "games.yaml").write_bytes(config_bytes)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    return repo


def write_report(report):
    folder = Path(tempfile.mkdtemp(prefix="scan-", dir=_workspace))
    return disarm_watch.write_scan(report, folder) / "alerts.json"


def propose(repo, report_path, alert_id=None, apply=False, data_dir=None):
    return disarm_watch.propose(report_path, alert_id or _incident_alert["alert_id"], data_dir or _incident_dir,
                                repo / "config" / "games.yaml", apply, _workspace / "proposals")


def refused(call, expected):
    try:
        call()
    except disarm_watch.WatchError as error:
        assert expected in str(error), f"refused for another reason: {error}"
        return
    raise AssertionError(f"expected refusal mentioning {expected!r}")


_report_path = write_report(_incident_report)


def test_refused_outside_a_git_repository():
    folder = Path(tempfile.mkdtemp(prefix="not-a-repo-", dir=_workspace))
    (folder / "config").mkdir()
    (folder / "config" / "games.yaml").write_bytes(CONFIG_BYTES)

    refused(lambda: propose(folder, _report_path, apply=True), "not inside a Git repository")
    assert (folder / "config" / "games.yaml").read_bytes() == CONFIG_BYTES


def test_dry_run_writes_nothing():
    repo = make_repo()
    result = propose(repo, _report_path)

    assert result["status"] == "dry_run"
    assert git(repo, "branch", "--list", "disarm/*") == ""
    assert git(repo, "status", "--porcelain") == ""


def test_applied_proposal_changes_one_flag_on_a_new_branch_only():
    repo = make_repo()
    (repo / "notes.txt").write_text("uncommitted work", encoding="utf-8")
    head = git(repo, "rev-parse", "HEAD")

    result = propose(repo, _report_path, apply=True)
    branch = result["branch"]

    assert result["status"] == "proposed"
    assert git(repo, "rev-parse", "HEAD") == head and git(repo, "branch", "--show-current") == "main"
    assert git(repo, "status", "--porcelain") == "?? notes.txt"
    assert (repo / "config" / "games.yaml").read_bytes() == CONFIG_BYTES

    diff = [line for line in git(repo, "diff", "--unified=0", "HEAD", branch).splitlines()
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]

    assert diff == ["-    enforcement_enabled: true", "+    enforcement_enabled: false"]

    proposed = game_config.validate(game_config.parse(git(repo, "show", f"{branch}:config/games.yaml").encode() + b"\n"))
    original = game_config.validate(game_config.parse(CONFIG_BYTES))

    assert proposed[INCIDENT_APP]["enforcement_enabled"] is False
    assert all(proposed[app]["enforcement_enabled"] == original[app]["enforcement_enabled"] for app in original if app != INCIDENT_APP)
    assert all(proposed[app]["settings"] == original[app]["settings"] for app in original)

    assert propose(repo, _report_path, apply=True)["status"] == "already_proposed"


def test_uncommitted_change_to_games_yaml_is_refused():
    repo = make_repo()
    (repo / "config" / "games.yaml").write_bytes(CONFIG_BYTES + b"\n# local edit\n")

    refused(lambda: propose(repo, _report_path, apply=True), "uncommitted changes")


def test_configuration_changed_since_the_alert_is_refused():
    repo = make_repo()
    (repo / "config" / "games.yaml").write_bytes(CONFIG_BYTES.replace(b"min_sessions: 20", b"min_sessions: 25"))
    git(repo, "commit", "-qam", "change thresholds")

    refused(lambda: propose(repo, _report_path, apply=True), "changed since the alert was computed")


def test_edited_alert_is_refused():
    document = json.loads(_report_path.read_text(encoding="utf-8"))
    alert = next(item for item in document["report"]["alerts"] if item["alert_id"] == _incident_alert["alert_id"])
    alert["peak_window"]["rate"] = 0.99
    edited = _workspace / "edited-alerts.json"
    edited.write_text(json.dumps(document), encoding="utf-8")

    refused(lambda: propose(make_repo(), edited, apply=True), "does not match a fresh scan")


def test_insufficient_baseline_alert_is_refused():
    insufficient = next(alert for alert in _incident_report["alerts"] if alert["status"] == "insufficient_baseline")

    refused(lambda: propose(make_repo(), _report_path, insufficient["alert_id"], apply=True), "insufficient baseline")


def test_already_disabled_game_gets_no_proposal():
    lines = CONFIG_BYTES.decode("utf-8").splitlines(keepends=True)
    start = lines.index(f'  - app_id: "{INCIDENT_APP}"\n')
    lines[start + 2] = "    enforcement_enabled: false\n"
    disabled = "".join(lines).encode("utf-8")

    report = scan_of(_incident_dir, disabled)
    alert = next(item for item in report["alerts"] if item["app_id"] == INCIDENT_APP and overlaps_incident(item, _incident))

    assert alert["status"] == "already_disabled"
    refused(lambda: propose(make_repo(disabled), write_report(report), alert["alert_id"], apply=True), "already disabled")


def test_conflicting_existing_branch_is_refused():
    repo = make_repo()
    branch = propose(repo, _report_path)["branch"]
    git(repo, "branch", branch)

    refused(lambda: propose(repo, _report_path, apply=True), "already exists with different contents")


def test_single_flag_proof_rejects_any_other_change():
    edited = disarm_watch.edit_flag(CONFIG_BYTES, INCIDENT_APP)

    assert disarm_watch.prove_single_flag_change(CONFIG_BYTES, edited, INCIDENT_APP) > 0

    threshold_too = edited.replace(b"baseline_multiplier: 5.0", b"baseline_multiplier: 50.0")
    other_game = disarm_watch.edit_flag(edited, "app_1001")
    for candidate in [threshold_too, other_game]:
        refused(lambda candidate=candidate: disarm_watch.prove_single_flag_change(CONFIG_BYTES, candidate, INCIDENT_APP), "lines")

    single_line_other_change = CONFIG_BYTES.replace(b'"Neon Drift"', b'"Neon  Drift"')
    refused(lambda: disarm_watch.prove_single_flag_change(CONFIG_BYTES, single_line_other_change, INCIDENT_APP),
            "something other than")


def test_layout_outside_the_contract_is_refused():
    single_quoted = CONFIG_BYTES.replace(f'"{INCIDENT_APP}"'.encode(), f"'{INCIDENT_APP}'".encode())

    refused(lambda: disarm_watch.edit_flag(single_quoted, INCIDENT_APP), "exactly once (layout rule)")


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