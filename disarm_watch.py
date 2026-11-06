# =============================================================================
# SAD-Lite — Game-Level Disarm Watch
# =============================================================================
#
# PURPOSE
# Check whether unusual anti-cheat terminations are happening across a whole
# game, rather than investigating individual accounts, and, when they are,
# prepare a narrow, reviewable proposal to pause that game's anti-cheat
# enforcement while someone investigates.
#
# WHY THIS IS NEEDED
# When anti-cheat misfires on one game, hundreds of ordinary players each get
# one bad session. User-level detection sees nothing unusual about any of
# them (that is tested), so a game-wide problem could otherwise be mistaken
# for many separate cheating cases, or not noticed at all.
#
# WHAT IT CAN AND CANNOT TELL
# It detects a pattern that needs review: a game's anti-cheat termination
# rate far above its own recent baseline. It cannot tell the cause. An
# anti-cheat fault, a game update, an infrastructure or telemetry problem,
# and a real cheat wave all produce the same pattern, and can overlap.
# Disarming during a real cheat wave helps cheaters, so every alert and
# proposal states this uncertainty and requires human review.
#
# WORKFLOW
# scan:
# 1. Read and verify the inputs: the published sessions stage (provenance.py)
#    and config/games.yaml (game_config.py). Games in the data without
#    configuration are reported as unmonitored.
# 2. Group sessions by app_id; the name is only a display label.
# 3. Check every rolling window for every game (game_monitor.py): sessions,
#    anti-cheat terminations, and their rate, by session_end.
# 4. Compare with the game's own earlier baseline, excluding the window, with
#    the minimum-history and floor rules.
# 5. Require all four alert conditions.
# 6. Write an evidence-backed alert report: JSON for software (agent_sim.py),
#    markdown for people. Each alert carries the dataset and config hashes,
#    the thresholds, the baseline counts, and the session IDs needed to check
#    the calculation.
# propose (dry run unless --apply), through prepare_disarm(), which
# agent_sim.py's propose_disarm tool also calls:
# 1. Check this is a Git repository before anything else.
# 2. Refuse if games.yaml has uncommitted changes, or if the committed file
#    differs from the configuration the alert was computed against.
# 3. Re-run the scan on the verified data and committed configuration and
#    require the alert to be in it (and, from a report, identical), so an
#    edited report or a caller's say-so cannot drive a proposal.
# 4. Only a proposal-eligible alert on a game whose enforcement is enabled
#    can produce a proposal.
# 5. Edit exactly one line: that game's enforcement_enabled true -> false.
#    Parse and validate both versions and prove the only difference is that
#    flag; refuse otherwise.
# 6. With --apply, under a lock and after re-checking the committed file,
#    commit the edit on a new branch disarm/<app_id>-<alert> and write the
#    merge-request description, labelled as a replay of synthetic data.
#
# THE GIT EDIT NEVER TOUCHES YOUR WORKING COPY
# The commit is built with Git plumbing (a temporary index, commit-tree, and
# update-ref) on top of HEAD. Your checked-out branch, working tree, and any
# uncommitted work are left exactly as they were. Nothing is merged.
#
# WHAT DISARMING DOES AND DOES NOT DO
# A merged proposal sets one game's simulated enforcement_enabled flag to
# false. It does not stop monitoring, does not reverse existing bans, and does
# not send anything to a real game server. Re-enabling is a manual edit; no
# automated path ever sets the flag back to true.
#
# Run:
#     python disarm_watch.py scan
#     python disarm_watch.py propose --report reports/disarm/<scan>/alerts.json --alert-id alert_...
#     python disarm_watch.py propose ... --apply
#
# SOURCES
# The rolling-window watcher, the three-part alert condition, and the
# branch-plus-merge-request idea follow the SAD-Lite reference implementation
# (sadlite/disarm_watch.py). Evaluating every window, the baseline rules,
# calibrated thresholds, scan regeneration, the single-flag proof, and the
# plumbing-only commit are this project's own.
# =============================================================================

import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import game_config
import game_monitor
import provenance


ROOT = Path(__file__).resolve().parent

WATCHER_VERSION = "1.1.0"
REPORT_SCHEMA_VERSION = 1

DEFAULT_CONFIG = ROOT / "config" / "games.yaml"

SESSION_IDS_SHOWN = 10

POSSIBLE_EXPLANATIONS = [
    "An anti-cheat fault, such as a bad signature update, terminating honest players.",
    "A game update or infrastructure problem that anti-cheat misreads as cheating.",
    "A telemetry or logging error that records terminations that did not happen as recorded.",
    "A real wave of cheating, for example after a new cheat is released. Disarming would help cheaters.",
]

CAUSE_STATEMENT = (
    "The watcher detects the pattern, not its cause. These explanations produce the same "
    "pattern and can overlap. Human review is required."
)

# The disarm edit's layout contract (see config/games.yaml).
GAME_LINE = '  - app_id: "{app_id}"'
FLAG_TRUE = "    enforcement_enabled: true"
FLAG_FALSE = "    enforcement_enabled: false"


class WatchError(Exception):
    """The scan or proposal was refused; nothing was written."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def as_json(value):
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True, default=str))


def code_record():
    return {"version": WATCHER_VERSION, "file": Path(__file__).name,
            "sha256": provenance.sha256_bytes(provenance.read_file_once(__file__))}


# ------------------------------------------------------------
# Scan.
# ------------------------------------------------------------

def load_sessions(data_dir):
    """Verified session bytes and the dataset ID."""

    try:
        _, contents = provenance.load_stage(data_dir, "generate", ["sessions.csv"])
    except provenance.ProvenanceError as error:
        raise WatchError(f"Session data cannot be verified: {error}") from error

    data = contents["sessions.csv"]

    return data, provenance.sha256_bytes(data)


def scan(sessions_bytes, dataset_id, config_bytes):
    """Evaluate every configured game. Pure and deterministic given its inputs."""

    try:
        config = game_config.parse(config_bytes)
        effective = game_config.validate(config)
    except game_config.ConfigError as error:
        raise WatchError(f"games.yaml is invalid: {error}") from error

    config_sha256 = provenance.sha256_bytes(config_bytes)
    frame = provenance.frame_from_bytes(sessions_bytes, dtype=str, keep_default_na=False)
    sessions = game_monitor.prepare_sessions(frame)
    results, unmonitored = game_monitor.evaluate_all(sessions, effective)

    alerts = []

    for app_id, (_, episodes) in sorted(results.items()):
        game = effective[app_id]

        for episode in episodes:
            peak = episode["peak"]
            effective_baseline = peak["effective_baseline"]

            if not episode["proposal_eligible"]:
                status = "insufficient_baseline"
            elif not game["enforcement_enabled"]:
                status = "already_disabled"
            else:
                status = "proposal_eligible"

            identity = {"app_id": app_id, "start": episode["start"], "end": episode["end"],
                        "dataset_id": dataset_id, "config_sha256": config_sha256, "watcher": WATCHER_VERSION}

            alerts.append({
                "alert_id": "alert_" + digest(identity)[:16],
                "app_id": app_id,
                "game_name": game["name"],
                "status": status,
                "enforcement_enabled": game["enforcement_enabled"],
                "episode": {"start": episode["start"], "end": episode["end"], "alerting_windows": episode["windows"]},
                "peak_window": {
                    "start": peak["window_start"],
                    "end": peak["window_end"],
                    "sessions": peak["sessions"],
                    "anticheat_terminations": peak["terminations"],
                    "rate": round(peak["rate"], 6),
                    "baseline_sessions": peak["baseline_sessions"],
                    "baseline_anticheat_terminations": peak["baseline_terminations"],
                    "observed_baseline_rate": None if peak["observed_baseline"] is None else round(peak["observed_baseline"], 6),
                    "effective_baseline_rate": None if effective_baseline is None else round(effective_baseline, 6),
                    "floor_applied": peak["floor_applied"],
                    "baseline_sufficient": peak["baseline_sufficient"],
                    "times_effective_baseline": None if not effective_baseline else round(peak["rate"] / effective_baseline, 2),
                    "conditions": peak["conditions"],
                    "session_ids": sorted(peak["session_ids"].tolist()),
                    "anticheat_terminated_session_ids": sorted(peak["terminated_session_ids"].tolist()),
                },
                "thresholds": game["settings"],
                "possible_explanations": POSSIBLE_EXPLANATIONS,
                "cause": CAUSE_STATEMENT,
            })

    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "watcher": code_record(),
        "dataset_id": dataset_id,
        "config_sha256": config_sha256,
        "games_monitored": sorted(effective),
        "unmonitored_app_ids": unmonitored,
        "window_assignment": "by session_end, [start, end)",
        "counted": "end_reason terminated_anticheat only",
        "alerts": alerts,
        "summary": {status: sum(1 for alert in alerts if alert["status"] == status)
                    for status in ["proposal_eligible", "already_disabled", "insufficient_baseline"]},
        "disarming_means": (
            "A merged proposal sets one game's simulated enforcement_enabled flag to false. It does not stop "
            "monitoring, does not reverse existing bans, and sends nothing to a real game server."
        ),
    }

    return as_json(report)


def render_markdown(report):
    """The alert report for a person."""

    lines = [
        "# Game-level disarm watch",
        "",
        f"Dataset `{report['dataset_id'][:16]}…` · games.yaml `{report['config_sha256'][:16]}…` · "
        f"watcher {report['watcher']['version']}",
        f"Monitored: {len(report['games_monitored'])} games. "
        f"Unmonitored (in the data, not configured): {', '.join(report['unmonitored_app_ids']) or 'none'}.",
        f"Windows assigned {report['window_assignment']}; counting {report['counted']}.",
        "",
    ]

    if not report["alerts"]:
        lines += ["No game met the alert conditions.", ""]

    for alert in report["alerts"]:
        peak = alert["peak_window"]
        thresholds = alert["thresholds"]
        baseline = peak["observed_baseline_rate"]

        lines += [
            f"## {investigate_safe(alert['game_name'])} (`{alert['app_id']}`) — {alert['status'].replace('_', ' ')}",
            "",
            f"- **Alert:** `{alert['alert_id']}`",
            f"- **Episode:** {alert['episode']['start']} to {alert['episode']['end']} ({alert['episode']['alerting_windows']} alerting windows)",
            f"- **Peak window:** {peak['start']} to {peak['end']}",
            f"- **Anti-cheat terminations:** {peak['anticheat_terminations']} of {peak['sessions']} sessions ({peak['rate']:.1%})",
        ]

        if peak["baseline_sufficient"]:
            floor = " (floor applied)" if peak["floor_applied"] else ""
            lines.append(
                f"- **Earlier baseline:** {peak['baseline_anticheat_terminations']} of {peak['baseline_sessions']} sessions "
                f"({baseline:.2%}); effective {peak['effective_baseline_rate']:.2%}{floor}; current rate is "
                f"{peak['times_effective_baseline']}× the effective baseline"
            )
        else:
            lines.append(
                f"- **Earlier baseline:** insufficient ({peak['baseline_sessions']} sessions; needs "
                f"{thresholds['min_baseline_sessions']}). No disarm proposal can be made from this alert."
            )

        conditions = peak["conditions"]
        lines += [
            f"- **Conditions:** sessions ≥ {thresholds['min_sessions']}: {conditions['sessions']}; "
            f"terminations ≥ {thresholds['min_terminations']}: {conditions['terminations']}; "
            f"rate ≥ {thresholds['termination_rate_threshold']}: {conditions['rate']}; "
            f"rate ≥ {thresholds['baseline_multiplier']}× effective baseline: "
            f"{conditions['relative'] if peak['baseline_sufficient'] else 'not assessed (insufficient baseline)'}",
            f"- **Terminated sessions (first {SESSION_IDS_SHOWN} of {len(peak['anticheat_terminated_session_ids'])}):** "
            + ", ".join(peak["anticheat_terminated_session_ids"][:SESSION_IDS_SHOWN]),
            "",
            "**Possible explanations:**",
            *[f"- {item}" for item in alert["possible_explanations"]],
            "",
            alert["cause"],
            "",
        ]

    lines += [report["disarming_means"], ""]

    return "\n".join(lines)


def investigate_safe(text):
    """Game names are configuration text; keep control characters out of the report."""

    return "".join(f"\\u{ord(character):04x}" if ord(character) < 32 or 127 <= ord(character) < 160 else character
                   for character in str(text))


def write_scan(report, output_root=ROOT / "reports" / "disarm"):
    """Write alerts.json and alerts.md; the folder name is derived from the inputs."""

    scan_id = "scan_" + digest({"dataset": report["dataset_id"], "config": report["config_sha256"],
                                "watcher": report["watcher"]})[:12]
    folder = Path(output_root) / scan_id
    folder.mkdir(parents=True, exist_ok=True)

    document = {"metadata": {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, "report": report}
    (folder / "alerts.json").write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    (folder / "alerts.md").write_text(render_markdown(report), encoding="utf-8")

    return folder


# ------------------------------------------------------------
# The single-flag edit.
# ------------------------------------------------------------

def edit_flag(config_bytes, app_id):
    """Change exactly one line: the target game's enforcement_enabled from true to false."""

    text = config_bytes.decode("utf-8")
    lines = text.splitlines(keepends=True)
    game_line = GAME_LINE.format(app_id=app_id)

    starts = [index for index, line in enumerate(lines) if line.rstrip("\r\n") == game_line]

    if len(starts) != 1:
        raise WatchError(f"games.yaml must contain the line {game_line!r} exactly once (layout rule); found {len(starts)}")

    start = starts[0]
    end = next((index for index in range(start + 1, len(lines)) if lines[index].startswith("  - app_id:")), len(lines))

    flags = [index for index in range(start, end) if lines[index].rstrip("\r\n") in (FLAG_TRUE, FLAG_FALSE)]

    if len(flags) != 1:
        raise WatchError(f"{app_id}'s entry must have exactly one enforcement_enabled line in the documented layout")

    line = lines[flags[0]]

    if line.rstrip("\r\n") == FLAG_FALSE:
        raise WatchError(f"{app_id} already has enforcement disabled")

    lines[flags[0]] = FLAG_FALSE + line[len(FLAG_TRUE):]

    return "".join(lines).encode("utf-8")


def prove_single_flag_change(old_bytes, new_bytes, app_id):
    """Both versions valid, one line different, and the only semantic change is the target flag."""

    old_lines = old_bytes.decode("utf-8").splitlines(keepends=True)
    new_lines = new_bytes.decode("utf-8").splitlines(keepends=True)

    if len(old_lines) != len(new_lines):
        raise WatchError("The edit changed the number of lines")

    changed = [index for index, (old, new) in enumerate(zip(old_lines, new_lines)) if old != new]

    if len(changed) != 1:
        raise WatchError(f"The edit changed {len(changed)} lines; exactly one is allowed")

    try:
        old_config = game_config.parse(old_bytes)
        new_config = game_config.parse(new_bytes)
        game_config.validate(old_config)
        game_config.validate(new_config)
    except game_config.ConfigError as error:
        raise WatchError(f"A version of games.yaml is invalid: {error}") from error

    targets = [game for game in old_config["games"] if game["app_id"] == app_id]

    if len(targets) != 1 or targets[0]["enforcement_enabled"] is not True:
        raise WatchError(f"{app_id} must appear once with enforcement enabled before the edit")

    expected = copy.deepcopy(old_config)
    next(game for game in expected["games"] if game["app_id"] == app_id)["enforcement_enabled"] = False

    if new_config != expected:
        raise WatchError("The edit changed something other than the target game's enforcement_enabled flag")

    return changed[0] + 1


def merge_request_body(alert, branch, base_commit, config_path, line_number):
    peak = alert["peak_window"]

    return "\n".join([
        f"# Proposal: pause anti-cheat enforcement for {investigate_safe(alert['game_name'])} ({alert['app_id']})",
        "",
        "## What this changes",
        f"One line of `{config_path}` (line {line_number}): `enforcement_enabled: true` → `false` for {alert['app_id']}.",
        "Nothing else: no thresholds, no other games. This was checked by parsing both versions.",
        "",
        "## Why",
        f"Alert `{alert['alert_id']}`: {peak['anticheat_terminations']} of {peak['sessions']} sessions "
        f"({peak['rate']:.1%}) ended in anti-cheat termination between {peak['start']} and {peak['end']}, "
        f"{peak['times_effective_baseline']}× this game's effective baseline of {peak['effective_baseline_rate']:.2%} "
        f"({peak['baseline_anticheat_terminations']} of {peak['baseline_sessions']} earlier sessions"
        f"{', floor applied' if peak['floor_applied'] else ''}). All four alert conditions were met.",
        "",
        "## Replay, not a live decision",
        REPLAY_NOTE,
        "",
        "## What is not known",
        CAUSE_STATEMENT,
        *[f"- {item}" for item in POSSIBLE_EXPLANATIONS],
        "",
        "## Before merging",
        "- Check the anti-cheat team's view of this window and any recent game or signature update.",
        "- If a real cheat wave is plausible, do not merge: disarming would help cheaters.",
        "- Merging does not stop monitoring, does not reverse existing bans, and sends nothing to a real server.",
        "- Re-enabling later is a manual edit of the same line.",
        "",
        "## Provenance",
        f"- Branch `{branch}` from `{base_commit}`; this proposal is not merged automatically.",
        f"- Dataset `{alert['dataset_id']}`, games.yaml `{alert['config_sha256']}`.",
        f"- Thresholds: {json.dumps(alert['thresholds'], sort_keys=True)}",
        f"- Terminated sessions in the peak window: {len(peak['anticheat_terminated_session_ids'])}, "
        f"listed in the alert report.",
        "",
    ])


# ------------------------------------------------------------
# Git, without touching the working copy.
# ------------------------------------------------------------

def git(repo, *arguments, data=None, env=None):
    """Run git in repo; raise WatchError on failure."""

    result = subprocess.run(
        ["git", *arguments], cwd=repo, input=data, capture_output=True,
        env={**os.environ, **(env or {})},
    )

    if result.returncode != 0:
        raise WatchError(f"git {' '.join(arguments)} failed: {result.stderr.decode('utf-8', 'replace').strip()}")

    return result.stdout


def repository_root(path):
    """The repository containing path. Checked before anything is read or edited."""

    try:
        result = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=path, capture_output=True)
    except FileNotFoundError as error:
        raise WatchError("git is not installed") from error

    if result.returncode != 0:
        raise WatchError(f"{path} is not inside a Git repository; refusing to prepare an edit")

    return Path(result.stdout.decode("utf-8").strip()).resolve()


def committed_config(repo, relative):
    """The committed games.yaml at HEAD, its blob mode, and the HEAD commit."""

    head = git(repo, "rev-parse", "HEAD").decode().strip()
    listing = git(repo, "ls-tree", head, "--", relative).decode().split()

    if len(listing) < 4 or listing[1] != "blob":
        raise WatchError(f"{relative} is not committed at HEAD")

    return git(repo, "cat-file", "blob", listing[2]), listing[0], head


@contextlib.contextmanager
def edit_lock(repo):
    path = Path(repo) / git(repo, "rev-parse", "--git-path", "sadlite-disarm.lock").decode().strip()

    with open(path, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def identity_env(repo):
    """Use the repository's configured identity; fall back to a clearly synthetic one."""

    configured = subprocess.run(["git", "config", "user.email"], cwd=repo, capture_output=True).returncode == 0
    fallback = {"GIT_AUTHOR_NAME": "SAD-Lite disarm watch", "GIT_AUTHOR_EMAIL": "disarm-watch@sadlite.invalid",
                "GIT_COMMITTER_NAME": "SAD-Lite disarm watch", "GIT_COMMITTER_EMAIL": "disarm-watch@sadlite.invalid"}

    return {} if configured else fallback


def commit_on_branch(repo, relative, mode, new_bytes, head, branch, message):
    """Create branch at a new commit on HEAD containing new_bytes. Index and working tree untouched."""

    blob = git(repo, "hash-object", "-w", "--stdin", data=new_bytes).decode().strip()

    with tempfile.TemporaryDirectory(prefix="sadlite-index-") as folder:
        index = {"GIT_INDEX_FILE": str(Path(folder) / "index")}
        git(repo, "read-tree", head, env=index)
        git(repo, "update-index", "--cacheinfo", f"{mode},{blob},{relative}", env=index)
        tree = git(repo, "write-tree", env=index).decode().strip()

    commit = git(repo, "commit-tree", tree, "-p", head, data=message.encode("utf-8"), env=identity_env(repo)).decode().strip()

    # An empty old value means the branch must not already exist.
    git(repo, "update-ref", f"refs/heads/{branch}", commit, "")

    return commit


def branch_file(repo, branch, relative):
    """games.yaml on an existing branch, or None if the branch does not exist."""

    exists = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"], cwd=repo, capture_output=True)

    if exists.returncode != 0:
        return None

    return git(repo, "show", f"refs/heads/{branch}:{relative}")


# ------------------------------------------------------------
# Propose.
# ------------------------------------------------------------

REPLAY_NOTE = (
    "Replay of SAD-Lite synthetic data: the spike is in the dataset's past, not happening now. "
    "An old spike is not evidence that a real game should be disabled today."
)


def prepare_disarm(alert_id, data_dir, config_path, apply, output_root,
                   expected_dataset_id, expected_config_sha256, expected_app_id=None,
                   expected_alert=None, expected_watcher=None):
    """The one place a disarm proposal is made; used by `propose` and by agent_sim.py.

    It never takes the caller's word for a breach: the alert must be found in a
    fresh scan of the verified data and the committed games.yaml, and must match
    whatever the caller expected (dataset, configuration, game, and, from a
    report, the full alert).
    """

    config_path = Path(config_path).resolve()

    # 1. A Git repository, before anything is read or edited.
    repo = repository_root(config_path.parent)
    relative = config_path.relative_to(repo).as_posix()

    # 2. The committed configuration must be the one the evidence was computed against.
    committed, mode, head = committed_config(repo, relative)

    try:
        working = provenance.read_file_once(config_path)
    except provenance.ProvenanceError as error:
        raise WatchError(str(error)) from error

    if working != committed:
        raise WatchError(f"{relative} has uncommitted changes; commit or discard them before proposing")

    if provenance.sha256_bytes(committed) != expected_config_sha256:
        raise WatchError("games.yaml has changed since the alert was computed; re-run the scan")

    # 3. Re-run the scan on verified data and the committed configuration.
    sessions_bytes, dataset_id = load_sessions(data_dir)

    if dataset_id != expected_dataset_id:
        raise WatchError("The alert was computed on a different dataset than the one in the data directory")

    regenerated = scan(sessions_bytes, dataset_id, committed)
    alert = next((item for item in regenerated["alerts"] if item["alert_id"] == alert_id), None)

    if alert is None:
        raise WatchError(f"No alert {alert_id} in a fresh scan of this data and configuration (unknown or stale evidence)")

    if expected_alert is not None and (canonical(alert) != canonical(expected_alert)
                                       or regenerated["watcher"] != expected_watcher):
        raise WatchError("The alert does not match a fresh scan of this data and configuration; it may have been edited")

    if expected_app_id is not None and alert["app_id"] != expected_app_id:
        raise WatchError(f"Alert {alert_id} is for {alert['app_id']}, not {expected_app_id}")

    if alert["status"] != "proposal_eligible":
        raise WatchError(f"Alert {alert_id} is {alert['status'].replace('_', ' ')}; no disarm proposal can be made from it")

    # 4. The edit, and proof that it is the only change.
    new_bytes = edit_flag(committed, alert["app_id"])
    line_number = prove_single_flag_change(committed, new_bytes, alert["app_id"])

    branch = f"disarm/{alert['app_id']}-{alert_id[len('alert_'):len('alert_') + 8]}"
    alert_with_context = {**alert, "dataset_id": dataset_id, "config_sha256": expected_config_sha256}
    body = merge_request_body(alert_with_context, branch, head, relative, line_number)

    proposal = {
        "alert_id": alert_id, "app_id": alert["app_id"], "branch": branch, "base_commit": head,
        "file": relative, "line": line_number, "change": f"{FLAG_TRUE.strip()} -> {FLAG_FALSE.strip()}",
        "diff": [f"@@ {relative} line {line_number} @@", f"-{FLAG_TRUE}", f"+{FLAG_FALSE}"],
        "mode": "replay", "replay_note": REPLAY_NOTE,
        "merge_request_body": body,
    }

    if not apply:
        return {"status": "dry_run", **proposal}

    # 5. Commit under the lock, after re-checking the committed file.
    with edit_lock(repo):
        recheck, _, recheck_head = committed_config(repo, relative)

        if recheck != committed or recheck_head != head:
            raise WatchError("HEAD or the committed games.yaml changed while preparing the proposal; re-run")

        folder = Path(output_root) / "proposals"
        body_path = folder / f"merge_request_{alert_id}.md"
        existing = branch_file(repo, branch, relative)

        if existing is not None:
            if existing == new_bytes:
                return {"status": "already_proposed", "merge_request_path": str(body_path), **proposal}

            raise WatchError(f"Branch {branch} already exists with different contents")

        message = (f"Propose disarming anti-cheat enforcement for {alert['app_id']}\n\n"
                   f"Alert {alert_id}. One line changed: {relative} line {line_number}.\n"
                   f"Replay of synthetic data. Not merged automatically; see the merge-request description.\n")
        commit = commit_on_branch(repo, relative, mode, new_bytes, head, branch, message)

    folder.mkdir(parents=True, exist_ok=True)
    body_path.write_text(body, encoding="utf-8")

    return {"status": "proposed", "commit": commit, "merge_request_path": str(body_path), **proposal}


def propose(report_path, alert_id, data_dir=ROOT / "data", config_path=DEFAULT_CONFIG, apply=False,
            output_root=ROOT / "reports" / "disarm"):
    """Command-line proposal from a written alert report; the report must match a fresh scan."""

    # A Git repository, before anything is read.
    repository_root(Path(config_path).resolve().parent)

    try:
        document = json.loads(provenance.read_file_once(report_path).decode("utf-8"))
        submitted_report = document["report"]
        submitted = next(alert for alert in submitted_report["alerts"] if alert["alert_id"] == alert_id)
    except StopIteration as error:
        raise WatchError(f"No alert {alert_id} in {report_path}") from error
    except (provenance.ProvenanceError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise WatchError(f"Cannot read an alert report from {report_path}: {error}") from error

    return prepare_disarm(alert_id, data_dir, config_path, apply, output_root,
                          expected_dataset_id=submitted_report.get("dataset_id"),
                          expected_config_sha256=submitted_report.get("config_sha256"),
                          expected_alert=submitted, expected_watcher=submitted_report.get("watcher"))


# ------------------------------------------------------------
# Command line.
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Watch games for termination spikes; propose a disarm edit for review.")
    commands = parser.add_subparsers(dest="command", required=True)

    scan_parser = commands.add_parser("scan", help="check every game and window; write the alert report")
    scan_parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    scan_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    propose_parser = commands.add_parser("propose", help="prepare the single-flag disarm edit for one alert")
    propose_parser.add_argument("--report", type=Path, required=True)
    propose_parser.add_argument("--alert-id", required=True)
    propose_parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    propose_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    propose_parser.add_argument("--apply", action="store_true", help="commit the edit on a new branch (never merged)")

    args = parser.parse_args()

    try:
        if args.command == "scan":
            sessions_bytes, dataset_id = load_sessions(args.data_dir)

            try:
                config_bytes = provenance.read_file_once(args.config)
            except provenance.ProvenanceError as error:
                raise WatchError(str(error)) from error

            report = scan(sessions_bytes, dataset_id, config_bytes)
            folder = write_scan(report)

            print(render_markdown(report))
            print(f"Alert report: {folder / 'alerts.json'} and {folder / 'alerts.md'}")
        else:
            result = propose(args.report, args.alert_id, args.data_dir, args.config, args.apply)

            print(result["merge_request_body"])
            print(json.dumps({key: value for key, value in result.items() if key != "merge_request_body"}, indent=2))

            if result["status"] == "dry_run":
                print("\nDry run: nothing was written. Repeat with --apply to commit the edit on a new branch.")
    except WatchError as error:
        print(f"Refused, nothing written: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()