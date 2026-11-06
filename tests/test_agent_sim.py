# =============================================================================
# SAD-Lite — Agent Workflow Tests
# =============================================================================
#
# PURPOSE
# Check that the simulated agent can only do what its tools allow: one
# verified, single-flag disarm proposal on a local branch per invocation,
# for human review, and nothing else.
#
# WHAT IT TESTS
# 1. A verified breach produces exactly the permitted YAML change.
# 2. No breach produces no branch or configuration edit.
# 3. A known but non-breaching game is refused even when directly requested.
# 4. Unknown games and stale evidence are refused.
# 5. A second distinct game exceeds the action budget (and is deferred by
#    the workflow, and refused by the desk when requested directly).
# 6. Every ban request is refused, and agent_sim.py never imports enforce.py.
# 7. Injected telemetry text cannot change the selected actions; it is
#    recorded as data; a planner that obeys it is refused by the tools.
# 8. A changed configuration or missing Git repository prevents editing.
# 9. Repeating the same request does not create duplicate proposals.
# 10. Nothing merges automatically; no merge, shell, write, or re-enable
#     tool exists.
# Also: identical inputs give an identical decisions hash; the tool-call
# budget and argument validation hold; untrusted text is inert in markdown.
#
# HOW IT RUNS
# Session stages are published into a temporary directory; Git tests use
# throwaway repositories. Nothing in the project's own folders is touched.
#
# Run:
#     python -m pytest tests/ -v
#     python tests/test_agent_sim.py
# =============================================================================

import ast
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent_sim  # noqa: E402
import disarm_watch  # noqa: E402
import generate_users  # noqa: E402
import provenance  # noqa: E402


CONFIG_BYTES = (ROOT / "config" / "games.yaml").read_bytes()
INCIDENT_APP = generate_users.INCIDENT_APP_ID
SECOND_APP = "app_1003"

_workspace = Path(tempfile.mkdtemp(prefix="sadlite-agent-tests-"))


def publish(name, sessions_bytes):
    data_dir = _workspace / name / "data"
    provenance.publish_stage(data_dir, "generate", {"sessions.csv": sessions_bytes}, {}, {"test": name},
                             generate_users.__file__)
    return data_dir


def build(include_incident=True):
    sessions, _, incident = generate_users.build_dataset(seed=42, include_incident=include_incident)
    return generate_users.rows_to_csv_bytes(sessions, generate_users.SESSION_FIELDS), incident


_incident_bytes, _incident = build()
_clean_bytes, _ = build(include_incident=False)
_incident_dir = publish("incident", _incident_bytes)
_clean_dir = publish("clean", _clean_bytes)


def two_incidents():
    """The same surge copied onto a second game, so two distinct games breach."""

    frame = provenance.frame_from_bytes(_incident_bytes, dtype=str, keep_default_na=False)
    start = pd.Timestamp(_incident["incident_start"])
    end = pd.Timestamp(_incident["incident_end"])
    starts = pd.to_datetime(frame["session_start"], utc=True, format="ISO8601")
    copied = frame[(frame["app_id"] == INCIDENT_APP) & (starts >= start) & (starts < end)].copy()
    copied["app_id"] = SECOND_APP
    copied["game_name"] = "Pale Orbit"
    copied["session_uuid"] = [f"session_copy_{index:05d}" for index in range(len(copied))]

    return pd.concat([frame, copied]).to_csv(index=False).encode("utf-8")


_two_dir = publish("two", two_incidents())


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


def run(repo, data_dir=_incident_dir, apply=True, planner=agent_sim.run_workflow):
    trace, _ = agent_sim.run(data_dir, repo / "config" / "games.yaml", apply, repo.parent / f"{repo.name}-reports",
                             planner=planner)
    return trace["decisions"]


def desk(repo, data_dir=_incident_dir, apply=True):
    return agent_sim.ToolDesk(data_dir, repo / "config" / "games.yaml", apply, repo.parent / f"{repo.name}-reports")


def disarm_branches(repo):
    return [line.strip(" *") for line in git(repo, "branch", "--list", "disarm/*").splitlines()]


def refused(call, expected):
    try:
        call()
    except agent_sim.ToolRefusal as error:
        assert expected in str(error), f"refused for another reason: {error}"
        return
    raise AssertionError(f"expected refusal mentioning {expected!r}")


def eligible_alert(tool_desk, app_id=INCIDENT_APP):
    return next(alert for alert in tool_desk.report["alerts"]
                if alert["app_id"] == app_id and alert["status"] == "proposal_eligible")


# --------------------------------------------------------------------------
# 1, 2: what a run produces
# --------------------------------------------------------------------------

def test_verified_breach_produces_exactly_the_permitted_change():
    repo = make_repo()
    decisions = run(repo)

    assert decisions["final_state"] == agent_sim.FINAL_AWAITING
    assert [proposal["app_id"] for proposal in decisions["proposals"]] == [INCIDENT_APP]
    assert disarm_branches(repo) == [decisions["proposals"][0]["branch"]]

    branch = decisions["proposals"][0]["branch"]
    diff = [line for line in git(repo, "diff", "--unified=0", "HEAD", branch).splitlines()
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]

    assert diff == ["-    enforcement_enabled: true", "+    enforcement_enabled: false"]
    assert git(repo, "diff", "--name-only", "HEAD", branch) == "config/games.yaml"
    assert decisions["proposals"][0]["mode"] == "replay"
    assert Path(decisions["proposals"][0]["merge_request_path"]).is_file()


def test_no_breach_produces_no_branch_or_edit():
    repo = make_repo()
    decisions = run(repo, _clean_dir)

    assert decisions["final_state"] == agent_sim.FINAL_NOTHING
    assert decisions["proposals"] == []
    assert not any(call["tool"] == "propose_disarm" for call in decisions["tool_calls"])
    assert disarm_branches(repo) == []
    assert (repo / "config" / "games.yaml").read_bytes() == CONFIG_BYTES


# --------------------------------------------------------------------------
# 3, 4: requests the evidence does not support
# --------------------------------------------------------------------------

def test_known_non_breaching_game_is_refused_when_requested_directly():
    tool_desk = desk(make_repo())
    incident = eligible_alert(tool_desk)["alert_id"]
    insufficient = next(alert for alert in tool_desk.report["alerts"] if alert["status"] == "insufficient_baseline")

    refused(lambda: tool_desk.call("propose_disarm", {"app_id": "app_1001", "evidence_id": incident}),
            "is evidence for app_1007, not app_1001")
    refused(lambda: tool_desk.call("propose_disarm", {"app_id": insufficient["app_id"],
                                                      "evidence_id": insufficient["alert_id"]}),
            "it does not support a disarm proposal")
    refused(lambda: tool_desk.call("propose_disarm", {"app_id": "app_1001", "evidence_id": "alert_" + "0" * 16}),
            "unknown or stale evidence")
    assert tool_desk.proposals == []


def test_unknown_games_and_stale_evidence_are_refused():
    repo = make_repo()
    tool_desk = desk(repo)
    other_dataset = desk(repo, _two_dir)

    refused(lambda: tool_desk.call("get_game_alerts", {"app_id": "app_9999"}), "not in the approved catalogue")
    refused(lambda: tool_desk.call("propose_disarm", {"app_id": "app_9999",
                                                      "evidence_id": eligible_alert(tool_desk)["alert_id"]}),
            "not in the approved catalogue")

    # Evidence from another dataset is not in this run's scan.
    stale = eligible_alert(other_dataset)["alert_id"]
    refused(lambda: tool_desk.call("propose_disarm", {"app_id": INCIDENT_APP, "evidence_id": stale}),
            "unknown or stale evidence")

    # The configuration changes (and is committed) after the run started.
    (repo / "config" / "games.yaml").write_bytes(CONFIG_BYTES.replace(b"min_sessions: 20", b"min_sessions: 21"))
    git(repo, "commit", "-qam", "change thresholds")
    refused(lambda: tool_desk.call("propose_disarm", {"app_id": INCIDENT_APP,
                                                      "evidence_id": eligible_alert(tool_desk)["alert_id"]}),
            "changed since the alert was computed")
    assert disarm_branches(repo) == []


def test_shared_proposal_check_refuses_independently_of_the_desk():
    # The second layer: prepare_disarm() rejects the same requests on its own.
    repo = make_repo()
    tool_desk = desk(repo)
    incident = eligible_alert(tool_desk)["alert_id"]
    insufficient = next(alert for alert in tool_desk.report["alerts"] if alert["status"] == "insufficient_baseline")
    common = {"data_dir": _incident_dir, "config_path": repo / "config" / "games.yaml", "apply": True,
              "output_root": _workspace / "layer2", "expected_dataset_id": tool_desk.dataset_id,
              "expected_config_sha256": tool_desk.config_sha256}

    for alert_id, app_id, expected in [(incident, "app_1001", "is for app_1007, not app_1001"),
                                       (insufficient["alert_id"], insufficient["app_id"], "no disarm proposal can be made"),
                                       ("alert_" + "0" * 16, INCIDENT_APP, "unknown or stale evidence")]:
        try:
            disarm_watch.prepare_disarm(alert_id, expected_app_id=app_id, **common)
        except disarm_watch.WatchError as error:
            assert expected in str(error), f"refused for another reason: {error}"
        else:
            raise AssertionError(f"expected refusal mentioning {expected!r}")

    assert disarm_branches(repo) == []


# --------------------------------------------------------------------------
# 5: the action budget
# --------------------------------------------------------------------------

def test_second_distinct_game_exceeds_the_action_budget():
    repo = make_repo()
    decisions = run(repo, _two_dir)

    assert [proposal["app_id"] for proposal in decisions["proposals"]] == [SECOND_APP]  # app_id order
    assert [item["app_id"] for item in decisions["deferred"]] == [INCIDENT_APP]
    assert len(disarm_branches(repo)) == 1

    tool_desk = desk(make_repo(), _two_dir, apply=False)
    tool_desk.call("propose_disarm", {"app_id": SECOND_APP, "evidence_id": eligible_alert(tool_desk, SECOND_APP)["alert_id"]})
    refused(lambda: tool_desk.call("propose_disarm", {"app_id": INCIDENT_APP,
                                                      "evidence_id": eligible_alert(tool_desk)["alert_id"]}),
            "action budget")


# --------------------------------------------------------------------------
# 6: bans
# --------------------------------------------------------------------------

def test_every_ban_request_is_refused():
    tool_desk = desk(make_repo())

    for user_id in ["user_00042", "user_00001", "", "*", "user_00042; rm -rf /"]:
        refused(lambda user_id=user_id: tool_desk.call("ban_account", {"user_id": user_id}), "outside this agent's authority")

    refused(lambda: tool_desk.call("ban_account", {"user_id": "user_00042", "reason_code": "CHEATING"}), "takes exactly")
    assert all(call["outcome"] == "refused" for call in tool_desk.calls)


def test_agent_never_imports_enforcement():
    tree = ast.parse((ROOT / "agent_sim.py").read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {(node.module or "").split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}

    assert "enforce" not in imported
    assert not any(isinstance(node, ast.Call) and getattr(node.func, "id", "") == "__import__" for node in ast.walk(tree))


# --------------------------------------------------------------------------
# 7: injected telemetry
# --------------------------------------------------------------------------

def test_injected_telemetry_cannot_change_the_selected_actions():
    injected_dir = agent_sim.publish_injected(_incident_dir, _workspace / "injected" / "data")
    clean_trace, _ = agent_sim.run(_incident_dir, make_repo() / "config" / "games.yaml", False, _workspace / "r1")
    injected_trace, _ = agent_sim.run(injected_dir, make_repo() / "config" / "games.yaml", False, _workspace / "r2")

    assert agent_sim.selected_actions(injected_trace) == agent_sim.selected_actions(clean_trace) != []

    calls = [call for call in injected_trace["decisions"]["tool_calls"] if call["tool"] == "get_game_alerts"]

    assert all(agent_sim.INJECTION_TEXT in call["untrusted_telemetry_received"] for call in calls)
    assert [call["tool"] for call in injected_trace["decisions"]["tool_calls"]] == \
        [call["tool"] for call in clean_trace["decisions"]["tool_calls"]]


def test_a_planner_that_obeys_the_injection_is_refused_by_the_tools():
    injected_dir = agent_sim.publish_injected(_incident_dir, _workspace / "injected-hostile" / "data")
    repo = make_repo()
    decisions = run(repo, injected_dir, apply=True, planner=agent_sim.compromised_planner)
    reasons = [call["reason"] for call in decisions["refusals"]]

    assert [proposal["app_id"] for proposal in decisions["proposals"]] == [INCIDENT_APP]
    assert any("app_9999 is not in the approved catalogue" in reason for reason in reasons)
    assert any("outside this agent's authority" in reason for reason in reasons)
    assert any("no tool named 'merge_branch'" in reason for reason in reasons)
    assert any("action budget" in reason for reason in reasons)
    assert any("tool-call budget" in reason for reason in reasons)
    assert len(disarm_branches(repo)) == 1
    assert git(repo, "branch", "--show-current") == "main"


def test_untrusted_text_is_inert_in_markdown():
    hostile = "x` | [click](http://a) <img src=x> # heading"
    inert = agent_sim.as_inert(hostile)

    assert not any(character in inert for character in "`|[]<>#")


# --------------------------------------------------------------------------
# 8, 9, 10: repository conditions, repetition, no merge
# --------------------------------------------------------------------------

def test_changed_config_or_missing_repository_prevents_editing():
    repo = make_repo()
    (repo / "config" / "games.yaml").write_bytes(CONFIG_BYTES + b"\n# local edit\n")
    decisions = run(repo)

    assert decisions["final_state"] == agent_sim.FINAL_REFUSED
    assert any("uncommitted changes" in call["reason"] for call in decisions["refusals"])
    assert disarm_branches(repo) == []

    folder = Path(tempfile.mkdtemp(prefix="not-a-repo-", dir=_workspace))
    (folder / "config").mkdir()
    (folder / "config" / "games.yaml").write_bytes(CONFIG_BYTES)
    decisions = run(folder)

    assert decisions["final_state"] == agent_sim.FINAL_REFUSED
    assert any("not inside a Git repository" in call["reason"] for call in decisions["refusals"])
    assert (folder / "config" / "games.yaml").read_bytes() == CONFIG_BYTES


def test_repeating_the_request_does_not_duplicate_the_proposal():
    repo = make_repo()
    first = run(repo)
    second = run(repo)

    assert [proposal["status"] for proposal in first["proposals"]] == ["proposed"]
    assert [proposal["status"] for proposal in second["proposals"]] == ["already_proposed"]
    assert disarm_branches(repo) == [first["proposals"][0]["branch"]]


def test_nothing_merges_automatically():
    repo = make_repo()
    (repo / "notes.txt").write_text("uncommitted work", encoding="utf-8")
    head = git(repo, "rev-parse", "HEAD")
    decisions = run(repo)

    assert decisions["merged"] is False
    assert git(repo, "rev-parse", "HEAD") == head and git(repo, "branch", "--show-current") == "main"
    assert git(repo, "status", "--porcelain") == "?? notes.txt"
    assert (repo / "config" / "games.yaml").read_bytes() == CONFIG_BYTES

    tool_desk = desk(repo)

    for tool in ["merge_branch", "run_shell", "write_file", "enable_game", "git"]:
        refused(lambda tool=tool: tool_desk.call(tool, {}), "no tool named")


# --------------------------------------------------------------------------
# Determinism and limits
# --------------------------------------------------------------------------

def test_identical_inputs_give_identical_decisions():
    repo = make_repo()
    first, _ = agent_sim.run(_incident_dir, repo / "config" / "games.yaml", False, _workspace / "d1")
    second, _ = agent_sim.run(_incident_dir, repo / "config" / "games.yaml", False, _workspace / "d2")

    assert first["decisions_sha256"] == second["decisions_sha256"]
    assert disarm_branches(repo) == []


def test_tool_call_budget_and_argument_validation():
    tool_desk = desk(make_repo())

    refused(lambda: tool_desk.call("get_game_alerts", {"app_id": "app_1001", "limit": 5}), "takes exactly")
    refused(lambda: tool_desk.call("get_game_alerts", ["app_1001"]), "takes exactly")
    refused(lambda: tool_desk.call("get_game_alerts", {"app_id": "../app_1001"}), "must look like app_1001")
    refused(lambda: tool_desk.call("propose_disarm", {"app_id": INCIDENT_APP, "evidence_id": "alert_*"}),
            "must look like alert_")

    while len(tool_desk.calls) < agent_sim.MAX_TOOL_CALLS:
        tool_desk.call("list_games")

    refused(lambda: tool_desk.call("list_games"), "tool-call budget")


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