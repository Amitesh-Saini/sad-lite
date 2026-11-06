# =============================================================================
# SAD-Lite — Enforcement Tests
# =============================================================================
#
# PURPOSE
# Try to make enforce.py record a decision it should refuse, lose evidence,
# duplicate an action, or corrupt its log, and check that each attempt fails
# in the documented way.
#
# WHY THIS IS NEEDED
# enforce.py is the only component that acts against an account. Its failures
# are the ones an appeal finds: a ban on edited evidence, a ban whose evidence
# cannot be re-checked, a duplicate, or a decision log that no longer adds up.
#
# WHAT IT TESTS
# - Requests: dry run writes nothing; dry run and apply are the same request;
#   exact retries after a commit return the original; a reused key with a
#   different request is refused; notes and codes are validated.
# - Evidence: edited reports, reports from another dataset, a changed
#   investigator, and reports with an unverified stage are refused; entity
#   reports and the wrong account are refused.
# - Policy: a reason the report does not support is refused; an override is
#   recorded; an override never bypasses report or provenance checks.
# - State: already-banned accounts; ban, reverse, ban again; concurrent
#   requests serialised by the lock.
# - Reversal: works after data/ is regenerated; works with damaged evidence
#   and records the failure; records whether typed reviewer names match.
# - Preservation: game-window evidence reproduces from a bundle alone;
#   bundles are never overwritten and their identity covers every stage.
# - Crash recovery: crash between evidence and decision; torn trailing
#   record; a valid record missing its newline; a complete invalid record;
#   a log ahead of its head; a log shorter than its head.
#
# HOW IT RUNS
# Two pipelines (seeds 42 and 7) are published into a temporary directory
# once, with case reports written there. Every test uses its own fresh
# enforcement directory. Nothing in the project's own folders is touched.
#
# Run:
#     python -m pytest tests/ -v
#     python tests/test_enforce.py      # works without pytest installed
# =============================================================================

import json
import shutil
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import enforce  # noqa: E402
import investigate  # noqa: E402
import provenance  # noqa: E402
from pipeline_fixture import publish_model, publish_pipeline  # noqa: E402


NOTES = "Shared device and payment with many accounts created within two days."
REVIEWER = "Reviewer One"

_workspace = Path(tempfile.mkdtemp(prefix="sadlite-enforce-tests-"))
_main = _workspace / "seed_42"
_other = _workspace / "seed_7"

_labels, _incident = publish_pipeline(_main, 42)
publish_pipeline(_other, 7)


def account_of(pattern, position=0):
    """Choose subjects by group. Labels are read only here, never by the code under test."""

    return sorted(_labels.loc[_labels["pattern"] == pattern, "user_id"])[position]


SMURF = account_of("smurf_ring")
SMURF_2 = account_of("smurf_ring", 1)
SMURF_3 = account_of("smurf_ring", 2)

# An incident participant whose only terminations were during the game-wide spike.
INCIDENT = None


def write_case(data_dir, subject_type, subject_value, folder):
    """Investigate from data_dir and write the case files, as investigate.py's main() does."""

    report = investigate.investigate(investigate.load_inputs(data_dir), subject_type, subject_value)
    _, _, json_path = investigate.write_report(report, folder)
    return json_path, report


_cases = _workspace / "cases"
SMURF_REPORT, _ = write_case(_main, "user_id", SMURF, _cases)
SMURF_2_REPORT, _ = write_case(_main, "user_id", SMURF_2, _cases)
SMURF_3_REPORT, _ = write_case(_main, "user_id", SMURF_3, _cases)

for participant in _incident["participants"]:
    path, report = write_case(_main, "user_id", participant, _cases)

    names = {item["name"] for item in report["hypotheses"]}

    if "Caught in a game-wide termination spike" in names and "Cheating" not in names:
        INCIDENT, INCIDENT_REPORT = participant, path
        break


def fresh_enforcement():
    return Path(tempfile.mkdtemp(prefix="enforcement-", dir=_workspace)) / "enforcement"


def ban(enforcement_dir, report=None, user_id=SMURF, reason="MULTI_ACCOUNTING", key="req_00000001",
        apply=True, data_dir=None, **options):
    return enforce.ban(report or SMURF_REPORT, data_dir or _main, user_id, reason, options.pop("reviewer", REVIEWER),
                       options.pop("notes", NOTES), key, apply, options.pop("override_justification", None), enforcement_dir)


def reverse(enforcement_dir, decision_id, key="req_r0000001", reviewer="Reviewer Two", apply=True):
    return enforce.reverse(decision_id, "INSUFFICIENT_EVIDENCE", reviewer,
                           "Appeal showed the shared device was a gaming café terminal.", key, apply, enforcement_dir)


def refused(call, expected_text, error=enforce.EnforcementError):
    try:
        call()
    except error as caught:
        assert expected_text in str(caught), f"refused for another reason: {caught}"
        return
    raise AssertionError(f"expected a refusal mentioning {expected_text!r}")


def log_lines(enforcement_dir):
    path = enforcement_dir / enforce.LOG_NAME
    return path.read_bytes().splitlines() if path.exists() else []


def edited_report(field_path, value):
    """A copy of the smurf case with one field changed."""

    document = json.loads(SMURF_REPORT.read_text(encoding="utf-8"))
    target = document["report"]

    for key in field_path[:-1]:
        target = target[key]

    target[field_path[-1]] = value
    path = _workspace / f"edited-{len(list(_workspace.glob('edited-*')))}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------

def test_dry_run_writes_nothing():
    enforcement_dir = fresh_enforcement()
    result = ban(enforcement_dir, key=None, apply=False)

    assert result["status"] == "dry_run"
    assert result["suggested_request_key"].startswith("req_")
    assert not enforcement_dir.exists()


def test_apply_preserves_evidence_and_records_the_decision():
    enforcement_dir = fresh_enforcement()
    result = ban(enforcement_dir)

    assert result["status"] == "applied"
    assert Path(result["bundle"]).is_dir()
    assert len(log_lines(enforcement_dir)) == 1

    dataset_id = json.loads(log_lines(enforcement_dir)[0])["dataset_id"]
    state = enforce.status(dataset_id, SMURF, enforcement_dir)

    assert state["banned"] and state["evidence"] == "verified"
    assert enforce.verify(enforcement_dir)["evidence_problems"] == {}


def test_dry_run_and_apply_are_the_same_request():
    enforcement_dir = fresh_enforcement()
    dry = ban(enforcement_dir, apply=False)
    ban(enforcement_dir)

    entry = json.loads(log_lines(enforcement_dir)[0])

    assert dry["proposal"]["request_sha256"] == entry["request_sha256"]


def test_exact_retry_after_commit_returns_the_original():
    enforcement_dir = fresh_enforcement()
    first = ban(enforcement_dir)
    second = ban(enforcement_dir)

    assert second["status"] == "already_applied"
    assert second["decision_id"] == first["decision_id"]
    assert len(log_lines(enforcement_dir)) == 1


def test_reused_key_with_a_different_request_is_refused():
    enforcement_dir = fresh_enforcement()
    ban(enforcement_dir)

    refused(lambda: ban(enforcement_dir, notes="A different explanation of the same decision entirely."), "already used for a different request")
    assert len(log_lines(enforcement_dir)) == 1


def test_apply_needs_a_request_key():
    refused(lambda: ban(fresh_enforcement(), key=None), "--apply needs --request-key")


def test_notes_must_explain_the_decision():
    refused(lambda: ban(fresh_enforcement(), notes="too short"), "must explain")
    refused(lambda: ban(fresh_enforcement(), notes="MULTI_ACCOUNTING"), "must explain")
    refused(lambda: ban(fresh_enforcement(), reason="BECAUSE_I_SAID_SO"), "Unknown ban code")


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------

def test_edited_report_is_refused():
    report = edited_report(["hypotheses", 0, "statement"], "Edited to look worse.")

    refused(lambda: ban(fresh_enforcement(), report=report), "does not match a fresh investigation")


def test_report_from_another_dataset_is_refused():
    refused(lambda: ban(fresh_enforcement(), data_dir=_other), "supply the data directory the report was produced from")


def test_report_from_another_investigator_version_is_refused():
    original = investigate.INVESTIGATOR_VERSION
    investigate.INVESTIGATOR_VERSION = "9.9.9"

    try:
        refused(lambda: ban(fresh_enforcement()), "different investigator version")
    finally:
        investigate.INVESTIGATOR_VERSION = original


def test_ban_needs_all_four_stages_verified():
    data_dir = _workspace / "no-model" / "data"
    shutil.copytree(_main, data_dir)
    shutil.rmtree(data_dir / "model")
    report, _ = write_case(data_dir, "user_id", SMURF, _workspace / "no-model" / "cases")

    refused(lambda: ban(fresh_enforcement(), report=report, data_dir=data_dir), "Stage 'model'")


def test_entity_report_and_wrong_account_are_refused():
    sessions = provenance.frame_from_bytes(provenance.load_stage(_main, "generate", ["sessions.csv"])[1]["sessions.csv"], dtype=str)
    device = sessions.loc[sessions["user_id"] == SMURF, "device_id"].iloc[0]
    entity_report, _ = write_case(_main, "device_id", device, _workspace / "entity-cases")

    refused(lambda: ban(fresh_enforcement(), report=entity_report), "Only a completed account case report")
    refused(lambda: ban(fresh_enforcement(), user_id=SMURF_2), "The report covers")


# --------------------------------------------------------------------------
# Evidence policy and the override
# --------------------------------------------------------------------------

def test_reason_the_report_does_not_support_is_refused():
    assert INCIDENT is not None, "fixture needs an incident participant with no cheating hypothesis"

    refused(lambda: ban(fresh_enforcement(), report=INCIDENT_REPORT, user_id=INCIDENT, reason="CHEATING"), "needs the hypothesis 'Cheating'")


def test_override_is_recorded_with_its_justification():
    enforcement_dir = fresh_enforcement()
    justification = "Anti-cheat team confirmed a memory-editing tool on this account."

    ban(enforcement_dir, report=INCIDENT_REPORT, user_id=INCIDENT, reason="CHEATING", override_justification=justification)
    entry = json.loads(log_lines(enforcement_dir)[0])

    assert entry["override"]["justification"] == justification
    assert entry["hypothesis"]["confidence"] is None


def test_override_never_bypasses_report_verification():
    report = edited_report(["hypotheses", 0, "confidence"], "Edited")

    refused(lambda: ban(fresh_enforcement(), report=report,
                        override_justification="Trying to push an edited report through the override."),
            "does not match a fresh investigation")


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

def test_already_banned_account_is_refused():
    enforcement_dir = fresh_enforcement()
    ban(enforcement_dir)

    refused(lambda: ban(enforcement_dir, key="req_00000002"), "already banned")


def test_ban_reverse_and_ban_again():
    enforcement_dir = fresh_enforcement()
    first = ban(enforcement_dir)
    reverse(enforcement_dir, first["decision_id"])
    second = ban(enforcement_dir, key="req_00000002")

    assert second["status"] == "applied"
    assert second["decision_id"] != first["decision_id"]
    assert len(log_lines(enforcement_dir)) == 3


def test_concurrent_requests_are_serialised():
    enforcement_dir = fresh_enforcement()
    results = []

    def attempt(key):
        try:
            results.append(ban(enforcement_dir, key=key)["status"])
        except enforce.EnforcementError as error:
            results.append(str(error))

    threads = [threading.Thread(target=attempt, args=(f"req_concurrent{index}",)) for index in range(2)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert results.count("applied") == 1
    assert any("already banned" in item for item in results)
    assert len(log_lines(enforcement_dir)) == 1


# --------------------------------------------------------------------------
# Reversal
# --------------------------------------------------------------------------

def test_reversal_works_after_data_is_regenerated():
    data_dir = _workspace / "regenerated" / "data"
    shutil.copytree(_main, data_dir)
    report, _ = write_case(data_dir, "user_id", SMURF, _workspace / "regenerated" / "cases")

    enforcement_dir = fresh_enforcement()
    decision = ban(enforcement_dir, report=report, data_dir=data_dir)
    shutil.rmtree(data_dir)

    result = reverse(enforcement_dir, decision["decision_id"])

    assert result["status"] == "applied"
    assert result["evidence_integrity"]["verified"]


def test_reversal_works_with_damaged_evidence_and_records_it():
    enforcement_dir = fresh_enforcement()
    decision = ban(enforcement_dir)
    (Path(decision["bundle"]) / "report.md").write_text("damaged", encoding="utf-8")

    dataset_id = json.loads(log_lines(enforcement_dir)[0])["dataset_id"]
    assert enforce.status(dataset_id, SMURF, enforcement_dir)["evidence"] == "INTEGRITY FAILURE"

    result = reverse(enforcement_dir, decision["decision_id"])
    entry = json.loads(log_lines(enforcement_dir)[1])

    assert result["status"] == "applied"
    assert not entry["evidence_integrity"]["verified"]
    assert any("report.md" in problem for problem in entry["evidence_integrity"]["problems"])


def test_reversal_works_when_the_bundle_index_is_corrupted():
    """Malformed or wrongly shaped index.json must be reported, not raised."""

    for damage in [b"{not json", b"[]", b'{"files": "x", "sessions_sha256": 1}',
                   b'{"files": {"../../decisions.jsonl": "0"}, "sessions_sha256": "../x"}']:
        enforcement_dir = fresh_enforcement()
        decision = ban(enforcement_dir)
        (Path(decision["bundle"]) / "index.json").write_bytes(damage)

        dataset_id = json.loads(log_lines(enforcement_dir)[0])["dataset_id"]

        assert enforce.status(dataset_id, SMURF, enforcement_dir)["evidence"] == "INTEGRITY FAILURE"
        assert decision["decision_id"] in enforce.verify(enforcement_dir)["evidence_problems"]

        result = reverse(enforcement_dir, decision["decision_id"])

        assert result["status"] == "applied", f"reversal blocked by index {damage!r}"
        assert not result["evidence_integrity"]["verified"]


def test_same_reviewer_reversal_is_allowed_and_recorded():
    enforcement_dir = fresh_enforcement()
    decision = ban(enforcement_dir)
    reverse(enforcement_dir, decision["decision_id"], reviewer=REVIEWER)

    assert json.loads(log_lines(enforcement_dir)[1])["same_reviewer_as_ban"] is True


# --------------------------------------------------------------------------
# Preservation
# --------------------------------------------------------------------------

def test_game_window_evidence_reproduces_after_data_is_regenerated():
    data_dir = _workspace / "reproduce" / "data"
    shutil.copytree(_main, data_dir)
    report, _ = write_case(data_dir, "user_id", INCIDENT, _workspace / "reproduce" / "cases")

    enforcement_dir = fresh_enforcement()
    decision = ban(enforcement_dir, report=report, data_dir=data_dir, user_id=INCIDENT, reason="CHEATING",
                   override_justification="Testing that the game-wide evidence survives regeneration.")

    # Regenerate data/ as a different dataset.
    shutil.rmtree(data_dir)
    shutil.copytree(_other, data_dir)

    original, reproduced = enforce.reproduce_report(enforcement_dir, Path(decision["bundle"]).name)
    context = next(item for item in reproduced["observations"] if item["kind"] == "termination_context")

    assert reproduced == original
    assert any(item["elevated"] for item in context["details"]), "the game-wide spike must be reproducible"


def test_bundle_identity_covers_every_stage_and_is_never_overwritten():
    data_dir = _workspace / "rescored" / "data"
    shutil.copytree(_main, data_dir)

    enforcement_dir = fresh_enforcement()
    first = ban(enforcement_dir)
    first_index = (Path(first["bundle"]) / "index.json").read_bytes()
    reverse(enforcement_dir, first["decision_id"])

    # Same sessions, different model scores.
    publish_model(data_dir, forest_seed=1)
    report, _ = write_case(data_dir, "user_id", SMURF, _workspace / "rescored" / "cases")
    second = ban(enforcement_dir, report=report, data_dir=data_dir, key="req_00000002")

    first_entry, _, second_entry = (json.loads(line) for line in log_lines(enforcement_dir))

    assert first_entry["dataset_id"] == second_entry["dataset_id"], "same sessions, same dataset"
    assert first["bundle"] != second["bundle"], "different scores must give a different bundle"
    assert (Path(first["bundle"]) / "index.json").read_bytes() == first_index
    assert len(list((enforcement_dir / "sessions").iterdir())) == 1, "sessions are stored once per dataset"


# --------------------------------------------------------------------------
# Crash recovery
# --------------------------------------------------------------------------

def test_crash_between_evidence_and_decision_leaves_no_ban():
    enforcement_dir = fresh_enforcement()
    original = enforce.append_entry

    def crash(*args, **kwargs):
        raise RuntimeError("simulated crash after evidence was preserved")

    enforce.append_entry = crash

    try:
        try:
            ban(enforcement_dir)
        except RuntimeError:
            pass
    finally:
        enforce.append_entry = original

    bundles = list((enforcement_dir / "bundles").iterdir())

    assert len(bundles) == 1, "evidence is preserved first"
    assert not log_lines(enforcement_dir), "no decision was recorded"

    retry = ban(enforcement_dir)

    assert retry["status"] == "applied"
    assert Path(retry["bundle"]) == bundles[0], "the retry reuses the preserved evidence"


def test_torn_trailing_record_is_removed():
    enforcement_dir = fresh_enforcement()
    ban(enforcement_dir)

    with open(enforcement_dir / enforce.LOG_NAME, "ab") as file:
        file.write(b'{"sequence": 1, "decision_id": "decision_torn')

    result = ban(enforcement_dir, report=SMURF_2_REPORT, user_id=SMURF_2, key="req_00000002")

    assert result["status"] == "applied"
    assert any("removed an incomplete trailing record" in note for note in result["recovery"])
    assert len(log_lines(enforcement_dir)) == 2
    assert enforce.verify(enforcement_dir)["evidence_problems"] == {}


def test_valid_record_missing_its_newline_is_completed():
    enforcement_dir = fresh_enforcement()
    ban(enforcement_dir)
    path = enforcement_dir / enforce.LOG_NAME
    path.write_bytes(path.read_bytes().rstrip(b"\n"))

    result = ban(enforcement_dir, report=SMURF_2_REPORT, user_id=SMURF_2, key="req_00000002")

    assert any("completed trailing record" in note for note in result["recovery"])
    assert len(log_lines(enforcement_dir)) == 2


def test_complete_but_invalid_record_is_an_integrity_failure():
    enforcement_dir = fresh_enforcement()
    ban(enforcement_dir)
    path = enforcement_dir / enforce.LOG_NAME
    path.write_bytes(path.read_bytes().replace(b"MULTI_ACCOUNTING", b"CHEATING", 1))

    refused(lambda: ban(enforcement_dir, report=SMURF_2_REPORT, user_id=SMURF_2, key="req_00000002"),
            "entry_sha256 does not match", enforce.IntegrityError)
    assert len(log_lines(enforcement_dir)) == 1


def test_log_ahead_of_its_head_is_reconciled():
    enforcement_dir = fresh_enforcement()
    ban(enforcement_dir)
    original = enforce.write_head

    def crash(*args, **kwargs):
        raise RuntimeError("simulated crash after the log was synced")

    enforce.write_head = crash

    try:
        try:
            ban(enforcement_dir, report=SMURF_2_REPORT, user_id=SMURF_2, key="req_00000002")
        except RuntimeError:
            pass
    finally:
        enforce.write_head = original

    assert json.loads((enforcement_dir / enforce.HEAD_NAME).read_text())["entries"] == 1
    assert len(log_lines(enforcement_dir)) == 2

    result = ban(enforcement_dir, report=SMURF_3_REPORT, user_id=SMURF_3, key="req_00000003")

    assert any("reconciled" in note for note in result["recovery"])
    assert json.loads((enforcement_dir / enforce.HEAD_NAME).read_text())["entries"] == 3


def test_log_shorter_than_its_head_is_an_integrity_failure():
    enforcement_dir = fresh_enforcement()
    ban(enforcement_dir)
    ban(enforcement_dir, report=SMURF_2_REPORT, user_id=SMURF_2, key="req_00000002")

    path = enforcement_dir / enforce.LOG_NAME
    path.write_bytes(path.read_bytes().splitlines(keepends=True)[0])

    refused(lambda: enforce.status("any", SMURF, enforcement_dir), "entries are missing", enforce.IntegrityError)


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