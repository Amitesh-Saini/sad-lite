# =============================================================================
# SAD-Lite — Enforcement Decisions
# =============================================================================
#
# PURPOSE
# Record a reviewer's decision to ban an account, or to reverse a ban, so that
# anyone handling an appeal later can see exactly what was decided, by whom,
# why, and on what evidence, and can re-check that evidence themselves.
#
# WHAT "BANNING" MEANS HERE
# SAD-Lite has no account system. The decision log and evidence bundles are
# the whole effect: ban status is derived from the log. Nothing else changes.
#
# WHY THIS IS NEEDED
# Every other component produces evidence. This is the only one that turns
# evidence into an action against a person, so it is where mistakes cost most
# and where an appeal will look first.
#
# IDENTITY
# An account is identified by (dataset_id, user_id). The dataset_id is the
# SHA-256 of the session data, because the same u.ser ID names different
# synthetic people in different datasets.
#
# WHAT A BAN REQUIRES
# 1. A case report for one account, produced by investigate.py.
# 2. The data directory it was produced from. A hash identifies a dataset but
#    does not locate it, so the directory is supplied and checked.
# 3. All four input stages verified: sessions, features, rules, and model.
#    This checks that the evidence is complete and consistent; it does not
#    require either detector to have flagged the account.
# 4. The report must be reproduced exactly: enforcement re-runs the
#    investigation on the verified data and compares the result with the
#    submitted report. A difference means the report was edited, belongs to
#    another dataset, or came from another investigator version or settings.
#    A match proves the reasoning is reproducible, not that it is correct.
# 5. The reason code must match a hypothesis in the report at Moderate
#    confidence or above. A reviewer may override this one requirement, and
#    only this one, with a written justification that is recorded.
# 6. Notes explaining the specific decision. The code is not the explanation.
#
# WHAT A REVERSAL REQUIRES
# The ID of the ban it reverses, a reversal code, the reviewer, and notes.
# It is judged against the ban's preserved evidence, never the current data,
# which may have been regenerated since. Damaged or missing evidence does not
# block a reversal; the failure is recorded with it. The same reviewer may
# reverse their own decision; the log records whether the typed names match,
# because two typed names do not prove two people.
#
# SAFETY PROPERTIES
# - Dry run by default. Without --apply, everything is validated and the
#   proposed decision is shown, but nothing is written.
# - Idempotent. Every applied request carries an explicit request key. The
#   same key with the same request returns the original decision; the same key
#   with a different request is refused; a new action needs a new key. The
#   request is identified by its meaning (action, target, report or decision,
#   reason, reviewer, notes, justification), never by --apply, paths, or times,
#   so a dry run and its applied twin are the same request.
# - Evidence before decision. The evidence bundle is written and made durable
#   (files and directories fsynced) before the decision is appended. A crash
#   in between leaves preserved evidence and no ban; a retry reuses it.
# - Serialised. An exclusive lock covers reading the log, the duplicate and
#   state checks, and committing the decision.
# - Append-only log, hash-chained. Each entry records the previous entry's
#   hash. A separate head record holds the entry count and last hash.
#   Recovery under the lock: a trailing record with no final newline is
#   completed if it is valid and removed if it is not; a log ahead of its head
#   (a crash after the log was synced, before the head was updated) is
#   reconciled; a log shorter than its head, a complete but invalid record,
#   or a broken chain is an integrity failure and nothing is written.
#
# TRUST MODEL
# Like provenance.py, this detects accidents: mixed datasets, edited reports,
# interrupted writes, accidental truncation. It does not detect deliberate
# rewriting by someone who can write to enforcement/ and data/; the log and
# its head can be rewritten together. The typed reviewer name is attribution
# for this local project, not authenticated identity.
#
# STORAGE (enforcement/, separate from generated data)
# - decisions.jsonl: the append-only decision log, the source of truth.
# - head.json: entry count and last entry hash.
# - sessions/<dataset_id>/sessions.csv: each dataset's sessions, stored once.
# - bundles/<bundle_id>/: one complete evidence bundle per distinct set of
#   evidence: the report, every stage's files and manifests, configuration and
#   versions. Its ID covers all of it, so the same sessions scored with a
#   different configuration or seed get a different bundle. Bundles are never
#   overwritten.
#
# Locking uses fcntl, so this runs on macOS and Linux, not Windows.
#
# Run:
#     python enforce.py ban --report reports/cases/<case>.json --user-id user_00042 \
#         --reason MULTI_ACCOUNTING --reviewer "A. Saini" --notes "..."            # dry run
#     python enforce.py ban ... --request-key <key> --apply
#     python enforce.py reverse --decision-id decision_... --reason INSUFFICIENT_EVIDENCE \
#         --reviewer "..." --notes "..." --request-key <key> --apply
#     python enforce.py status --dataset-id <sha256> --user-id user_00042
#     python enforce.py verify
# =============================================================================

import argparse
import contextlib
import fcntl
import getpass
import json
import os
import re
import secrets
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import investigate
import provenance


ROOT = Path(__file__).resolve().parent

# Ban codes and the hypothesis each one needs in the report.
BAN_CODES = {
    "ACCOUNT_SHARING": "Account sharing or resale",
    "MULTI_ACCOUNTING": "Coordinated multi-accounting",
    "AUTOMATION_ABUSE": "Automation",
    "CHEATING": "Cheating",
}

# Reversal codes say why the original decision no longer stands.
REVERSAL_CODES = {
    "INCORRECT_ATTRIBUTION": "The behaviour belonged to someone other than the account holder.",
    "INSUFFICIENT_EVIDENCE": "The evidence did not support the decision.",
    "INNOCENT_EXPLANATION_CONFIRMED": "An innocent explanation was confirmed.",
    "PROCESS_ERROR": "The decision was made in error (wrong account, code, or procedure).",
}

SUPPORTING_CONFIDENCE = {"Moderate", "Strong"}

# Higher is stronger; used to order supported reasons (pipeline_demo.py's preview).
CONFIDENCE_RANK = {"Strong": 2, "Moderate": 1}

REQUIRED_STAGES = ["generate", "features", "rules", "model"]

MIN_NOTES_LENGTH = 20
MIN_JUSTIFICATION_LENGTH = 20

REQUEST_KEY = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

LOG_NAME = "decisions.jsonl"
HEAD_NAME = "head.json"
LOCK_NAME = ".lock"


class EnforcementError(Exception):
    """The request was refused; nothing was written."""


class IntegrityError(EnforcementError):
    """The decision log or its head failed an integrity check; nothing was written."""


# ------------------------------------------------------------
# Serialisation and durable writes.
# ------------------------------------------------------------

def canonical(value):
    """One fixed byte encoding for hashing and comparison."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("utf-8")


def digest(value):
    return provenance.sha256_bytes(canonical(value))


def as_json(value):
    """Round-trip through JSON so in-memory and loaded reports compare like for like."""

    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True, default=str))


def fsync_directory(path):
    """Make a directory's entries (new files, renames) durable."""

    descriptor = os.open(path, os.O_RDONLY)

    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_file_durably(path, data):
    """Write a file and flush it to disk."""

    with open(path, "wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())


def replace_durably(path, data):
    """Atomically replace a small file: write a temporary copy, sync, rename, sync the directory."""

    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    write_file_durably(temporary, data)
    os.replace(temporary, path)
    fsync_directory(path.parent)


def publish_directory_durably(files, final_dir):
    """Write {relative_path: bytes} into final_dir as one unit; never overwrites an existing directory."""

    final_dir = Path(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_dir.parent / f".{final_dir.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
    temporary.mkdir()

    try:
        directories = {temporary}

        for relative, data in sorted(files.items()):
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            directories.add(path.parent)
            write_file_durably(path, data)

        # Deepest directories first, so every entry is durable before its parent.
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            fsync_directory(directory)

        os.rename(temporary, final_dir)
        fsync_directory(final_dir.parent)

    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


# ------------------------------------------------------------
# The decision log.
# ------------------------------------------------------------

@contextlib.contextmanager
def exclusive_lock(enforcement_dir):
    """Serialise every command that reads state in order to change it."""

    enforcement_dir.mkdir(parents=True, exist_ok=True)

    with open(enforcement_dir / LOCK_NAME, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def entry_hash(entry):
    """Hash an entry's content, excluding its own hash field."""

    return digest({key: value for key, value in entry.items() if key != "entry_sha256"})


def check_entry(entry, index, previous_hash):
    """Validate one parsed entry against its position in the chain."""

    if not isinstance(entry, dict):
        return "record is not an object"

    if entry.get("sequence") != index:
        return f"sequence {entry.get('sequence')} where {index} was expected"

    if entry.get("previous_entry_sha256") != previous_hash:
        return "previous_entry_sha256 does not match the preceding entry"

    if entry.get("entry_sha256") != entry_hash(entry):
        return "entry_sha256 does not match its content"

    return None


def read_log(enforcement_dir, repair):
    """Read and verify the log. With repair=True (under the lock), fix a torn trailing record.

    Returns (entries, recovery_notes). Raises IntegrityError for anything
    other than an unmistakably incomplete trailing record.
    """

    path = enforcement_dir / LOG_NAME
    notes = []

    if not path.exists():
        return [], notes

    data = provenance.read_file_once(path)

    complete = data
    fragment = b""

    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        complete, fragment = data[:cut], data[cut:]

    entries = []
    previous_hash = None

    for index, line in enumerate(complete.splitlines()):
        try:
            entry = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise IntegrityError(f"Decision log line {index + 1} is a complete but unreadable record") from error

        problem = check_entry(entry, index, previous_hash)

        if problem:
            raise IntegrityError(f"Decision log line {index + 1} is invalid: {problem}")

        entries.append(entry)
        previous_hash = entry["entry_sha256"]

    if fragment:
        # Only a record without its final newline is treated as torn.
        try:
            entry = json.loads(fragment.decode("utf-8"))
            valid = check_entry(entry, len(entries), previous_hash) is None
        except (UnicodeDecodeError, json.JSONDecodeError):
            valid = False

        if valid:
            entries.append(entry)
            notes.append(f"completed trailing record {entry['decision_id']} that was missing its final newline")

            if repair:
                with open(path, "ab") as file:
                    file.write(b"\n")
                    file.flush()
                    os.fsync(file.fileno())
        else:
            notes.append(f"removed an incomplete trailing record ({len(fragment)} bytes)")

            if repair:
                with open(path, "r+b") as file:
                    file.truncate(len(complete))
                    file.flush()
                    os.fsync(file.fileno())

    return entries, notes


def read_head(enforcement_dir):
    path = enforcement_dir / HEAD_NAME

    if not path.exists():
        return None

    try:
        head = json.loads(provenance.read_file_once(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError("head.json is unreadable") from error

    if not isinstance(head, dict) or not isinstance(head.get("entries"), int):
        raise IntegrityError("head.json does not have the expected fields")

    return head


def write_head(enforcement_dir, entries):
    head = {"entries": len(entries), "last_entry_sha256": entries[-1]["entry_sha256"] if entries else None}
    replace_durably(enforcement_dir / HEAD_NAME, json.dumps(head, indent=2, sort_keys=True).encode("utf-8"))


def reconcile_head(enforcement_dir, entries, repair):
    """Check the head against the log; reconcile a log that is ahead of it."""

    head = read_head(enforcement_dir)
    notes = []

    # A missing head is treated as an empty one, so a crash before the first
    # head write is recoverable. (Deleting the head therefore hides truncation;
    # see TRUST MODEL.)
    recorded = head["entries"] if head else 0
    recorded_hash = head.get("last_entry_sha256") if head else None

    if recorded > len(entries):
        raise IntegrityError(f"Decision log has {len(entries)} entries but its head records {recorded}: entries are missing")

    if recorded and entries[recorded - 1]["entry_sha256"] != recorded_hash:
        raise IntegrityError("The head does not match the decision log at the position it records")

    if len(entries) > recorded:
        notes.append(f"head was {len(entries) - recorded} entries behind the log (a crash after the log was synced); reconciled")

        if repair:
            write_head(enforcement_dir, entries)

    return notes


def load_state(enforcement_dir, repair):
    """Read and verify the log and head. repair=True only under the lock."""

    entries, notes = read_log(enforcement_dir, repair)
    notes += reconcile_head(enforcement_dir, entries, repair)

    return entries, notes


def append_entry(enforcement_dir, entries, entry):
    """Append one decision, chained to the previous one, then update the head."""

    entry = dict(entry)
    entry["sequence"] = len(entries)
    entry["previous_entry_sha256"] = entries[-1]["entry_sha256"] if entries else None
    entry["entry_sha256"] = entry_hash(entry)

    path = enforcement_dir / LOG_NAME
    created = not path.exists()

    with open(path, "ab") as file:
        file.write(canonical(entry) + b"\n")
        file.flush()
        os.fsync(file.fileno())

    if created:
        fsync_directory(enforcement_dir)

    entries = entries + [entry]
    write_head(enforcement_dir, entries)

    return entry


def by_request_key(entries, key):
    return next((entry for entry in entries if entry.get("request_key") == key), None)


def active_ban(entries, dataset_id, user_id):
    """The account's current ban, if any, derived from the log's history."""

    reversed_ids = {entry["reverses"] for entry in entries if entry["entry_type"] == "reversal"}

    return next(
        (entry for entry in reversed(entries)
         if entry["entry_type"] == "ban" and entry["dataset_id"] == dataset_id
         and entry["user_id"] == user_id and entry["decision_id"] not in reversed_ids),
        None,
    )


# ------------------------------------------------------------
# Evidence bundles.
# ------------------------------------------------------------

def manifest_bytes(manifest):
    """The exact manifest bytes provenance.publish_stage writes."""

    return json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")


def load_verified_pipeline(data_dir):
    """Read all four stages once, verifying hashes and the chain between them."""

    stages = {}

    for stage in REQUIRED_STAGES:
        try:
            manifest, contents = provenance.load_stage(data_dir, stage, investigate.INPUT_STAGES[stage])
        except provenance.ProvenanceError as error:
            raise EnforcementError(f"Stage '{stage}' in {data_dir} is not usable: {error}") from error

        stages[stage] = (manifest, contents)

    sessions_digest = provenance.sha256_bytes(stages["generate"][1]["sessions.csv"])
    features_digest = provenance.sha256_bytes(stages["features"][1]["features.csv"])

    try:
        provenance.check_input(stages["features"][0], "sessions", "generate", "sessions.csv", sessions_digest)
        provenance.check_input(stages["rules"][0], "features", "features", "features.csv", features_digest)
        provenance.check_input(stages["model"][0], "features", "features", "features.csv", features_digest)
    except provenance.ProvenanceError as error:
        raise EnforcementError(f"Stages in {data_dir} come from different runs: {error}") from error

    return stages, sessions_digest


def materialise(stages, target_dir):
    """Write verified stage bytes into a private directory the investigator can read."""

    for stage, (manifest, contents) in stages.items():
        directory = Path(target_dir) / stage
        directory.mkdir(parents=True)

        for name, data in contents.items():
            (directory / name).write_bytes(data)

        (directory / provenance.MANIFEST_NAME).write_bytes(manifest_bytes(manifest))


def regenerate_report(stages, subject_type, subject_value):
    """Re-run the investigation on exactly these verified bytes."""

    with tempfile.TemporaryDirectory(prefix="sadlite-enforce-") as folder:
        materialise(stages, folder)
        inputs = investigate.load_inputs(folder)
        report = investigate.investigate(inputs, subject_type, subject_value)

    return as_json(report)


def bundle_files(stages, report):
    """Every file an evidence bundle preserves, except the shared session data."""

    files = {
        "report.json": json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8"),
        "report.md": investigate.render_markdown(report).encode("utf-8"),
    }

    for stage, (manifest, contents) in stages.items():
        files[f"stages/{stage}/{provenance.MANIFEST_NAME}"] = manifest_bytes(manifest)

        if stage == "generate":
            continue

        for name, data in contents.items():
            files[f"stages/{stage}/{name}"] = data

    return files


def bundle_index(files, sessions_digest, report):
    """What identifies a bundle: every file's hash, the sessions, and the investigator."""

    return {
        "schema_version": 1,
        "sessions_sha256": sessions_digest,
        "files": {name: provenance.sha256_bytes(data) for name, data in sorted(files.items())},
        "investigator": report["investigator"],
        "settings_sha256": digest(report["settings"]),
    }


def preserve_evidence(enforcement_dir, stages, sessions_digest, report):
    """Store the sessions once per dataset and the bundle once per evidence set; never overwrite."""

    sessions_bytes = stages["generate"][1]["sessions.csv"]
    sessions_dir = enforcement_dir / "sessions" / sessions_digest

    if sessions_dir.exists():
        stored = provenance.read_file_once(sessions_dir / "sessions.csv")

        if provenance.sha256_bytes(stored) != sessions_digest:
            raise IntegrityError(f"Archived sessions for dataset {sessions_digest[:12]}… do not match their hash")
    else:
        publish_directory_durably({"sessions.csv": sessions_bytes}, sessions_dir)

    files = bundle_files(stages, report)
    index = bundle_index(files, sessions_digest, report)
    index_bytes = json.dumps(index, indent=2, sort_keys=True).encode("utf-8")
    bundle_id = "bundle_" + provenance.sha256_bytes(canonical(index))[:24]
    bundle_dir = enforcement_dir / "bundles" / bundle_id

    if bundle_dir.exists():
        # The same evidence was preserved before (a retry, or another decision
        # on identical evidence). Confirm it is intact; never rewrite it.
        problems = verify_bundle(enforcement_dir, bundle_id, provenance.sha256_bytes(index_bytes))

        if problems:
            raise IntegrityError(f"Existing bundle {bundle_id} is damaged: {problems[0]}")
    else:
        publish_directory_durably({**files, "index.json": index_bytes}, bundle_dir)

    return bundle_id, provenance.sha256_bytes(index_bytes), bundle_dir


def verify_bundle(enforcement_dir, bundle_id, expected_index_sha256):
    """Check a bundle and its archived sessions against the hashes recorded in the log."""

    bundle_dir = enforcement_dir / "bundles" / bundle_id
    problems = []

    try:
        index_bytes = provenance.read_file_once(bundle_dir / "index.json")
    except provenance.ProvenanceError as error:
        return [f"index.json unavailable: {error}"]

    if provenance.sha256_bytes(index_bytes) != expected_index_sha256:
        problems.append("index.json does not match the hash recorded in the decision log")

    # Damaged evidence must be reported, never raised: a reversal depends on
    # this function returning, whatever state the bundle is in.
    try:
        index = json.loads(index_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return problems + [f"index.json is unreadable: {error.__class__.__name__}"]

    files = index.get("files") if isinstance(index, dict) else None
    sessions_sha256 = index.get("sessions_sha256") if isinstance(index, dict) else None

    if (not isinstance(files, dict) or not isinstance(sessions_sha256, str)
            or not all(isinstance(name, str) and isinstance(value, str) for name, value in files.items())):
        return problems + ["index.json does not have the expected structure"]

    for name, expected in sorted(files.items()):
        path = bundle_dir / name

        # A damaged index could name a path outside the bundle; never follow it.
        if bundle_dir.resolve() not in path.resolve().parents:
            problems.append(f"{name}: path outside the bundle")
            continue

        try:
            data = provenance.read_file_once(path)
        except provenance.ProvenanceError as error:
            problems.append(f"{name}: {error}")
            continue

        if provenance.sha256_bytes(data) != expected:
            problems.append(f"{name} does not match its recorded hash")

    if not re.fullmatch(r"[0-9a-f]{64}", sessions_sha256):
        return problems + ["index.json names an invalid sessions hash"]

    sessions_path = enforcement_dir / "sessions" / sessions_sha256 / "sessions.csv"

    try:
        if provenance.sha256_bytes(provenance.read_file_once(sessions_path)) != sessions_sha256:
            problems.append("archived sessions do not match their hash")
    except provenance.ProvenanceError as error:
        problems.append(f"archived sessions unavailable: {error}")

    return problems


def reproduce_report(enforcement_dir, bundle_id):
    """Re-run the investigation from a bundle alone, independent of the current data/."""

    bundle_dir = enforcement_dir / "bundles" / bundle_id
    index = json.loads(provenance.read_file_once(bundle_dir / "index.json").decode("utf-8"))
    original = json.loads(provenance.read_file_once(bundle_dir / "report.json").decode("utf-8"))

    with tempfile.TemporaryDirectory(prefix="sadlite-reproduce-") as folder:
        for stage in REQUIRED_STAGES:
            (Path(folder) / stage).mkdir()

        sessions = provenance.read_file_once(enforcement_dir / "sessions" / index["sessions_sha256"] / "sessions.csv")
        (Path(folder) / "generate" / "sessions.csv").write_bytes(sessions)

        for name in index["files"]:
            if name.startswith("stages/"):
                _, stage, file_name = name.split("/")
                (Path(folder) / stage / file_name).write_bytes(provenance.read_file_once(bundle_dir / name))

        reproduced = as_json(investigate.investigate(investigate.load_inputs(folder), original["subject"]["type"], original["subject"]["value"]))

    return original, reproduced


# ------------------------------------------------------------
# Requests.
# ------------------------------------------------------------

def reviewer_record(typed):
    """Attribution only: the typed name and the local OS user, neither authenticated."""

    try:
        os_user = getpass.getuser()
    except Exception:  # noqa: BLE001
        os_user = None

    return {"typed": investigate.clean_text(typed).strip(), "os_user": os_user, "authenticated": False}


def check_text(label, value, minimum, code=None):
    value = investigate.clean_text(value or "").strip()

    if len(value) < minimum:
        raise EnforcementError(f"{label} must explain the specific decision (at least {minimum} characters)")

    if code is not None and value.replace("_", " ").lower() in {code.lower(), code.replace("_", " ").lower()}:
        raise EnforcementError(f"{label} must explain the decision, not repeat the code {code}")

    return value


def check_request_key(key, apply):
    if key is None:
        if apply:
            raise EnforcementError("--apply needs --request-key; run without --apply to get a suggested key")
        return None

    if not REQUEST_KEY.match(key):
        raise EnforcementError("--request-key must be 8–64 letters, digits, '-' or '_'")

    return key


def decision_id_for(request_key, request_sha256):
    return "decision_" + provenance.sha256_bytes(f"{request_key}:{request_sha256}".encode("utf-8"))[:20]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_only_state(enforcement_dir):
    """State for a dry run or status: verified, never repaired, nothing created."""

    if not enforcement_dir.exists():
        return [], []

    return load_state(enforcement_dir, repair=False)


def duplicate_or_conflict(entries, request_key, request_sha256):
    """Return the original decision for an exact retry; refuse a reused key."""

    if request_key is None:
        return None

    existing = by_request_key(entries, request_key)

    if existing is None:
        return None

    if existing["request_sha256"] != request_sha256:
        raise EnforcementError(
            f"Request key {request_key} was already used for a different request ({existing['decision_id']}); "
            f"use a new key for a new action"
        )

    return existing


# ------------------------------------------------------------
# Ban.
# ------------------------------------------------------------

def supported_reasons(report):
    """Ban codes the report supports at Moderate or above, strongest first.

    Uses the structured hypotheses, the same code-to-hypothesis mapping, and
    the same threshold as ban(). Ties at equal confidence are broken by the
    code's alphabetical order, so the selection is deterministic. Supporting a
    reason is not a decision: ban() still verifies the report and a person
    still decides.
    """

    by_name = {}

    for item in report.get("hypotheses", []):
        by_name.setdefault(item.get("name"), item.get("confidence"))  # first match, as ban() uses
    supported = [(code, by_name[name]) for code, name in BAN_CODES.items()
                 if by_name.get(name) in SUPPORTING_CONFIDENCE]

    return sorted(supported, key=lambda pair: (-CONFIDENCE_RANK[pair[1]], pair[0]))


def ban(report_path, data_dir, user_id, reason, reviewer, notes, request_key=None, apply=False,
        override_justification=None, enforcement_dir=ROOT / "enforcement"):
    """Validate a ban and, with apply=True, preserve its evidence and record it."""

    enforcement_dir = Path(enforcement_dir)

    # --- The request itself -------------------------------------------------------
    if reason not in BAN_CODES:
        raise EnforcementError(f"Unknown ban code {reason!r}; choose from {sorted(BAN_CODES)}")

    notes = check_text("Notes", notes, MIN_NOTES_LENGTH, reason)
    reviewer = reviewer_record(reviewer)

    if not reviewer["typed"]:
        raise EnforcementError("A reviewer name is required")

    if override_justification is not None:
        override_justification = check_text("Override justification", override_justification, MIN_JUSTIFICATION_LENGTH)

    request_key = check_request_key(request_key, apply)

    # --- The report ---------------------------------------------------------------
    try:
        document = json.loads(provenance.read_file_once(report_path).decode("utf-8"))
        submitted = document["report"]
    except (provenance.ProvenanceError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise EnforcementError(f"Cannot read a case report from {report_path}: {error}") from error

    if submitted.get("status") != "case" or "account" not in submitted:
        raise EnforcementError("Only a completed account case report can support a ban (not a diagnostic or an entity report)")

    if submitted["account"] != user_id:
        raise EnforcementError(f"The report covers {submitted['account']}, not {user_id}")

    # --- The evidence -------------------------------------------------------------
    stages, dataset_id = load_verified_pipeline(data_dir)

    report_dataset = next((record.get("files", {}).get("sessions.csv") for record in submitted.get("provenance", [])
                           if record.get("stage") == "generate"), None)

    if report_dataset != dataset_id:
        raise EnforcementError(
            f"The report is for dataset {str(report_dataset)[:12]}…, but {data_dir} holds dataset {dataset_id[:12]}…; "
            f"supply the data directory the report was produced from"
        )

    current = investigate.code_record()

    if submitted.get("investigator") != current or submitted.get("settings") != as_json(investigate.SETTINGS):
        raise EnforcementError("The report was produced by a different investigator version or settings; re-investigate")

    regenerated = regenerate_report(stages, submitted["subject"]["type"], submitted["subject"]["value"])

    if canonical(regenerated) != canonical(submitted):
        raise EnforcementError("The report does not match a fresh investigation of this data; it may have been edited")

    unverified = [record["stage"] for record in regenerated["provenance"] if record["status"] != "verified"]

    if unverified:
        raise EnforcementError(f"A ban needs every input stage verified; not verified: {unverified}")

    # --- Evidence policy: the reason must match the report --------------------------
    required = BAN_CODES[reason]
    hypothesis = next((item for item in regenerated["hypotheses"] if item["name"] == required), None)
    supported = hypothesis is not None and hypothesis["confidence"] in SUPPORTING_CONFIDENCE

    if not supported and override_justification is None:
        found = f"{hypothesis['confidence']}" if hypothesis else "absent"
        raise EnforcementError(
            f"{reason} needs the hypothesis '{required}' at Moderate or above; the report has it {found}. "
            f"To decide against the evidence policy, supply --override-justification (recorded with the decision)"
        )

    override = None if supported else {"justification": override_justification, "policy": f"{reason} requires '{required}' at Moderate or above"}

    # --- The request's identity: its meaning, not how it was run ----------------------
    report_sha256 = digest(submitted)
    request = {
        "action": "ban", "dataset_id": dataset_id, "user_id": user_id, "report_sha256": report_sha256,
        "reason": reason, "reviewer": reviewer["typed"], "notes": notes,
        "override_justification": override["justification"] if override else None,
    }
    request_sha256 = digest(request)

    proposal = {
        "action": "ban", "dataset_id": dataset_id, "user_id": user_id, "reason": reason,
        "hypothesis": {"name": required, "confidence": hypothesis["confidence"] if hypothesis else None},
        "override": override, "notes": notes, "reviewer": reviewer,
        "case_id": submitted["case_id"], "request_sha256": request_sha256,
    }

    if not apply:
        entries, recovery = read_only_state(enforcement_dir)
        existing = duplicate_or_conflict(entries, request_key, request_sha256)
        current_ban = active_ban(entries, dataset_id, user_id)

        return {
            "status": "already_applied" if existing else ("would_refuse_already_banned" if current_ban else "dry_run"),
            "proposal": proposal,
            "existing_decision": existing["decision_id"] if existing else (current_ban["decision_id"] if current_ban else None),
            "suggested_request_key": request_key or "req_" + secrets.token_hex(8),
            "recovery_needed": recovery,
        }

    # --- Commit, under the lock -----------------------------------------------------
    with exclusive_lock(enforcement_dir):
        entries, recovery = load_state(enforcement_dir, repair=True)

        # A completed matching request is checked before current state.
        existing = duplicate_or_conflict(entries, request_key, request_sha256)

        if existing:
            return {"status": "already_applied", "decision_id": existing["decision_id"],
                    "bundle": str(enforcement_dir / "bundles" / existing["evidence"]["bundle_id"]), "recovery": recovery}

        current_ban = active_ban(entries, dataset_id, user_id)

        if current_ban:
            raise EnforcementError(f"{user_id} in dataset {dataset_id[:12]}… is already banned by {current_ban['decision_id']}")

        # Evidence first. A crash after this leaves evidence and no ban.
        bundle_id, index_sha256, bundle_dir = preserve_evidence(enforcement_dir, stages, dataset_id, regenerated)

        entry = append_entry(enforcement_dir, entries, {
            "entry_type": "ban",
            "decision_id": decision_id_for(request_key, request_sha256),
            "request_key": request_key,
            "request_sha256": request_sha256,
            "dataset_id": dataset_id,
            "user_id": user_id,
            "reason": reason,
            "hypothesis": proposal["hypothesis"],
            "override": override,
            "notes": notes,
            "reviewer": reviewer,
            "evidence": {"bundle_id": bundle_id, "bundle_index_sha256": index_sha256, "report_sha256": report_sha256,
                         "case_id": submitted["case_id"], "sessions_sha256": dataset_id},
            "log_recovery": recovery,
            "recorded_at": now(),
        })

    return {"status": "applied", "decision_id": entry["decision_id"], "bundle": str(bundle_dir), "recovery": recovery}


# ------------------------------------------------------------
# Reversal.
# ------------------------------------------------------------

def reverse(decision_id, reason, reviewer, notes, request_key=None, apply=False, enforcement_dir=ROOT / "enforcement"):
    """Reverse a ban using its preserved evidence, never the current data."""

    enforcement_dir = Path(enforcement_dir)

    if reason not in REVERSAL_CODES:
        raise EnforcementError(f"Unknown reversal code {reason!r}; choose from {sorted(REVERSAL_CODES)}")

    notes = check_text("Notes", notes, MIN_NOTES_LENGTH, reason)
    reviewer = reviewer_record(reviewer)

    if not reviewer["typed"]:
        raise EnforcementError("A reviewer name is required")

    request_key = check_request_key(request_key, apply)

    request = {"action": "reverse", "decision_id": decision_id, "reason": reason, "reviewer": reviewer["typed"], "notes": notes}
    request_sha256 = digest(request)

    def plan(entries):
        original = next((entry for entry in entries if entry["entry_type"] == "ban" and entry["decision_id"] == decision_id), None)

        if original is None:
            raise EnforcementError(f"No ban {decision_id} in the decision log")

        reversal = next((entry for entry in entries if entry["entry_type"] == "reversal" and entry["reverses"] == decision_id), None)

        if reversal is not None:
            raise EnforcementError(f"{decision_id} was already reversed by {reversal['decision_id']}")

        # Damaged evidence is recorded, not a reason to refuse.
        problems = verify_bundle(enforcement_dir, original["evidence"]["bundle_id"], original["evidence"]["bundle_index_sha256"])

        return original, {"verified": not problems, "problems": problems}

    if not apply:
        entries, recovery = read_only_state(enforcement_dir)
        existing = duplicate_or_conflict(entries, request_key, request_sha256)

        if existing:
            return {"status": "already_applied", "existing_decision": existing["decision_id"], "recovery_needed": recovery}

        original, integrity = plan(entries)

        return {
            "status": "dry_run",
            "proposal": {"action": "reverse", "reverses": decision_id, "user_id": original["user_id"],
                         "dataset_id": original["dataset_id"], "reason": reason, "notes": notes, "reviewer": reviewer,
                         "evidence_integrity": integrity,
                         "same_reviewer_as_ban": reviewer["typed"].lower() == original["reviewer"]["typed"].lower()},
            "suggested_request_key": request_key or "req_" + secrets.token_hex(8),
            "recovery_needed": recovery,
        }

    with exclusive_lock(enforcement_dir):
        entries, recovery = load_state(enforcement_dir, repair=True)
        existing = duplicate_or_conflict(entries, request_key, request_sha256)

        if existing:
            return {"status": "already_applied", "decision_id": existing["decision_id"], "recovery": recovery}

        original, integrity = plan(entries)

        entry = append_entry(enforcement_dir, entries, {
            "entry_type": "reversal",
            "decision_id": decision_id_for(request_key, request_sha256),
            "request_key": request_key,
            "request_sha256": request_sha256,
            "reverses": decision_id,
            "dataset_id": original["dataset_id"],
            "user_id": original["user_id"],
            "reason": reason,
            "notes": notes,
            "reviewer": reviewer,
            # Typed names only: matching names do not prove one person, nor different names two.
            "same_reviewer_as_ban": reviewer["typed"].lower() == original["reviewer"]["typed"].lower(),
            "evidence_integrity": integrity,
            "log_recovery": recovery,
            "recorded_at": now(),
        })

    return {"status": "applied", "decision_id": entry["decision_id"], "reverses": decision_id,
            "evidence_integrity": integrity, "recovery": recovery}


# ------------------------------------------------------------
# Status and verification.
# ------------------------------------------------------------

def status(dataset_id, user_id, enforcement_dir=ROOT / "enforcement"):
    """An account's decision history and current state, with evidence re-checked."""

    enforcement_dir = Path(enforcement_dir)
    entries, recovery = read_only_state(enforcement_dir)

    history = [
        {"decision_id": entry["decision_id"], "type": entry["entry_type"], "reason": entry["reason"],
         "reviewer": entry["reviewer"]["typed"], "recorded_at": entry["recorded_at"], "reverses": entry.get("reverses")}
        for entry in entries if entry["dataset_id"] == dataset_id and entry["user_id"] == user_id
    ]

    current = active_ban(entries, dataset_id, user_id)
    result = {"dataset_id": dataset_id, "user_id": user_id, "history": history, "banned": current is not None,
              "recovery_needed": recovery}

    if current is not None:
        problems = verify_bundle(enforcement_dir, current["evidence"]["bundle_id"], current["evidence"]["bundle_index_sha256"])
        result["active_ban"] = current["decision_id"]
        result["evidence"] = "verified" if not problems else "INTEGRITY FAILURE"
        result["evidence_problems"] = problems

    return result


def verify(enforcement_dir=ROOT / "enforcement"):
    """Check the whole log, its head, and every referenced bundle."""

    enforcement_dir = Path(enforcement_dir)
    entries, recovery = read_only_state(enforcement_dir)
    problems = {}

    for entry in entries:
        if entry["entry_type"] == "ban":
            found = verify_bundle(enforcement_dir, entry["evidence"]["bundle_id"], entry["evidence"]["bundle_index_sha256"])

            if found:
                problems[entry["decision_id"]] = found

    return {"entries": len(entries), "recovery_needed": recovery, "evidence_problems": problems}


# ------------------------------------------------------------
# Command line.
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Record or reverse enforcement decisions. Dry run unless --apply.")
    parser.add_argument("--enforcement-dir", type=Path, default=ROOT / "enforcement")
    commands = parser.add_subparsers(dest="command", required=True)

    ban_parser = commands.add_parser("ban", help="ban an account on the evidence of a case report")
    ban_parser.add_argument("--report", type=Path, required=True, help="case report JSON written by investigate.py")
    ban_parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="the data directory the report was produced from")
    ban_parser.add_argument("--user-id", required=True)
    ban_parser.add_argument("--reason", required=True, choices=sorted(BAN_CODES))
    ban_parser.add_argument("--reviewer", required=True, help="attribution only; not authenticated")
    ban_parser.add_argument("--notes", required=True)
    ban_parser.add_argument("--override-justification", help="decide against the reason-to-evidence policy, with this recorded reason")
    ban_parser.add_argument("--request-key")
    ban_parser.add_argument("--apply", action="store_true")

    reverse_parser = commands.add_parser("reverse", help="reverse a ban")
    reverse_parser.add_argument("--decision-id", required=True)
    reverse_parser.add_argument("--reason", required=True, choices=sorted(REVERSAL_CODES))
    reverse_parser.add_argument("--reviewer", required=True, help="attribution only; not authenticated")
    reverse_parser.add_argument("--notes", required=True)
    reverse_parser.add_argument("--request-key")
    reverse_parser.add_argument("--apply", action="store_true")

    status_parser = commands.add_parser("status", help="an account's decision history")
    status_parser.add_argument("--dataset-id", required=True)
    status_parser.add_argument("--user-id", required=True)

    commands.add_parser("verify", help="check the log, its head, and every evidence bundle")

    args = parser.parse_args()

    try:
        if args.command == "ban":
            result = ban(args.report, args.data_dir, args.user_id, args.reason, args.reviewer, args.notes,
                         args.request_key, args.apply, args.override_justification, args.enforcement_dir)
        elif args.command == "reverse":
            result = reverse(args.decision_id, args.reason, args.reviewer, args.notes,
                             args.request_key, args.apply, args.enforcement_dir)
        elif args.command == "status":
            result = status(args.dataset_id, args.user_id, args.enforcement_dir)
        else:
            result = verify(args.enforcement_dir)
    except IntegrityError as error:
        print(f"INTEGRITY FAILURE, nothing written: {error}")
        sys.exit(3)
    except EnforcementError as error:
        print(f"Refused, nothing written: {error}")
        sys.exit(1)

    print(json.dumps(result, indent=2, sort_keys=True, default=str))

    if result.get("status") == "dry_run":
        print(f"\nDry run: nothing was written. To apply, repeat with --request-key {result['suggested_request_key']} --apply")

    if args.command == "verify" and result["evidence_problems"]:
        sys.exit(3)


if __name__ == "__main__":
    main()