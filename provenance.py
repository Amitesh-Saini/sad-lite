# =============================================================================
# SAD-Lite — Artifact Provenance
# =============================================================================
#
# PURPOSE
# Make sure every stage of the pipeline reads the outputs of the same run,
# and that the investigator never presents evidence from mixed runs.
#
# WHY THIS IS NEEDED
# User IDs are reused across generator seeds: user_00042 at seed 42 and at
# seed 7 are different accounts. Without provenance, scores from an old run
# could be shown next to sessions from a new one, attaching one account's
# score to another account's behaviour, and nothing would notice.
#
# TRUST MODEL (read this before relying on it)
# Guaranteed to detect:
# - Mixed-run artifacts: a stage's recorded input hash does not match the
#   upstream file actually present.
# - Incomplete stages: a stage directory without its manifest.
# - Inputs changed while a stage runs: every stage hashes and parses the same
#   bytes, read once, so it can never record one file and use another.
# - Files outside the stage: path traversal and symlinks are rejected.
# NOT guaranteed:
# - Deliberate modification by anyone who can write to data/. A writer who
#   changes a file and rewrites its manifest to match passes verification.
#   Detecting that needs a signature checked against a key held outside the
#   writer's control, and even a signature only shows who endorsed the
#   artifacts, not that their contents were computed correctly.
#
# HOW A STAGE IS PUBLISHED
# 1. Outputs are written into a temporary directory next to the stage.
# 2. manifest.json is written last; it is the commit marker.
# 3. The temporary directory is renamed into place.
# A reader accepts a stage only if its manifest exists and every listed file
# matches its recorded hash. During a re-publish the stage is briefly absent,
# which a reader reports as missing, never as mixed.
# Published stages in data/ are overwritten by reruns; immutable per-run
# directories are planned with run_all.sh.
#
# MANIFEST CONTENTS
# Stage name, schema version, each output file's SHA-256 and size, each
# input's stage, file, and SHA-256, settings, software versions, and the
# SHA-256 of the producing script's source code.
# =============================================================================

import hashlib
import io
import json
import os
import platform
import re
import secrets
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1

# Refuse to read anything larger; the pipeline's files are a few megabytes.
MAX_INPUT_BYTES = 200 * 1024 * 1024

# Stage and file names: plain names only, no separators or parent references.
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ProvenanceError(Exception):
    """A stage is missing, incomplete, unsafe, or does not match its manifest."""


def sha256_bytes(data):
    """Return the hex SHA-256 of some bytes."""

    return hashlib.sha256(data).hexdigest()


def check_name(name, kind):
    """Reject names that could escape the stage directory."""

    if not isinstance(name, str) or not SAFE_NAME.match(name) or ".." in name:
        raise ProvenanceError(f"Unsafe {kind} name: {name!r}")


def read_file_once(path, max_bytes=MAX_INPUT_BYTES):
    """Read a regular file's bytes exactly once, refusing symlinks and oversized files."""

    path = Path(path)

    if path.is_symlink():
        raise ProvenanceError(f"Refusing symlink: {path}")

    if not path.is_file():
        raise ProvenanceError(f"Missing file: {path}")

    with path.open("rb") as file:
        data = file.read(max_bytes + 1)

    if len(data) > max_bytes:
        raise ProvenanceError(f"File exceeds {max_bytes} bytes: {path}")

    return data


def read_verified(path, max_bytes=MAX_INPUT_BYTES):
    """Read a file once and return its bytes with their hash.

    Callers must parse the returned bytes, never re-open the file, so the
    hash always describes the data actually used.
    """

    data = read_file_once(path, max_bytes)

    return data, sha256_bytes(data)


def frame_from_bytes(data, **read_csv_options):
    """Parse CSV bytes that have already been read and hashed."""

    return pd.read_csv(io.BytesIO(data), **read_csv_options)


def frame_to_bytes(frame, **to_csv_options):
    """Serialise a table to the exact CSV bytes that will be published."""

    return frame.to_csv(index=False, **to_csv_options).encode("utf-8")


def software_versions():
    """Record the versions that can change numerical output."""

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    sklearn = sys.modules.get("sklearn")

    if sklearn is not None:
        versions["scikit-learn"] = sklearn.__version__

    return versions


def stage_directory(data_dir, stage):
    """Return a stage's directory, refusing names or links that leave data_dir."""

    check_name(stage, "stage")

    data_dir = Path(data_dir).resolve()
    directory = data_dir / stage

    if directory.is_symlink():
        raise ProvenanceError(f"Refusing symlinked stage directory: {directory}")

    if directory.resolve().parent != data_dir:
        raise ProvenanceError(f"Stage directory escapes the data directory: {directory}")

    return directory


def input_record(stage, file_name, digest):
    """Describe one input a stage consumed."""

    return {"stage": stage, "file": file_name, "sha256": digest}


def publish_stage(data_dir, stage, files, inputs, settings, code_path):
    """Publish a stage's outputs as one unit, with the manifest written last.

    files: {file_name: bytes}. inputs: {role: input_record(...)}.
    """

    directory = stage_directory(data_dir, stage)
    directory.parent.mkdir(parents=True, exist_ok=True)

    for file_name in files:
        check_name(file_name, "file")

        if file_name == MANIFEST_NAME:
            raise ProvenanceError("A stage cannot publish its own manifest as an output")

    code_bytes = read_file_once(code_path)

    manifest = {
        "stage": stage,
        "schema_version": SCHEMA_VERSION,
        "files": {
            file_name: {"sha256": sha256_bytes(data), "bytes": len(data)}
            for file_name, data in sorted(files.items())
        },
        "inputs": inputs,
        "settings": settings,
        "software": software_versions(),
        "code": {"file": Path(code_path).name, "sha256": sha256_bytes(code_bytes)},
    }

    token = secrets.token_hex(4)
    temporary = directory.parent / f".{stage}.tmp-{os.getpid()}-{token}"
    temporary.mkdir()

    try:
        for file_name, data in files.items():
            (temporary / file_name).write_bytes(data)

        # The manifest is the commit marker, so it is written last.
        (temporary / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        # Move any previous version aside, then move the new one into place.
        previous = None

        if directory.exists():
            previous = directory.parent / f".{stage}.old-{os.getpid()}-{token}"
            directory.rename(previous)

        temporary.rename(directory)

        if previous is not None:
            shutil.rmtree(previous)

    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return manifest


def load_stage(data_dir, stage, expected_files):
    """Verify a published stage and return its manifest and file bytes.

    Each file is read once; the returned bytes are the ones that were hashed.
    Raises ProvenanceError if the stage is missing, incomplete, unsafe, or
    does not match its manifest.
    """

    directory = stage_directory(data_dir, stage)

    if not directory.is_dir():
        raise ProvenanceError(f"Stage '{stage}' has not been published")

    manifest_path = directory / MANIFEST_NAME

    if not manifest_path.exists():
        raise ProvenanceError(f"Stage '{stage}' is incomplete: no manifest (interrupted before publication?)")

    try:
        manifest = json.loads(read_file_once(manifest_path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"Stage '{stage}' has an unreadable manifest") from error

    if manifest.get("stage") != stage or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ProvenanceError(f"Stage '{stage}' manifest does not describe this stage")

    listed = manifest.get("files", {})

    for file_name in listed:
        check_name(file_name, "file")

    if set(listed) != set(expected_files):
        raise ProvenanceError(
            f"Stage '{stage}' lists {sorted(listed)}; expected exactly {sorted(expected_files)}"
        )

    contents = {}

    for file_name in sorted(expected_files):
        path = directory / file_name

        # A symlink is refused even if it points inside the stage.
        if path.is_symlink():
            raise ProvenanceError(f"Refusing symlink in stage '{stage}': {file_name}")

        if path.resolve().parent != directory.resolve():
            raise ProvenanceError(f"File escapes stage '{stage}': {file_name}")

        data, digest = read_verified(path)

        if digest != listed[file_name]["sha256"]:
            raise ProvenanceError(f"Stage '{stage}': {file_name} does not match its manifest hash")

        contents[file_name] = data

    return manifest, contents


def check_input(manifest, role, upstream_stage, upstream_file, upstream_digest):
    """Confirm a stage was built from the upstream file actually present."""

    recorded = manifest.get("inputs", {}).get(role)

    if recorded is None:
        raise ProvenanceError(f"Stage '{manifest.get('stage')}' does not record its {role} input")

    expected = input_record(upstream_stage, upstream_file, upstream_digest)

    if recorded != expected:
        raise ProvenanceError(
            f"Stage '{manifest.get('stage')}' was built from a different {role} "
            f"({recorded.get('sha256', '?')[:12]}…) than the one present ({upstream_digest[:12]}…)"
        )