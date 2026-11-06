# =============================================================================
# SAD-Lite — Pipeline Demonstration Helpers (used by run_all.sh)
# =============================================================================
#
# PURPOSE
# Make the two choices run_all.sh needs for its account demonstration, so the
# shell script does not have to parse CSV or JSON itself:
# - select-user: which account to investigate when --demo-user is not given.
# - enforcement-preview: whether the case report supports any ban reason, and
#   if so, preview (never apply) that ban through enforce.py.
#
# WHY THIS IS NEEDED
# Running the whole pipeline must not turn the top-ranked account into a ban.
# A high model rank is a review priority, not evidence, so the preview only
# happens when the investigation itself supports a permitted reason at
# Moderate confidence or above. Otherwise it says so and stops; that is a
# successful outcome, not a failure.
#
# RULES
# - select-user reads the verified model stage only; it never reads truth/.
# - The reason comes from the report's structured hypotheses through
#   enforce.supported_reasons(): enforcement's own code-to-hypothesis
#   mapping and threshold, strongest first, ties by code in alphabetical
#   order.
# - enforce.ban() is called with apply=False and no override, with a reviewer
#   and notes that say this is an automated preview, not a human decision.
#   ban() re-verifies the data and regenerates the report, so a stale or
#   edited report, or a provenance failure, is an error (exit 1), never
#   "no supported reason".
#
# Exit codes: 0 success (including "no ban previewed"), 1 refused or error.
#
# Run (normally from run_all.sh):
#     python pipeline_demo.py select-user --data-dir data
#     python pipeline_demo.py enforcement-preview --data-dir data --report reports/cases/<case>.json
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

import enforce
import provenance


ROOT = Path(__file__).resolve().parent

DEMO_REVIEWER = "run_all.sh demonstration (automated preview, not a human reviewer)"
DEMO_NOTES = ("Automated pipeline preview: the reason was selected from the case report's strongest supported "
              "hypothesis. This is not a human decision and nothing is applied.")
NO_REASON_MESSAGE = ("Investigation completed. No hypothesis supports a permitted ban reason at Moderate "
                     "confidence or above. No ban previewed.")


def select_user(data_dir):
    """The model's rank-1 account, from the verified model stage."""

    _, contents = provenance.load_stage(data_dir, "model", ["model_scores.csv", "model_settings.json"])
    scores = provenance.frame_from_bytes(contents["model_scores.csv"], dtype={"user_id": str})
    top = scores.sort_values(["rank", "user_id"]).iloc[0]

    return str(top["user_id"])


def enforcement_preview(data_dir, report_path, output_dir, enforcement_dir):
    """Preview the best-supported ban for the report's account, or record that none qualifies."""

    try:
        document = json.loads(provenance.read_file_once(report_path).decode("utf-8"))
        report = document["report"]
        account = report["account"]
    except (provenance.ProvenanceError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise enforce.EnforcementError(f"Cannot read an account case report from {report_path}: {error}") from error

    supported = enforce.supported_reasons(report)
    record = {"case_id": report.get("case_id"), "user_id": account, "supported_reasons": supported}

    if not supported:
        record.update({"status": "no_ban_previewed", "message": NO_REASON_MESSAGE})
    else:
        reason = supported[0][0]
        result = enforce.ban(report_path, data_dir, account, reason, DEMO_REVIEWER, DEMO_NOTES,
                             request_key=None, apply=False, override_justification=None,
                             enforcement_dir=enforcement_dir)
        result.pop("suggested_request_key", None)  # random; not part of the preview
        record.update({"status": result["status"], "selected_reason": reason, "preview": result,
                       "message": f"Previewed {reason} for {account} ({result['status']}). Nothing was applied."})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"preview_{record['case_id']}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str), encoding="utf-8")

    return record, path


def main():
    parser = argparse.ArgumentParser(description="Demonstration choices for run_all.sh.")
    commands = parser.add_subparsers(dest="command", required=True)

    select_parser = commands.add_parser("select-user", help="print the model's top-ranked account")
    select_parser.add_argument("--data-dir", type=Path, default=ROOT / "data")

    preview_parser = commands.add_parser("enforcement-preview", help="preview a supported ban, never apply")
    preview_parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    preview_parser.add_argument("--report", type=Path, required=True)
    preview_parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "enforcement_preview")
    preview_parser.add_argument("--enforcement-dir", type=Path, default=ROOT / "enforcement")

    args = parser.parse_args()

    try:
        if args.command == "select-user":
            print(select_user(args.data_dir))
        else:
            record, path = enforcement_preview(args.data_dir, args.report, args.output_dir, args.enforcement_dir)
            print(record["message"])
            print(f"Preview record: {path}")
    except provenance.ProvenanceError as error:
        print(f"Provenance check failed: {error}", file=sys.stderr)
        sys.exit(1)
    except enforce.IntegrityError as error:
        print(f"INTEGRITY FAILURE: {error}", file=sys.stderr)
        sys.exit(1)
    except enforce.EnforcementError as error:
        print(f"Enforcement refused the preview: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()