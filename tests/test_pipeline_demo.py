# =============================================================================
# SAD-Lite — Pipeline Demonstration Tests
# =============================================================================
#
# PURPOSE
# Check the choices run_all.sh delegates to pipeline_demo.py and enforce.py:
# which ban reason a case report supports for the dry-run preview, and that
# the demonstration never reads truth.
#
# WHAT IT TESTS
# - Only Moderate or Strong hypotheses that map to a ban code qualify; Weak,
#   "Not applicable", and non-ban hypotheses (Shared network, game spike)
#   never do.
# - Strongest first; ties broken by code in alphabetical order.
# - The first matching hypothesis is used, as in enforce.ban().
# - No qualifying hypothesis gives an empty list (no ban previewed).
# - pipeline_demo.py never refers to the truth stage or labels.
#
# Run:
#     python -m pytest tests/ -v
#     python tests/test_pipeline_demo.py
# =============================================================================

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import enforce  # noqa: E402


def report(*pairs):
    return {"hypotheses": [{"name": name, "confidence": confidence} for name, confidence in pairs]}


def test_only_moderate_or_strong_ban_hypotheses_qualify():
    assert enforce.supported_reasons(report(
        ("Cheating", "Weak"), ("Automation", "Not applicable"), ("Shared network", "Strong"),
        ("Caught in a game-wide termination spike", "Strong"), ("Nothing stands out", "Not applicable"),
    )) == []


def test_strongest_first_then_alphabetical():
    supported = enforce.supported_reasons(report(
        ("Cheating", "Moderate"), ("Automation", "Moderate"),
        ("Account sharing or resale", "Strong"), ("Coordinated multi-accounting", "Moderate"),
    ))

    assert supported == [("ACCOUNT_SHARING", "Strong"), ("AUTOMATION_ABUSE", "Moderate"),
                         ("CHEATING", "Moderate"), ("MULTI_ACCOUNTING", "Moderate")]


def test_first_matching_hypothesis_is_used():
    assert enforce.supported_reasons(report(("Cheating", "Weak"), ("Cheating", "Strong"))) == []


def test_demonstration_never_reads_truth():
    source = (ROOT / "pipeline_demo.py").read_text(encoding="utf-8")
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))

    assert "truth" not in code and "labels" not in code


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
    sys.exit(1 if failures else 0)