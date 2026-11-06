# =============================================================================
# SAD-Lite — Game Configuration Tests
# =============================================================================
#
# PURPOSE
# Check that config/games.yaml loads, and that every way of writing a
# misleading configuration is rejected before any watcher or disarm proposal
# could act on it.
#
# WHAT IT TESTS
# - The real config/games.yaml loads and covers every game in the catalogue.
# - Strict loading: repeated keys (between games and inside one), loose
#   booleans (yes/no/on/off), anchors, aliases, and merge keys are rejected.
# - Validation: unknown settings, missing settings, booleans as counts,
#   floats as counts, negative counts, rates outside [0, 1], non-finite
#   values, duplicate app_ids and names, malformed app_ids, cross-field
#   rules, and overrides without reasons (or reasons without overrides).
# - Overrides apply only to their game and are re-validated.
#
# Run:
#     python -m pytest tests/ -v
#     python tests/test_game_config.py
# =============================================================================

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import game_config  # noqa: E402
import generate_users  # noqa: E402


CONFIG_PATH = ROOT / "config" / "games.yaml"
REAL = CONFIG_PATH.read_text(encoding="utf-8")


def load_text(text):
    return game_config.validate(game_config.parse(text.encode("utf-8")))


def rejected(text, expected):
    try:
        load_text(text)
    except game_config.ConfigError as error:
        assert expected in str(error), f"rejected for another reason: {error}"
        return
    raise AssertionError(f"expected rejection mentioning {expected!r}")


def changed(old, new, count=1):
    assert REAL.count(old) >= 1, f"fixture text not found: {old!r}"
    return REAL.replace(old, new, count)


# --------------------------------------------------------------------------
# The real file
# --------------------------------------------------------------------------

def test_real_configuration_loads_and_covers_the_catalogue():
    _, digest, _, effective = game_config.load(CONFIG_PATH)

    assert len(digest) == 64
    assert set(effective) == {app_id for app_id, _ in generate_users.GAMES}
    assert all(game["enforcement_enabled"] is True for game in effective.values())


def test_every_game_uses_shared_defaults():
    _, _, config, effective = game_config.load(CONFIG_PATH)

    assert all(game["settings"] == config["defaults"] for game in effective.values())


# --------------------------------------------------------------------------
# Strict loading
# --------------------------------------------------------------------------

def test_repeated_key_inside_one_game_is_rejected():
    rejected(changed('    name: "Kettle Creek"\n    enforcement_enabled: true\n',
                     '    name: "Kettle Creek"\n    enforcement_enabled: true\n    enforcement_enabled: false\n'),
             "Repeated key 'enforcement_enabled'")


def test_repeated_key_in_defaults_is_rejected():
    rejected(changed("  min_sessions: 20\n", "  min_sessions: 20\n  min_sessions: 2\n"), "Repeated key 'min_sessions'")


def test_duplicate_app_id_is_rejected():
    rejected(changed('  - app_id: "app_1012"', '  - app_id: "app_1001"'), "Duplicate app_id app_1001")


def test_loose_booleans_are_rejected():
    for spelling in ["yes", "no", "on", "off", "True", "NO"]:
        rejected(changed("    enforcement_enabled: true\n", f"    enforcement_enabled: {spelling}\n"), "must be written true or false")


def test_anchors_aliases_and_merge_keys_are_rejected():
    rejected(changed("defaults:\n", "defaults: &shared\n"), "anchors are not allowed")
    rejected(changed('  - app_id: "app_1001"\n    name: "Neon Drift"\n    enforcement_enabled: true\n',
                     '  - app_id: "app_1001"\n    name: "Neon Drift"\n    enforcement_enabled: true\n    overrides: *shared\n'),
             "aliases are not allowed")
    rejected(changed("calibration:\n", "extra: &a {min_sessions: 1}\ncalibration:\n"), "anchors are not allowed")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def test_unsupported_and_missing_settings_are_rejected():
    rejected(changed("  min_sessions: 20\n", "  min_sessions: 20\n  alert_on_any_spike: true\n"), "unsupported ['alert_on_any_spike']")
    rejected(changed("  min_sessions: 20\n", ""), "missing ['min_sessions']")
    rejected(changed('    name: "Ironwood"\n', '    name: "Ironwood"\n    priority: 1\n'), "may have")
    rejected(changed("version: 1\n", "version: 1\nnotes: anything\n"), "Top level")


def test_counts_must_be_real_integers():
    rejected(changed("  min_sessions: 20\n", "  min_sessions: true\n"), "must be an integer, not bool")
    rejected(changed("  min_sessions: 20\n", "  min_sessions: 20.0\n"), "must be an integer, not float")
    rejected(changed("  min_terminations: 10\n", "  min_terminations: -1\n"), "must be at least 1")
    rejected(changed("  window_hours: 6\n", '  window_hours: "6"\n'), "must be an integer, not str")


def test_rates_must_be_finite_and_within_range():
    rejected(changed("  termination_rate_threshold: 0.15\n", "  termination_rate_threshold: 1.5\n"), "between 0 and 1")
    rejected(changed("  baseline_floor_rate: 0.0132\n", "  baseline_floor_rate: -0.01\n"), "between 0 and 1")
    rejected(changed("  termination_rate_threshold: 0.15\n", "  termination_rate_threshold: .nan\n"), "finite number")
    rejected(changed("  baseline_multiplier: 5.0\n", "  baseline_multiplier: .inf\n"), "finite number")
    rejected(changed("  baseline_multiplier: 5.0\n", "  baseline_multiplier: 0.5\n"), "at least 1")
    rejected(changed("  termination_rate_threshold: 0.15\n", "  termination_rate_threshold: false\n"), "finite number")


def test_cross_field_rules():
    rejected(changed("  check_interval_hours: 1\n", "  check_interval_hours: 7\n"), "cannot exceed window_hours")
    rejected(changed("  baseline_lookback_hours: 168\n", "  baseline_lookback_hours: 6\n"), "at least two windows")
    rejected(changed("  min_terminations: 10\n", "  min_terminations: 25\n"), "cannot exceed min_sessions")


def test_identifiers_and_names():
    rejected(changed('"app_1003"', '"game-3"'), "app_id must be a string like app_1001")
    rejected(changed('"app_1003"', "1003"), "app_id must be a string like app_1001")
    rejected(changed('name: "Pale Orbit"', 'name: "Neon Drift"'), "Duplicate game name")
    rejected(changed('name: "Pale Orbit"', 'name: ""'), "non-empty string")


def test_overrides_need_reasons_and_are_revalidated():
    base = '    name: "Vanta Rally"\n    enforcement_enabled: true\n'
    reason = '    override_reason: "Cheaters keep this game\'s baseline persistently elevated."\n'

    rejected(changed(base, base + "    overrides:\n      baseline_multiplier: 3.0\n"), "must be given together")
    rejected(changed(base, base + reason), "must be given together")
    rejected(changed(base, base + "    overrides:\n      baseline_multiplier: 3.0\n    override_reason: short\n"), "must explain")
    rejected(changed(base, base + "    overrides:\n      allow_everything: true\n" + reason), "unsupported settings")
    rejected(changed(base, base + "    overrides:\n      check_interval_hours: 12\n" + reason), "cannot exceed window_hours")

    effective = load_text(changed(base, base + "    overrides:\n      baseline_multiplier: 3.0\n" + reason))

    assert effective["app_1005"]["settings"]["baseline_multiplier"] == 3.0
    assert effective["app_1004"]["settings"]["baseline_multiplier"] == 5.0


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