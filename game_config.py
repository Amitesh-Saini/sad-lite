# =============================================================================
# SAD-Lite — Game Monitoring Configuration
# =============================================================================
#
# PURPOSE
# Load and validate config/games.yaml: which games the game-level watcher
# monitors, the thresholds it uses, and whether each game's anti-cheat
# enforcement is enabled.
#
# WHY THIS IS NEEDED
# The watcher's only output that changes anything is a proposal to flip one
# game's enforcement_enabled flag. A configuration file that silently loses a
# duplicate entry, reads "no" as false, or accepts a setting nobody uses would
# make that proposal act on something other than what a reviewer reads.
#
# WHAT "ENFORCEMENT ENABLED" MEANS HERE
# Simulated configuration. Setting it to false does not disable a real game
# and does not undo any account ban.
#
# FILE SHAPE
#   version: 1
#   defaults:            every setting, used by every game
#   games:               a list; each entry has app_id, name, enforcement_enabled,
#                        and optionally overrides plus a written override_reason
#   calibration:         free-text record of how the defaults were chosen
#
# STRICT LOADING (before validation sees anything)
# - Repeated keys are rejected in every mapping, not only between games.
#   YAML requires unique keys, but PyYAML silently keeps the last one.
# - Booleans must be spelled exactly true or false. YAML 1.1 also reads
#   yes/no/on/off as booleans; those are rejected.
# - Anchors, aliases, and merge keys are rejected, so every value is written
#   out where it applies.
# - Keys must be strings.
#
# VALIDATION
# - Exactly the known settings: unknown or missing keys are rejected at
#   every level ("unsupported settings").
# - Counts and hours are real integers: type(value) is int, which excludes
#   booleans (bool is a subclass of int in Python).
# - Rates are finite numbers in [0, 1]; the multiplier is finite and >= 1.
# - Cross-field rules: the check interval cannot exceed the window, or an
#   incident between checks could be missed; the lookback must cover at least
#   two windows; minimum terminations cannot exceed minimum sessions.
# - app_id values match app_NNNN and are unique; names are unique strings.
# - An override needs a written override_reason, and an override_reason
#   needs an override. Each game's effective settings are validated again.
#
# The file is read once and parsed from the same bytes that are hashed, so
# a disarm proposal can record exactly which configuration it was computed
# against (see provenance.py).
# =============================================================================

import math
import re
from pathlib import Path

import yaml

import provenance


CONFIG_VERSION = 1

SETTINGS = {
    # Integer settings and their minimum values.
    "window_hours": ("int", 1),
    "check_interval_hours": ("int", 1),
    "baseline_lookback_hours": ("int", 1),
    "min_sessions": ("int", 1),
    "min_terminations": ("int", 1),
    "min_baseline_sessions": ("int", 1),
    # Rates in [0, 1].
    "termination_rate_threshold": ("rate", None),
    "baseline_floor_rate": ("rate", None),
    # How many times the effective baseline the window rate must reach.
    "baseline_multiplier": ("multiplier", None),
}

GAME_REQUIRED = {"app_id", "name", "enforcement_enabled"}
GAME_OPTIONAL = {"overrides", "override_reason"}
TOP_LEVEL_REQUIRED = {"version", "defaults", "games"}
TOP_LEVEL_OPTIONAL = {"calibration"}

APP_ID = re.compile(r"^app_\d{4}$")
MIN_OVERRIDE_REASON = 20


class ConfigError(Exception):
    """games.yaml cannot be loaded or does not satisfy the documented rules."""


# ------------------------------------------------------------
# Strict loading.
# ------------------------------------------------------------

class StrictLoader(yaml.SafeLoader):
    """A SafeLoader that refuses duplicate keys, loose booleans, and aliases."""

    def compose_node(self, parent, index):
        if self.check_event(yaml.events.AliasEvent):
            raise ConfigError("YAML aliases are not allowed; write each value out")

        event = self.peek_event()

        if getattr(event, "anchor", None):
            raise ConfigError("YAML anchors are not allowed; write each value out")

        return super().compose_node(parent, index)


def construct_strict_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    mapping = {}

    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise ConfigError("YAML merge keys (<<) are not allowed")

        key = loader.construct_object(key_node, deep=deep)

        if not isinstance(key, str):
            raise ConfigError(f"Keys must be strings; found {key!r} at line {key_node.start_mark.line + 1}")

        if key in mapping:
            raise ConfigError(f"Repeated key {key!r} at line {key_node.start_mark.line + 1}")

        mapping[key] = loader.construct_object(value_node, deep=deep)

    return mapping


def construct_strict_bool(loader, node):
    value = loader.construct_scalar(node)

    if value not in ("true", "false"):
        raise ConfigError(f"Booleans must be written true or false, not {value!r} (line {node.start_mark.line + 1})")

    return value == "true"


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_strict_mapping)
StrictLoader.add_constructor("tag:yaml.org,2002:bool", construct_strict_bool)


def parse(data):
    """Parse YAML bytes with the strict loader."""

    try:
        return yaml.load(data.decode("utf-8"), Loader=StrictLoader)  # noqa: S506 - StrictLoader derives from SafeLoader
    except ConfigError:
        raise
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ConfigError(f"games.yaml is not valid YAML: {error}") from error


# ------------------------------------------------------------
# Validation.
# ------------------------------------------------------------

def check_value(name, value, where):
    kind, minimum = SETTINGS[name]

    if kind == "int":
        if type(value) is not int:
            raise ConfigError(f"{where}.{name} must be an integer, not {type(value).__name__}")

        if value < minimum:
            raise ConfigError(f"{where}.{name} must be at least {minimum}")

        return

    if type(value) not in (int, float) or not math.isfinite(value):
        raise ConfigError(f"{where}.{name} must be a finite number")

    if kind == "rate" and not 0 <= value <= 1:
        raise ConfigError(f"{where}.{name} must be between 0 and 1")

    if kind == "multiplier" and value < 1:
        raise ConfigError(f"{where}.{name} must be at least 1")


def check_settings(settings, where):
    """Validate one complete set of settings, including cross-field rules."""

    if not isinstance(settings, dict) or set(settings) != set(SETTINGS):
        found = set(settings) if isinstance(settings, dict) else set()
        raise ConfigError(
            f"{where} must contain exactly {sorted(SETTINGS)} "
            f"(missing {sorted(set(SETTINGS) - found)}, unsupported {sorted(found - set(SETTINGS))})"
        )

    for name, value in settings.items():
        check_value(name, value, where)

    if settings["check_interval_hours"] > settings["window_hours"]:
        raise ConfigError(f"{where}: check_interval_hours cannot exceed window_hours, or an incident between checks could be missed")

    if settings["baseline_lookback_hours"] < 2 * settings["window_hours"]:
        raise ConfigError(f"{where}: baseline_lookback_hours must cover at least two windows")

    if settings["min_terminations"] > settings["min_sessions"]:
        raise ConfigError(f"{where}: min_terminations cannot exceed min_sessions")


def validate(config):
    """Validate a parsed configuration; return {app_id: effective settings}."""

    if not isinstance(config, dict):
        raise ConfigError("games.yaml must be a mapping at the top level")

    keys = set(config)

    if not TOP_LEVEL_REQUIRED <= keys or keys - TOP_LEVEL_REQUIRED - TOP_LEVEL_OPTIONAL:
        raise ConfigError(
            f"Top level must have {sorted(TOP_LEVEL_REQUIRED)} and may have {sorted(TOP_LEVEL_OPTIONAL)}; "
            f"found {sorted(keys)}"
        )

    if type(config["version"]) is not int or config["version"] != CONFIG_VERSION:
        raise ConfigError(f"version must be {CONFIG_VERSION}")

    check_settings(config["defaults"], "defaults")

    games = config["games"]

    if not isinstance(games, list) or not games:
        raise ConfigError("games must be a non-empty list")

    effective = {}
    names = set()

    for position, game in enumerate(games):
        where = f"games[{position}]"

        if not isinstance(game, dict):
            raise ConfigError(f"{where} must be a mapping")

        keys = set(game)

        if not GAME_REQUIRED <= keys or keys - GAME_REQUIRED - GAME_OPTIONAL:
            raise ConfigError(
                f"{where} must have {sorted(GAME_REQUIRED)} and may have {sorted(GAME_OPTIONAL)}; found {sorted(keys)}"
            )

        app_id, name = game["app_id"], game["name"]

        if not isinstance(app_id, str) or not APP_ID.match(app_id):
            raise ConfigError(f"{where}.app_id must be a string like app_1001")

        if app_id in effective:
            raise ConfigError(f"Duplicate app_id {app_id}")

        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"{where}.name must be a non-empty string")

        if name in names:
            raise ConfigError(f"Duplicate game name {name!r}")

        if type(game["enforcement_enabled"]) is not bool:
            raise ConfigError(f"{where}.enforcement_enabled must be true or false")

        settings = dict(config["defaults"])

        if ("overrides" in game) != ("override_reason" in game):
            raise ConfigError(f"{where}: overrides and override_reason must be given together")

        if "overrides" in game:
            overrides, reason = game["overrides"], game["override_reason"]

            if not isinstance(overrides, dict) or not overrides:
                raise ConfigError(f"{where}.overrides must be a non-empty mapping")

            unsupported = set(overrides) - set(SETTINGS)

            if unsupported:
                raise ConfigError(f"{where}.overrides has unsupported settings {sorted(unsupported)}")

            if not isinstance(reason, str) or len(reason.strip()) < MIN_OVERRIDE_REASON:
                raise ConfigError(f"{where}.override_reason must explain the override (at least {MIN_OVERRIDE_REASON} characters)")

            settings.update(overrides)

        # Effective settings must satisfy every rule, including cross-field ones.
        check_settings(settings, f"{where} ({app_id}) effective settings")

        effective[app_id] = {"name": name, "enforcement_enabled": game["enforcement_enabled"], "settings": settings}
        names.add(name)

    return effective


def load(path):
    """Read games.yaml once; return (raw bytes, sha256, parsed config, effective settings per game)."""

    try:
        data, digest = provenance.read_verified(Path(path))
    except provenance.ProvenanceError as error:
        raise ConfigError(str(error)) from error

    config = parse(data)
    effective = validate(config)

    return data, digest, config, effective