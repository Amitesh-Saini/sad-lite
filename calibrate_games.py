# =============================================================================
# SAD-Lite — Game Threshold Calibration
# =============================================================================
#
# PURPOSE
# Choose the game watcher's thresholds from data that contains no injected
# incident, using a documented rule and a stated acceptable alert rate, so the
# settings in config/games.yaml are not tuned against the incident they must
# later detect.
#
# THE SEPARATION
# 1. Calibration data: generator seeds 101-105, with the injected incident
#    disabled. Ordinary noise, lookalikes, and abuse (including cheaters, who
#    raise termination rates on the games they play) are all present.
# 2. A documented rule chooses the calibrated settings (below).
# 3. The chosen settings are written into config/games.yaml and frozen.
# 4. Evaluation happens elsewhere, on different seeds, with and without
#    incidents, reporting missed incidents and alerts outside incidents.
# These seeds are used nowhere else in the project.
#
# WHICH SETTINGS ARE CALIBRATED
# Set by stated reasoning (FIXED below):
# - window_hours 6: long enough to gather a game's evening traffic.
# - check_interval_hours 1: an incident is seen within an hour of the window
#   filling; must not exceed the window.
# - baseline_lookback_hours 168: one week, covering every day of the week.
# - min_sessions 20 and min_terminations 10: a rate from fewer observations
#   is too noisy to act on.
# - min_baseline_sessions 100: "enough history" for a baseline rate.
# - baseline_multiplier 5: a change large enough to be a change, not drift.
# Calibrated here:
# - baseline_floor_rate: the median observed baseline rate across every
#   calibration window with a sufficient baseline. A game whose recent history
#   happens to be unusually clean is compared against a typical rate instead,
#   so a handful of terminations cannot look like a large multiple.
# - termination_rate_threshold: the LOWEST (most sensitive) value on a fixed
#   grid whose false alert episodes on calibration data stay within budget.
#
# ACCEPTABLE ALERT RATE (the budget)
# At most one alert episode per 30 days across all games, so
# one per calibration seed. Every alert here is false, because the
# calibration data contains no incident.
#
# LIMITS
# This shows behaviour on SAD-Lite's own generator, not on real traffic.
#
# Run:
#     python calibrate_games.py
# Writes reports/calibration/<timestamp>/calibration.json and prints the
# values to copy into config/games.yaml.
# =============================================================================

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import game_monitor
import generate_users


CALIBRATION_SEEDS = [101, 102, 103, 104, 105]

FIXED = {
    "window_hours": 6,
    "check_interval_hours": 1,
    "baseline_lookback_hours": 168,
    "min_sessions": 20,
    "min_terminations": 10,
    "min_baseline_sessions": 100,
    "baseline_multiplier": 5.0,
}

THRESHOLD_GRID = [round(value, 2) for value in np.arange(0.05, 0.61, 0.05)]

# One false alert episode per 30 days across all games.
ALERTS_PER_SEED = 1


def calibration_sessions(seed):
    sessions, _, _ = generate_users.build_dataset(seed=seed, include_incident=False)
    return game_monitor.prepare_sessions(pd.DataFrame(sessions))


def count_episodes(datasets, settings):
    """False alert episodes per game across all calibration datasets."""

    counts = {}

    for sessions in datasets:
        times = game_monitor.check_times(sessions, settings["window_hours"], settings["check_interval_hours"])

        for app_id in sorted(sessions["app_id"].unique()):
            _, episodes = game_monitor.evaluate_game(sessions, app_id, settings, times)
            counts[app_id] = counts.get(app_id, 0) + len(episodes)

    return counts


def main():
    datasets = [calibration_sessions(seed) for seed in CALIBRATION_SEEDS]

    # --- Floor: median observed baseline, measured with no floor and no alerts possible ---
    probe = {**FIXED, "baseline_floor_rate": 0.0, "termination_rate_threshold": 1.0}
    baselines = []

    for sessions in datasets:
        times = game_monitor.check_times(sessions, probe["window_hours"], probe["check_interval_hours"])

        for app_id in sorted(sessions["app_id"].unique()):
            windows, _ = game_monitor.evaluate_game(sessions, app_id, probe, times)
            baselines += [window["observed_baseline"] for window in windows if window["baseline_sufficient"]]

    floor = round(float(np.median(baselines)), 4)

    # --- Threshold: the most sensitive grid value within the alert budget ---
    budget = ALERTS_PER_SEED * len(CALIBRATION_SEEDS)
    rows = []

    for threshold in THRESHOLD_GRID:
        settings = {**FIXED, "baseline_floor_rate": floor, "termination_rate_threshold": threshold}
        counts = count_episodes(datasets, settings)
        rows.append({"threshold": threshold, "false_alert_episodes": sum(counts.values()), "by_game": counts})

    chosen = next((row for row in rows if row["false_alert_episodes"] <= budget), None)

    if chosen is None:
        raise SystemExit("No grid value meets the alert budget; revisit the fixed settings")

    table = pd.DataFrame([{"threshold": row["threshold"], "false_alert_episodes": row["false_alert_episodes"]} for row in rows])
    print(table.to_string(index=False))

    baseline_summary = pd.Series(baselines).describe(percentiles=[0.1, 0.5, 0.9])
    print(f"\nObserved baseline rates across {len(baselines)} calibration windows:")
    print(baseline_summary.round(4).to_string())

    print(f"\nBudget: {budget} false alert episodes across {len(CALIBRATION_SEEDS)} seeds (one per 30 days, all games).")
    print(f"Chosen: termination_rate_threshold {chosen['threshold']} ({chosen['false_alert_episodes']} false episodes), "
          f"baseline_floor_rate {floor}.")

    record = {
        "seeds": CALIBRATION_SEEDS,
        "incident": "disabled",
        "fixed_settings": FIXED,
        "floor_rule": "median observed baseline rate across calibration windows with a sufficient baseline",
        "baseline_floor_rate": floor,
        "threshold_rule": "lowest grid value whose false alert episodes stay within budget",
        "grid": THRESHOLD_GRID,
        "budget_episodes": budget,
        "results": rows,
        "termination_rate_threshold": chosen["threshold"],
    }

    output = Path(__file__).resolve().parent / "reports" / "calibration" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    print(f"\nRecord saved to {output / 'calibration.json'}")
    print("\nCopy into config/games.yaml defaults:")

    for name, value in {**FIXED, "termination_rate_threshold": chosen["threshold"], "baseline_floor_rate": floor}.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()