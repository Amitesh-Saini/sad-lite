# =============================================================================
# SAD-Lite — Game-Level Termination Windows
# =============================================================================
#
# PURPOSE
# Compute, for each game, the termination rate in every rolling window and
# compare it with that game's own recent baseline. This is the game-level view
# that user-level detection cannot provide: an incident spreads one bad
# session each across hundreds of ordinary players.
#
# Used by calibrate_games.py (to choose thresholds) and disarm_watch.py (to
# raise alerts). Pure computation: reads nothing, writes nothing.
#
# DEFINITIONS
# - A session belongs to a window by its session_end, with [start, end)
#   boundaries, for both the session count and the termination count.
# - A termination is end_reason terminated_anticheat or terminated_abuse.
# - Windows end on check times anchored to UTC midnight of the first day in
#   the data, every check_interval_hours. Only full windows are evaluated.
#   Every window is evaluated, not only each game's busiest one.
# - The baseline is the window's preceding baseline_lookback_hours, excluding
#   the window itself so a spike cannot inflate its own comparison.
# - effective_baseline = max(observed baseline rate, baseline_floor_rate).
#   The report shows both, and whether the floor was applied.
# - Fewer than min_baseline_sessions in the lookback is an insufficient
#   baseline: the window is reported, but no automatic disarm proposal may be
#   made from it.
#
# A WINDOW BREACHES when all four hold:
#   sessions >= min_sessions
#   AND terminations >= min_terminations
#   AND termination rate >= termination_rate_threshold
#   AND termination rate >= baseline_multiplier * effective_baseline
#
# EXCLUDING FLAGGED HISTORY
# Windows are processed in time order. Sessions in a breaching window are
# excluded from every later baseline, once each even when windows overlap, so
# an incident cannot raise the baseline used to judge the period after it.
# If exclusion leaves too little history, the baseline is reported as
# insufficient rather than computed from what remains.
#
# EPISODES
# Consecutive breaching windows form one episode: one alert, not one per hour.
#
# LIMITS
# Checking every window does not guarantee detection: an incident split across
# window boundaries, or below the thresholds, can be missed. A persistently
# elevated baseline (for example, a game many cheaters play) makes further
# increases harder to detect. This detects changes in termination rates, not
# every form of ongoing abuse.
# =============================================================================

from datetime import timedelta

import numpy as np
import pandas as pd


TERMINATED_ENDINGS = ("terminated_anticheat", "terminated_abuse")


def prepare_sessions(sessions):
    """Keep the columns the watcher needs, typed, with session_end as the window timestamp."""

    table = pd.DataFrame({
        "session_uuid": sessions["session_uuid"].astype(str),
        "app_id": sessions["app_id"].astype(str),
        "end": pd.to_datetime(sessions["session_end"], format="ISO8601", utc=True),
        "terminated": sessions["end_reason"].isin(TERMINATED_ENDINGS).to_numpy(),
    })

    if table["end"].isna().any():
        raise ValueError("Every session needs a session_end to be assigned to a window")

    return table.sort_values(["end", "session_uuid"], kind="mergesort").reset_index(drop=True)


def check_times(sessions, window_hours, check_interval_hours):
    """Window end times, anchored to UTC midnight of the first day, covering all sessions."""

    first_day = sessions["end"].min().floor("D")
    last = sessions["end"].max()

    times = []
    current = first_day + timedelta(hours=window_hours)

    while current <= last + timedelta(hours=check_interval_hours):
        times.append(current)
        current += timedelta(hours=check_interval_hours)

    return times


def evaluate_game(sessions, app_id, settings, times=None):
    """Evaluate every window for one game. Returns (windows, episodes)."""

    game = sessions[sessions["app_id"] == app_id]
    ends = game["end"].to_numpy(dtype="datetime64[ns]")
    terminated = game["terminated"].to_numpy()
    ids = game["session_uuid"].to_numpy()

    if times is None:
        times = check_times(sessions, settings["window_hours"], settings["check_interval_hours"])

    window = np.timedelta64(settings["window_hours"], "h")
    lookback = np.timedelta64(settings["baseline_lookback_hours"], "h")
    excluded = np.zeros(len(game), dtype=bool)

    windows = []

    for time in times:
        end = np.datetime64(time.tz_convert(None), "ns")
        start = end - window

        low, high = np.searchsorted(ends, start, side="left"), np.searchsorted(ends, end, side="left")
        base_low = np.searchsorted(ends, start - lookback, side="left")

        count = int(high - low)
        terminations = int(terminated[low:high].sum())
        rate = terminations / count if count else 0.0

        usable = ~excluded[base_low:low]
        baseline_sessions = int(usable.sum())
        baseline_terminations = int((terminated[base_low:low] & usable).sum())
        observed_baseline = baseline_terminations / baseline_sessions if baseline_sessions else None

        sufficient = baseline_sessions >= settings["min_baseline_sessions"]
        effective_baseline = max(observed_baseline or 0.0, settings["baseline_floor_rate"])

        conditions = {
            "sessions": count >= settings["min_sessions"],
            "terminations": terminations >= settings["min_terminations"],
            "rate": rate >= settings["termination_rate_threshold"],
            "relative": sufficient and rate >= settings["baseline_multiplier"] * effective_baseline,
        }

        absolute = conditions["sessions"] and conditions["terminations"] and conditions["rate"]
        breached = absolute and conditions["relative"]

        if breached:
            # Exclude this window's sessions from every later baseline.
            excluded[low:high] = True

        windows.append({
            "app_id": app_id,
            "window_start": pd.Timestamp(start, tz="UTC"),
            "window_end": time,
            "sessions": count,
            "terminations": terminations,
            "rate": rate,
            "baseline_sessions": baseline_sessions,
            "baseline_terminations": baseline_terminations,
            "observed_baseline": observed_baseline,
            "effective_baseline": effective_baseline if sufficient else None,
            "floor_applied": sufficient and (observed_baseline or 0.0) < settings["baseline_floor_rate"],
            "baseline_sufficient": sufficient,
            "conditions": conditions,
            "breached": breached,
            # Thresholds met but the baseline cannot support a proposal.
            "insufficient_baseline_alert": absolute and not sufficient,
            "session_ids": ids[low:high],
        })

    return windows, episodes_from(windows)


def episodes_from(windows):
    """Group consecutive alerting windows into episodes."""

    episodes = []
    current = None

    for window in windows:
        alerting = window["breached"] or window["insufficient_baseline_alert"]

        if alerting and current is not None:
            current["windows"].append(window)
            continue

        if current is not None:
            episodes.append(close_episode(current))
            current = None

        if alerting:
            current = {"windows": [window]}

    if current is not None:
        episodes.append(close_episode(current))

    return episodes


def close_episode(episode):
    windows = episode["windows"]
    peak = max(windows, key=lambda window: (window["rate"], window["terminations"]))

    return {
        "app_id": windows[0]["app_id"],
        "start": windows[0]["window_start"],
        "end": windows[-1]["window_end"],
        "windows": len(windows),
        "peak": peak,
        # A proposal needs a sufficient baseline at the first alerting window.
        "proposal_eligible": windows[0]["breached"],
    }


def evaluate_all(sessions, effective):
    """Evaluate every configured game. Returns (results per game, unmonitored app_ids in the data)."""

    results = {}

    for app_id, game in sorted(effective.items()):
        results[app_id] = evaluate_game(sessions, app_id, game["settings"])

    unmonitored = sorted(set(sessions["app_id"]) - set(effective))

    return results, unmonitored