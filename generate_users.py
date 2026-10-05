# =============================================================================
# SAD-Lite — Synthetic Session Data Generator
# =============================================================================
#
# PURPOSE
# Create fictional cloud-gaming session records so we can develop and evaluate
# suspicious-activity detectors without using real customer data.
#
# Each session records an account's activity: when it played, which device and
# location it used, the game played, input activity, streaming volume, session
# end reason, and any simulated payment event.
#
# ACCOUNT TYPES
# - Ordinary users: varied session lengths and input activity.
# - Bots: unusually consistent input activity and session lengths.
# - Account sharing: several devices and locations, with possible overlaps.
# - Coordinated accounts: shared devices/payment identifiers and similar
#   account creation dates.
# - Cheaters: more frequent anti-cheat or abuse terminations.
# - Payment-failure cheaters: cheater behaviour plus frequent payment failures.
# - Innocent shared networks: households, colleges, and workplaces sharing an
#   IP address, sometimes also sharing devices or payment identifiers.
#
# DATASET GENERATION
# - make_mixed_dataset creates a mixture of separate account types and groups.
# - make_combined_dataset creates accounts combining multiple behaviours.
# - Individual generators return lists of session dictionaries.
# - Dataset generators return sessions and separate account-level labels.
#
# WHY LABELS ARE SEPARATE
# Labels record which accounts were generated as innocent or abusive.
# They are used to evaluate predictions, not as inputs to the detectors.
# A label of 0 means synthetic innocent; 1 means synthetic abusive.
#
# RANDOMNESS AND CONSISTENCY
# A seeded random-number generator makes runs reproducible when the code,
# parameters, and sequence of calls stay the same.
# Account numbers must be unique across generators to avoid duplicate IDs.
# All generators must use the same game-ID-to-name mapping.
# Session end times and streamed bytes are derived from session duration.
#
# IMPORTANT LIMITATIONS
# All identities, payment identifiers, locations, and events are fictional.
# Behaviour probabilities and thresholds are modelling assumptions, not
# measured statistics about real players.
# Shared networks, shared payments, unusual activity, and failed payments
# are not proof of abuse; innocent examples help evaluate false positives.
# Payment events represent simulated purchases or renewals during sessions.
# Input activity is a session average, not individual button-press timing.
# Raw synthetic IDs and IP prefixes must not reveal account types to detectors.
# Good results on this generated data do not establish real-world accuracy.
# =============================================================================


import random
from datetime import datetime, timedelta, timezone
import csv
from pathlib import Path
from collections import Counter
import json

def add_payment_fields(sessions, rng, attempt_probability=0.15, failure_probability=0.02):
    """Add fictional payment events. Failure probability is per attempt."""

    for session in sessions:
        attempted = rng.random() < attempt_probability

        session["payment_attempted"] = attempted
        session["payment_status"] = "not_attempted"
        session["payment_failure_reason"] = None

        if attempted:
            if rng.random() < failure_probability:
                session["payment_status"] = "failed"
                session["payment_failure_reason"] = rng.choice(["insufficient_funds", "issuer_declined", "processing_error"])
            else:
                session["payment_status"] = "succeeded"

    return sessions


def make_ordinary_user(user_number, rng, simulation_start, days=30):
    """Generate ordinary gaming sessions for one fictional user."""

    if user_number < 0:
        raise ValueError("user_number must be nonnegative")

    if days < 1:
        raise ValueError("days must be at least 1")

    # Shared catalogue: each game ID always has the same name.
    games = [
        ("app_1001", "Call of Duty"),
        ("app_1002", "Destiny 2"),
        ("app_1003", "Rocket League"),
    ]

    # Illustrative streaming rates for our synthetic data, in Mbps.
    streaming_rates = {
        "basic": 15,
        "performance": 25,
        "ultimate": 45,
    }

    # Start the simulation at midnight on the supplied date.
    first_day = simulation_start.replace(hour=0, minute=0, second=0, microsecond=0)

    # ------------------------------------------------------------
    # Account details: generate once and reuse across sessions.
    # ------------------------------------------------------------

    user_id = f"user_{user_number:05d}"

    account_created_at = first_day - timedelta(days=rng.randint(30, 1000))

    country = rng.choice(["CA", "US", "GB", "DE", "FR", "JP"])

    # Fictional private-network address, not a real location lookup.
    # Reused for this user's sessions.
    ip_number = user_number % 65536
    client_ip = f"10.0.{ip_number // 256}.{ip_number % 256}"

    payment_hash = f"payment_{user_number:05d}"

    number_of_devices = rng.randint(1, 2)

    devices = [
        f"device_{user_number:05d}_{i}"
        for i in range(number_of_devices)
    ]

    rig_type = rng.choice(["basic", "performance", "ultimate"])

    # Give this user their own typical activity level.
    typical_input_rate = rng.uniform(80, 180)

    # This simple version allows one session per active day.
    active_days = [
        day
        for day in range(days)
        if rng.random() < 0.65
    ]

    # Ensure every generated account has at least one session.
    if not active_days:
        active_days = [rng.randrange(days)]

    # ------------------------------------------------------------
    # Session details: generate separately for each session.
    # ------------------------------------------------------------

    sessions = []

    for session_number, day in enumerate(active_days):

        session_uuid = (
            f"session_{user_number:05d}_{session_number:05d}"
        )

        # Ordinary users in this simplified model play in the evening.
        session_start = first_day + timedelta(
            days=day,
            hours=rng.randint(17, 22),
            minutes=rng.randint(0, 59),
        )

        # Most sessions are moderate in length, with some longer ones.
        sampled_duration = rng.lognormvariate(3.7, 0.5)

        duration_min = round(max(5.0, min(sampled_duration, 180.0)), 2,)

        session_end = session_start + timedelta(minutes=duration_min)

        device_id = rng.choice(devices)
        app_id, game_name = rng.choice(games)

        input_events_per_min = round(max(0.0, rng.gauss(typical_input_rate, 25)), 2,)

        # Mostly normal endings, with occasional technical issues
        # and rare anti-cheat terminations even for innocent users.
        end_reason = rng.choices(
            [
                "normal",
                "timeout",
                "crash",
                "terminated_anticheat",
            ],
            weights=[88, 6, 5, 1],
            k=1,
        )[0]

        bitrate_mbps = streaming_rates[rig_type]

        # Add modest variation to the assumed streaming rate.
        bitrate_mbps *= rng.uniform(0.85, 1.15)

        bytes_streamed = int(
            bitrate_mbps
            * 1_000_000
            * (duration_min * 60)
            / 8
        )

        session = {
            "session_uuid": session_uuid,
            "user_id": user_id,
            "account_created_at": account_created_at.isoformat(),
            "session_start": session_start.isoformat(),
            "session_end": session_end.isoformat(),
            "duration_min": duration_min,
            "client_ip": client_ip,
            "country": country,
            "device_id": device_id,
            "app_id": app_id,
            "game_name": game_name,
            "rig_type": rig_type,
            "input_events_per_min": input_events_per_min,
            "bytes_streamed": bytes_streamed,
            "payment_hash": payment_hash,
            "end_reason": end_reason,
        }

        sessions.append(session)

    return sessions

def make_bot_user(user_number, rng, simulation_start, days=30):
    """Generate synthetic sessions with consistent bot-like behaviour."""

    if user_number < 0:
        raise ValueError("user_number must be nonnegative")

    if days < 1:
        raise ValueError("days must be at least 1")

    # Keep this catalogue consistent across all user generators.
    games = [
        ("app_1001", "Call of Duty"),
        ("app_1002", "Destiny 2"),
        ("app_1003", "Rocket League"),
    ]

    # Illustrative streaming rates in Mbps.
    streaming_rates = {
        "basic": 15,
        "performance": 25,
        "ultimate": 45,
    }

    first_day = simulation_start.replace(hour=0, minute=0, second=0, microsecond=0)

    # Account characteristics: generated once per user.
    user_id = f"user_{user_number:05d}"

    account_created_at = first_day - timedelta(days=rng.randint(30, 1000))

    country = rng.choice(["CA", "US", "GB", "DE", "FR", "JP"])

    # Fictional private IP; country is generated independently.
    ip_number = user_number % 65536
    client_ip = f"10.0.{ip_number // 256}.{ip_number % 256}"

    payment_hash = f"payment_{user_number:05d}"

    number_of_devices = rng.randint(1, 2)
    devices = [
        f"device_{user_number:05d}_{i}"
        for i in range(number_of_devices)
    ]

    rig_type = rng.choice(["basic", "performance", "ultimate"])

    # Different bots have different baselines.
    # Each bot stays close to its own baselines across sessions.
    typical_input_rate = rng.uniform(80, 180)
    typical_duration = rng.uniform(45, 120)

    # One session per active day, with a 90% chance of being active.
    active_days = [
        day
        for day in range(days)
        if rng.random() < 0.90
    ]

    if not active_days:
        active_days = [rng.randrange(days)]

    sessions = []

    for session_number, day in enumerate(active_days):

        session_uuid = (f"session_{user_number:05d}_{session_number:05d}")

        # Sessions can start at any hour.
        session_start = first_day + timedelta(
            days=day,
            hours=rng.randint(0, 23),
            minutes=rng.randint(0, 59),
        )

        # Small variation around this bot's typical session length.
        duration_min = round(max(5.0, min(rng.gauss(typical_duration, 1.0), 180.0), ) ,2,)

        session_end = session_start + timedelta(minutes=duration_min)

        device_id = rng.choice(devices)
        app_id, game_name = rng.choice(games)

        # Small variation around this bot's typical input activity.
        input_events_per_min = round(max(0.0, rng.gauss(typical_input_rate, 1.5)), 2,)

        # Bot sessions can end normally without anti-cheat detection.
        end_reason = rng.choices(
            [
                "normal",
                "timeout",
                "crash",
                "terminated_anticheat",
            ],
            weights=[88, 6, 5, 1],
            k=1,
        )[0]

        bitrate_mbps = streaming_rates[rig_type]
        bitrate_mbps *= rng.uniform(0.85, 1.15)

        bytes_streamed = int(
            bitrate_mbps
            * 1_000_000
            * (duration_min * 60)
            / 8
        )

        session = {
            "session_uuid": session_uuid,
            "user_id": user_id,
            "account_created_at": account_created_at.isoformat(),
            "session_start": session_start.isoformat(),
            "session_end": session_end.isoformat(),
            "duration_min": duration_min,
            "client_ip": client_ip,
            "country": country,
            "device_id": device_id,
            "app_id": app_id,
            "game_name": game_name,
            "rig_type": rig_type,
            "input_events_per_min": input_events_per_min,
            "bytes_streamed": bytes_streamed,
            "payment_hash": payment_hash,
            "end_reason": end_reason,
        }

        sessions.append(session)

    return sessions


def account_sharing_users(user_number, rng, simulation_start, days=30):

    if user_number < 0:
        raise ValueError("user_number must be nonnegative")

    if days < 1:
        raise ValueError("days must be at least 1")

    # Keep this catalogue consistent across all generators.
    games = [
        ("app_1001", "Call of Duty"),
        ("app_1002", "Destiny 2"),
        ("app_1003", "Rocket League"),
    ]

    # Illustrative streaming rates in Mbps.
    streaming_rates = {
        "basic": 15,
        "performance": 25,
        "ultimate": 45,
    }

    first_day = simulation_start.replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    # Account details remain the same across all sessions.
    user_id = f"user_{user_number:05d}"

    account_created_at = first_day - timedelta(
        days=rng.randint(30, 1000)
    )

    payment_hash = f"payment_{user_number:05d}"
    rig_type = rng.choice(["basic", "performance", "ultimate"])
    typical_input_rate = rng.uniform(80, 180)

    # Create stable device–IP–country combinations for this account.
    number_of_devices = rng.randint(3, 6)

    countries = rng.sample(
        ["CA", "US", "GB", "DE", "FR", "JP"],
        k=number_of_devices,
    )

    access_profiles = []

    for device_number in range(number_of_devices):

        # Fictional private IP addresses; no real geolocation lookup.
        # The second component separates this from the ordinary pool.
        ip_number = user_number % 65536
        client_ip = (
            f"10.{device_number + 1}."
            f"{ip_number // 256}.{ip_number % 256}"
        )

        access_profiles.append({
            "device_id": f"device_{user_number:05d}_{device_number}",
            "client_ip": client_ip,
            "country": countries[device_number],
        })

    active_days = [
        day
        for day in range(days)
        if rng.random() < 0.65
    ]

    if not active_days:
        active_days = [rng.randrange(days)]

    sessions = []

    # This counter does not reset between days.
    session_number = 0

    # Outer loop: active days.
    for day in active_days:

        number_of_sessions = rng.randint(2, 5)

        first_session_start = None
        first_session_end = None
        first_profile = None

        # Inner loop: sessions within this day.
        for daily_session_number in range(number_of_sessions):

            profile = rng.choice(access_profiles)

            # On some days, deliberately overlap the second session
            # with the first, using a different device.
            make_overlap = (
                daily_session_number == 1
                and rng.random() < 0.40
            )

            if make_overlap:

                other_profiles = [
                    candidate
                    for candidate in access_profiles
                    if candidate["device_id"] != first_profile["device_id"]
                ]

                profile = rng.choice(other_profiles)

                first_duration_seconds = (
                    first_session_end - first_session_start
                ).total_seconds()

                # Start 10–50% of the way through the first session.
                session_start = first_session_start + timedelta(
                    seconds=first_duration_seconds
                    * rng.uniform(0.10, 0.50)
                )

            else:

                # Start by 20:59 so a session of up to three hours,
                # including the deliberately overlapping session,
                # Sessions may cross midnight
                session_start = first_day + timedelta(
                    days=day,
                    hours=rng.randint(0, 20),
                    minutes=rng.randint(0, 59),
                )

            # Ordinary-looking duration and input activity.
            sampled_duration = rng.lognormvariate(3.7, 0.5)

            duration_min = round(
                max(5.0, min(sampled_duration, 180.0)),
                2,
            )

            session_end = session_start + timedelta(
                minutes=duration_min
            )

            # Save the first session's details for possible overlap.
            if daily_session_number == 0:
                first_session_start = session_start
                first_session_end = session_end
                first_profile = profile

            app_id, game_name = rng.choice(games)

            input_events_per_min = round(
                max(0.0, rng.gauss(typical_input_rate, 25)),
                2,
            )

            end_reason = rng.choices(
                [
                    "normal",
                    "timeout",
                    "crash",
                    "terminated_anticheat",
                ],
                weights=[88, 6, 5, 1],
                k=1,
            )[0]

            bitrate_mbps = streaming_rates[rig_type]
            bitrate_mbps *= rng.uniform(0.85, 1.15)

            bytes_streamed = int(
                bitrate_mbps
                * 1_000_000
                * (duration_min * 60)
                / 8
            )

            session_uuid = (
                f"session_{user_number:05d}_{session_number:05d}"
            )

            session = {
                "session_uuid": session_uuid,
                "user_id": user_id,
                "account_created_at": account_created_at.isoformat(),
                "session_start": session_start.isoformat(),
                "session_end": session_end.isoformat(),
                "duration_min": duration_min,
                "client_ip": profile["client_ip"],
                "country": profile["country"],
                "device_id": profile["device_id"],
                "app_id": app_id,
                "game_name": game_name,
                "rig_type": rig_type,
                "input_events_per_min": input_events_per_min,
                "bytes_streamed": bytes_streamed,
                "payment_hash": payment_hash,
                "end_reason": end_reason,
            }

            sessions.append(session)
            session_number += 1

    # Randomly chosen start times may be out of order.
    sessions.sort(key=lambda session: session["session_start"])

    return sessions



def make_coordinated_accounts(first_user_number, rng, simulation_start, days=30, group_size=4):
    """Create related accounts sharing payment information and mostly one device."""

    if group_size < 2:
        raise ValueError("group_size must be at least 2")

    first_day = simulation_start.replace(hour=0, minute=0, second=0, microsecond=0)
    creation_anchor = first_day - timedelta(days=rng.randint(4, 14))

    home_country = rng.choice(["CA", "US", "GB", "DE", "FR", "JP"])
    other_countries = [country for country in ["CA", "US", "GB", "DE", "FR", "JP"] if country != home_country]

    shared_payment = f"payment_group_{first_user_number:05d}"
    shared_device = f"device_group_{first_user_number:05d}"
    alternate_device = f"device_group_{first_user_number:05d}_alternate"

    ip_number = first_user_number % 65536
    home_ip = f"10.20.{ip_number // 256}.{ip_number % 256}"

    # Stable alternative country–IP combinations, representing VPN exits.
    vpn_profiles = [
        {"country": country, "client_ip": f"10.{21 + index}.{ip_number // 256}.{ip_number % 256}"}
        for index, country in enumerate(other_countries)
    ]

    sessions = []

    for offset in range(group_size):
        user_number = first_user_number + offset
        user_sessions = make_ordinary_user(user_number, rng, simulation_start, days)

        # Accounts are created within three days of one another.
        account_created_at = creation_anchor + timedelta(hours=rng.randint(0, 72))

        for session in user_sessions:
            session["account_created_at"] = account_created_at.isoformat()
            session["payment_hash"] = shared_payment
            session["device_id"] = shared_device if rng.random() < 0.90 else alternate_device

            if rng.random() < 0.90:
                session["country"] = home_country
                session["client_ip"] = home_ip
            else:
                profile = rng.choice(vpn_profiles)
                session["country"] = profile["country"]
                session["client_ip"] = profile["client_ip"]

        sessions.extend(add_payment_fields(user_sessions, rng))

    return sessions


def apply_cheater_endings(sessions, rng, termination_probability=None):
    """Modify end reasons without changing session identities or timing."""

    if termination_probability is None:
        termination_probability = rng.uniform(0.20, 0.45)

    if not 0 <= termination_probability <= 1:
        raise ValueError("termination_probability must be between 0 and 1")

    for session in sessions:
        if rng.random() < termination_probability:
            session["end_reason"] = rng.choices(["terminated_anticheat", "terminated_abuse"], weights=[9, 1], k=1)[0]
        else:
            session["end_reason"] = rng.choices(["normal", "timeout", "crash"], weights=[89, 6, 5], k=1)[0]

    return sessions


def make_cheater_user(user_number, rng, simulation_start, days=30, payment_hash=None):
    """Create an account with more frequent synthetic anti-cheat terminations."""

    sessions = make_ordinary_user(user_number, rng, simulation_start, days)
    apply_cheater_endings(sessions, rng)

    if payment_hash is not None:
        for session in sessions:
            session["payment_hash"] = payment_hash

    return add_payment_fields(sessions, rng)


def make_cheater_accounts(first_user_number, rng, simulation_start, days=30, group_size=3):
    """Create multiple cheater accounts sharing one payment identifier."""

    if group_size < 2:
        raise ValueError("group_size must be at least 2")

    shared_payment = f"payment_cheater_group_{first_user_number:05d}"
    sessions = []

    for offset in range(group_size):
        user_sessions = make_cheater_user(first_user_number + offset, rng, simulation_start, days, payment_hash=shared_payment)
        sessions.extend(user_sessions)

    return sessions


def make_payment_failure_cheater(user_number, rng, simulation_start, days=30, payment_hash=None):
    """Combine cheater endings with frequent fictional payment failures."""

    sessions = make_cheater_user(user_number, rng, simulation_start, days, payment_hash=payment_hash)

    # Replace the ordinary payment events with this scenario's events.
    return add_payment_fields(sessions, rng, attempt_probability=0.50, failure_probability=0.60)


def make_shared_network_users(first_user_number, rng, simulation_start, days=30, network_type="household", group_size=None):
    """Create innocent accounts sharing a household, college, or workplace IP."""

    size_ranges = {
        "household": (2, 5),
        "college": (20, 60),
        "workplace": (5, 20),
    }

    if network_type not in size_ranges:
        raise ValueError("network_type must be household, college, or workplace")

    if group_size is None:
        minimum, maximum = size_ranges[network_type]
        group_size = rng.randint(minimum, maximum)

    if group_size < 2:
        raise ValueError("group_size must be at least 2")

    ip_number = first_user_number % 65536
    shared_ip = f"10.30.{ip_number // 256}.{ip_number % 256}"
    country = rng.choice(["CA", "US", "GB", "DE", "FR", "JP"])

    shared_device = f"device_network_{first_user_number:05d}"
    household_payment = f"payment_household_{first_user_number:05d}"

    sessions = []

    for offset in range(group_size):
        user_number = first_user_number + offset
        user_sessions = make_ordinary_user(user_number, rng, simulation_start, days)

        # Choose once per account, not separately for every session.
        uses_household_payment = network_type == "household" and rng.random() < 0.75

        for session in user_sessions:
            session["client_ip"] = shared_ip
            session["country"] = country

            if network_type == "household":
                if uses_household_payment:
                    session["payment_hash"] = household_payment

                if rng.random() < 0.40:
                    session["device_id"] = shared_device

            elif network_type == "workplace":
                # Occasional shared machine; payment identifiers stay separate.
                if rng.random() < 0.10:
                    session["device_id"] = shared_device

            # College accounts retain their individual devices and payments.

        sessions.extend(add_payment_fields(user_sessions, rng))

    return sessions


def make_combined_user(user_number, rng, simulation_start, days=30, behaviours=None):
    """Create one account combining two or more supported behaviours."""

    allowed = ("bot", "sharing", "cheater", "payment_failures")

    if behaviours is None:
        behaviours = rng.sample(allowed, k=rng.randint(2, len(allowed)))

    behaviours = set(behaviours)

    if len(behaviours) < 2 or not behaviours.issubset(allowed):
        raise ValueError("Choose at least two of: bot, sharing, cheater, payment_failures")

    # Sharing supplies multiple sessions, locations, and devices.
    if "sharing" in behaviours:
        sessions = account_sharing_users(user_number, rng, simulation_start, days)
    elif "bot" in behaviours:
        sessions = make_bot_user(user_number, rng, simulation_start, days)
    else:
        sessions = make_ordinary_user(user_number, rng, simulation_start, days)

    if "bot" in behaviours:
        typical_input_rate = rng.uniform(80, 180)
        typical_duration = rng.uniform(45, 120)

        for session in sessions:
            old_duration = session["duration_min"]

            # Preserve the existing average bitrate when duration changes.
            average_bytes_per_minute = session["bytes_streamed"] / old_duration
            new_duration = round(max(5.0, min(rng.gauss(typical_duration, 1.0), 180.0)), 2)

            session["duration_min"] = new_duration
            session_start = simulation_start.fromisoformat(session["session_start"])
            session["session_end"] = (session_start + timedelta(minutes=new_duration)).isoformat()
            session["bytes_streamed"] = int(average_bytes_per_minute * new_duration)
            session["input_events_per_min"] = round(max(0.0, rng.gauss(typical_input_rate, 1.5)), 2)

    if "cheater" in behaviours:
        apply_cheater_endings(sessions, rng)

    if "payment_failures" in behaviours:
        add_payment_fields(sessions, rng, attempt_probability=0.50, failure_probability=0.60)
    else:
        add_payment_fields(sessions, rng)

    return sessions


def append_labels(labels, sessions, label, pattern):
    """Record one ground-truth row per account, outside the session data."""

    user_ids = sorted({session["user_id"] for session in sessions})

    for user_id in user_ids:
        labels.append({"user_id": user_id, "label": label, "pattern": pattern})


def make_mixed_dataset(simulation_start, days=30, seed=42, batches=100, first_user_number=0, weights=None):
    """A: Mix separate account types and groups into one dataset."""

    if batches < 1:
        raise ValueError("batches must be at least 1")

    default_weights = {
        "ordinary": 65,
        "bot": 8,
        "sharing": 7,
        "coordinated": 3,
        "cheater_group": 3,
        "payment_failure_cheater": 2,
        "household": 8,
        "college": 1,
        "workplace": 3,
    }

    if weights is None:
        weights = default_weights

    if not weights or any(kind not in default_weights for kind in weights):
        raise ValueError("weights must contain supported account types")

    if any(weight < 0 for weight in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("weights must be nonnegative with a positive total")

    rng = random.Random(seed)
    sessions = []
    labels = []
    next_user_number = first_user_number

    for _ in range(batches):
        kind = rng.choices(list(weights), weights=list(weights.values()), k=1)[0]

        if kind == "ordinary":
            generated = make_ordinary_user(next_user_number, rng, simulation_start, days)
            add_payment_fields(generated, rng)
            label = 0

        elif kind == "bot":
            generated = make_bot_user(next_user_number, rng, simulation_start, days)
            add_payment_fields(generated, rng)
            label = 1

        elif kind == "sharing":
            generated = account_sharing_users(next_user_number, rng, simulation_start, days)
            add_payment_fields(generated, rng)
            label = 1

        elif kind == "coordinated":
            generated = make_coordinated_accounts(next_user_number, rng, simulation_start, days, group_size=rng.randint(3, 6))
            label = 1

        elif kind == "cheater_group":
            generated = make_cheater_accounts(next_user_number, rng, simulation_start, days, group_size=rng.randint(2, 5))
            label = 1

        elif kind == "payment_failure_cheater":
            generated = make_payment_failure_cheater(next_user_number, rng, simulation_start, days)
            label = 1

        else:
            generated = make_shared_network_users(next_user_number, rng, simulation_start, days, network_type=kind)
            label = 0

        sessions.extend(generated)
        append_labels(labels, generated, label, kind)

        # Group generators consume several consecutive account numbers.
        next_user_number += len({session["user_id"] for session in generated})

    sessions.sort(key=lambda session: session["session_start"])

    return sessions, labels


def make_combined_dataset(simulation_start, days=30, seed=43, number_of_users=20, first_user_number=0):
    """B: Generate accounts that each combine multiple behaviours."""

    if number_of_users < 1:
        raise ValueError("number_of_users must be at least 1")

    rng = random.Random(seed)
    sessions = []
    labels = []
    allowed = ("bot", "sharing", "cheater", "payment_failures")

    for offset in range(number_of_users):
        behaviours = rng.sample(allowed, k=rng.randint(2, len(allowed)))
        generated = make_combined_user(first_user_number + offset, rng, simulation_start, days, behaviours=behaviours)

        sessions.extend(generated)
        append_labels(labels, generated, 1, "+".join(sorted(behaviours)))

    sessions.sort(key=lambda session: session["session_start"])

    return sessions, labels


def inject_game_incident(sessions, rng, incident_start, app_id="app_1001", duration_hours=6, termination_probability=0.80):
    """Simulate a game incident by modifying existing session end reasons."""

    if duration_hours <= 0:
        raise ValueError("duration_hours must be positive")

    if not 0 <= termination_probability <= 1:
        raise ValueError("termination_probability must be between 0 and 1")

    if not any(session["app_id"] == app_id for session in sessions):
        raise ValueError(f"No sessions found for {app_id}")

    incident_end = incident_start + timedelta(hours=duration_hours)

    eligible_count = 0
    changed_count = 0
    terminated_count = 0
    affected_users = set()

    for session in sessions:
        session_end = datetime.fromisoformat(session["session_end"])

        # Model a faulty check at session end.
        # The incident window includes its start but excludes its end.
        if session["app_id"] != app_id:
            continue

        if not incident_start <= session_end < incident_end:
            continue

        eligible_count += 1

        if rng.random() < termination_probability:
            if session["end_reason"] not in ("terminated_anticheat", "terminated_abuse"):
                session["end_reason"] = "terminated_anticheat"
                changed_count += 1
                affected_users.add(session["user_id"])

        if session["end_reason"] in ("terminated_anticheat", "terminated_abuse"):
            terminated_count += 1

    # Return incident ground truth separately from detection inputs.
    return {
        "app_id": app_id,
        "incident_start": incident_start.isoformat(),
        "incident_end": incident_end.isoformat(),
        "termination_probability": termination_probability,
        "eligible_sessions": eligible_count,
        "changed_sessions": changed_count,
        "newly_affected_accounts": len(affected_users),
        "terminated_sessions": terminated_count,
        "termination_rate": terminated_count / eligible_count if eligible_count else 0.0,
    }



def main():
    simulation_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    days = 30

    # Generate the mixture of ordinary users and separate abuse scenarios.
    sessions, labels = make_mixed_dataset(simulation_start, days=days, seed=42, batches=100)

    # Account numbers currently begin at zero and are consecutive.
    next_user_number = len(labels)

    # Ensure the required innocent college/dorm group is present.
    if not any(row["pattern"] == "college" for row in labels):
        college_sessions = make_shared_network_users(next_user_number, random.Random(44), simulation_start, days=days, network_type="college", group_size=40)
        sessions.extend(college_sessions)
        append_labels(labels, college_sessions, 0, "college")
        next_user_number += 40

    # Add accounts that combine several behaviours.
    combined_sessions, combined_labels = make_combined_dataset(simulation_start, days=days, seed=43, number_of_users=20, first_user_number=next_user_number)

    sessions.extend(combined_sessions)
    labels.extend(combined_labels)
    sessions.sort(key=lambda session: session["session_start"])

    # Day 23 of the simulation, from 18:00 to midnight UTC.
    incident_start = simulation_start + timedelta(days=22, hours=18)
    incident = inject_game_incident(sessions, random.Random(45), incident_start)

    print(f"\nGame incident: {incident['app_id']}")
    print(f"Sessions in incident window: {incident['eligible_sessions']}")
    print(f"Session endings changed: {incident['changed_sessions']}")
    print(f"Termination rate in window: {incident['termination_rate']:.1%}")

    # Basic checks before writing files.
    session_ids = [row["session_uuid"] for row in sessions]
    label_ids = [row["user_id"] for row in labels]

    if len(session_ids) != len(set(session_ids)):
        raise ValueError("Duplicate session IDs found")

    if len(label_ids) != len(set(label_ids)):
        raise ValueError("Duplicate account labels found")

    if {row["user_id"] for row in sessions} != set(label_ids):
        raise ValueError("Session accounts and labelled accounts do not match")

    # Save relative to the directory where you run the command.
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "incident_truth.json").open("w", encoding="utf-8") as file:
        json.dump(incident, file, indent=2)

    with (output_dir / "sessions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(sessions[0]))
        writer.writeheader()
        writer.writerows(sessions)

    with (output_dir / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["user_id", "label", "pattern"])
        writer.writeheader()
        writer.writerows(labels)

    print(f"Generated {len(sessions):,} sessions for {len(labels):,} accounts")
    print(f"Accounts labelled abusive: {sum(row['label'] for row in labels)}")
    print(f"Files saved in: {output_dir.resolve()}")

    print("\nAccounts per scenario:")
    for pattern, count in sorted(Counter(row["pattern"] for row in labels).items()):
        print(f"  {pattern}: {count}")


if __name__ == "__main__":
    main()