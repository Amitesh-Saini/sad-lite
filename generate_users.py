# =============================================================================
# SAD-Lite — Synthetic Session Data Generator
# =============================================================================
#
# PURPOSE
# Create fictional cloud-gaming session records so we can develop and evaluate
# suspicious-activity detectors without using real customer data.
#
# Each session records an account's activity: when it played, which device and
# location it used, the game played, input activity, streaming volume, and how
# the session ended.
#
# ACCOUNT TYPES
# - Ordinary users (1,800): varied session counts, mostly evening play, one
#   home IP, one or two devices, and a few favourite games. About 17% of them
#   are innocent lookalikes (see INNOCENT LOOKALIKES below).
# - Dormitory users (60): ordinary users behind one shared IP address. These
#   are innocent lookalikes: the shared IP is the only unusual thing about them.
# - Smurf ring (30): fresh accounts created within 48 hours of one another,
#   sharing two devices, one payment identifier, and one IP address.
# - Account resale (25): two weeks of ordinary use, then a short burst of many
#   IP addresses, devices, and countries after the account changes hands.
# - Cheaters (40): independent accounts with frequent anti-cheat terminations,
#   concentrated on a small number of games.
# - Bots (25): round-the-clock sessions with unusually consistent session
#   lengths and input activity.
#
# INNOCENT LOOKALIKES
# Legitimate users whose behaviour overlaps a suspicious group on some signals
# but not others. Without them every innocent user behaves neatly, and the
# detectors solve an artificially easy problem.
# - Travellers (look like resale): a short trip abroad on their own device.
# - VPN users (look like resale): some sessions exit through a VPN in another
#   country; VPN exit addresses are shared by several VPN users.
# - Families (look like a smurf ring): 2–5 accounts sharing one payment
#   identifier, one home IP, and often one console, created years apart.
# - Unlucky players (look like cheaters): 3–5 anti-cheat terminations,
#   spread across whatever games they happen to play.
# - New players (look like smurfs and bots): accounts created during the
#   month, so they are young at their first session.
# - Night-shift players (look like bots): many sessions at any hour, but with
#   human variation in session length and input activity.
# All lookalikes keep label 0. Their pattern names start with "normal_" so
# evaluation can report which of them each detector flags.
# The fractions were chosen from plausibility before looking at detector
# results, and must not be tuned to reach a target score.
#
# DATASET GENERATION
# - make_mixed_dataset creates every account type with fixed counts, so a new
#   seed changes values but never the population.
# - Account numbers are shuffled before assignment, so a user_id does not
#   reveal which account type generated it.
# - Individual generators return lists of session dictionaries.
# - The dataset generator returns sessions and separate account-level labels.
#
# GAME INCIDENT
# - inject_game_incident adds a surge of short sessions on one game during a
#   six-hour window, most of them ending in anti-cheat termination.
# - Each participant is an ordinary user with exactly one incident session,
#   using their own usual device, IP address, country, and payment identifier.
# - The incident is a game-level event: no participant looks abusive alone.
# - Incident ground truth, including the participants, is saved separately.
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
# All generators share one game catalogue, so each game ID has one name.
# Session end times and streamed bytes are derived from session duration.
# IP addresses, devices, and payment identifiers are random and unique; their
# format never depends on the account type.
# Sessions belonging to one person do not overlap. Resale accounts may overlap,
# because several renters use the same account.
#
# IMPORTANT LIMITATIONS
# All identities, payment identifiers, locations, and events are fictional.
# Behaviour probabilities and thresholds are modelling assumptions, not
# measured statistics about real players.
# About 6% of accounts are abusive, far above any real platform. This flatters
# every precision figure measured on this data.
# Abuse here is stationary: nothing adapts to being detected.
# Shared networks and unusual activity are not proof of abuse; innocent
# examples help evaluate false positives.
# Input activity is a session average, not individual button-press timing.
# Raw synthetic IDs and IP addresses must not reveal account types to detectors.
# Good results on this generated data do not establish real-world accuracy.
#
# SOURCES
# Cohort sizes and most behaviour parameters follow the SAD-Lite reference
# implementation (sadlite/generate_sessions.py). Departures from it are marked
# "Differs from the reference" where they occur. The innocent lookalikes are
# not in the reference; their design and fractions are this project's own.
# =============================================================================


import argparse
import random
import math
from datetime import datetime, timedelta, timezone
import csv
from pathlib import Path
from collections import Counter
import json


# ------------------------------------------------------------
# Shared settings: every generator uses these values.
# ------------------------------------------------------------

# Fictional catalogue: each game ID always has the same name.
GAMES = [
    ("app_1001", "Neon Drift"),
    ("app_1002", "Harbour Siege"),
    ("app_1003", "Pale Orbit"),
    ("app_1004", "Kettle Creek"),
    ("app_1005", "Vanta Rally"),
    ("app_1006", "Ironwood"),
    ("app_1007", "Ledger of Ash"),
    ("app_1008", "Sunken Atlas"),
    ("app_1009", "Quiet Machines"),
    ("app_1010", "Third Harvest"),
    ("app_1011", "Salt & Signal"),
    ("app_1012", "Molten Yard"),
]

GAME_NAMES = dict(GAMES)
APP_IDS = [app_id for app_id, _ in GAMES]

# Games where cheaters concentrate, and the game that has an incident.
CHEAT_APP_IDS = ["app_1002", "app_1005"]
INCIDENT_APP_ID = "app_1007"

COUNTRIES = ["CA", "US", "GB", "DE", "FR", "JP", "BR", "IN"]

# Rig types and how often each is chosen.
RIG_TYPES = ["basic", "performance", "ultimate"]
RIG_WEIGHTS = [45, 35, 20]

# Illustrative streaming rates for our synthetic data, in Mbps.
STREAMING_RATES = {
    "basic": 15,
    "performance": 25,
    "ultimate": 45,
}

# Mostly normal endings, with occasional technical issues
# and rare anti-cheat terminations even for innocent users.
ORDINARY_ENDINGS = ["normal", "timeout", "crash", "terminated_anticheat"]
ORDINARY_ENDING_WEIGHTS = [88, 6, 5, 1]

TERMINATED_ENDINGS = ("terminated_anticheat", "terminated_abuse")

ORDINARY_USERS = 1800

# Share of the 1,800 ordinary users given each lookalike behaviour.
# Chosen before looking at results; do not tune these to move a score.
LOOKALIKE_FRACTIONS = {
    # A few percent of players take a short trip abroad in any given month.
    "normal_traveller": 0.03,
    # Some players route part of their traffic through a VPN.
    "normal_vpn": 0.02,
    # Shared family payment cards and consoles are common.
    "normal_family": 0.03,
    # Anti-cheat has a false-positive rate; a few players are hit repeatedly.
    "normal_unlucky": 0.02,
    # Platforms gain new players every month.
    "normal_new_account": 0.05,
    # Shift workers and very heavy players play at unusual hours.
    "normal_night_shift": 0.02,
}

LOOKALIKE_COUNTS = {
    pattern: round(fraction * ORDINARY_USERS)
    for pattern, fraction in LOOKALIKE_FRACTIONS.items()
}

# VPN exit addresses shared among VPN users, as real VPN services do.
VPN_EXIT_COUNT = 4

# Fixed account counts: 1,980 accounts, 120 of them abusive (6.1%).
# The ordinary users are split into plain "normal" users and lookalikes.
ACCOUNT_COUNTS = {
    "normal": ORDINARY_USERS - sum(LOOKALIKE_COUNTS.values()),
    **LOOKALIKE_COUNTS,
    "dorm_shared_ip": 60,
    "smurf_ring": 30,
    "account_resale": 25,
    "cheater": 40,
    "bot_farm": 25,
}

ABUSIVE_PATTERNS = {"smurf_ring", "account_resale", "cheater", "bot_farm"}

# Account types where several people may use one account at the same time.
SHARED_USE_PATTERNS = {"account_resale"}

SESSION_FIELDS = [
    "session_uuid", "user_id", "account_created_at", "session_start",
    "session_end", "duration_min", "client_ip", "country", "device_id",
    "app_id", "game_name", "rig_type", "input_events_per_min",
    "bytes_streamed", "payment_hash", "end_reason",
]


# ------------------------------------------------------------
# Small helpers used by every generator.
# ------------------------------------------------------------

def clip(value, minimum, maximum):
    """Limit a value to the range [minimum, maximum]."""

    return max(minimum, min(value, maximum))


def poisson(rng, mean):
    """Draw a Poisson count (Knuth's method); random.Random has no Poisson sampler."""

    if mean < 0:
        raise ValueError("mean must be nonnegative")

    limit = math.exp(-mean)
    count = 0
    product = rng.random()

    while product > limit:
        count += 1
        product *= rng.random()

    return count


def new_ip(rng, used_ids):
    """Create a unique fictional private IP address with no type-revealing structure."""

    while True:
        client_ip = f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"

        if client_ip not in used_ids:
            used_ids.add(client_ip)
            return client_ip


def new_identifier(rng, used_ids, prefix):
    """Create a unique opaque identifier such as device_3fa94c1e0b."""

    while True:
        identifier = f"{prefix}_{rng.getrandbits(40):010x}"

        if identifier not in used_ids:
            used_ids.add(identifier)
            return identifier


def make_session(user_id, account_created_at, session_start, duration_min,
                 client_ip, country, device_id, app_id, rig_type,
                 input_events_per_min, payment_hash, end_reason, rng):
    """Build one session record. End time and streamed bytes come from the duration."""

    # Round first so duration_min always agrees with the timestamps.
    duration_min = round(duration_min, 2)
    session_end = session_start + timedelta(minutes=duration_min)

    bitrate_mbps = STREAMING_RATES[rig_type]

    # Add modest variation to the assumed streaming rate.
    bitrate_mbps *= rng.uniform(0.85, 1.15)

    bytes_streamed = int(
        bitrate_mbps
        * 1_000_000
        * (duration_min * 60)
        / 8
    )

    return {
        # Assigned after all sessions exist; see assign_session_ids.
        "session_uuid": None,
        "user_id": user_id,
        "account_created_at": account_created_at.isoformat(),
        "session_start": session_start.isoformat(),
        "session_end": session_end.isoformat(),
        "duration_min": duration_min,
        "client_ip": client_ip,
        "country": country,
        "device_id": device_id,
        "app_id": app_id,
        "game_name": GAME_NAMES[app_id],
        "rig_type": rig_type,
        "input_events_per_min": round(max(0.0, input_events_per_min), 2),
        "bytes_streamed": bytes_streamed,
        "payment_hash": payment_hash,
        "end_reason": end_reason,
    }


def remove_overlaps(sessions, rng, simulation_end):
    """Move each session to start after the previous one ends; drop any that leave the window."""

    sessions.sort(key=lambda session: session["session_start"])

    kept = []
    previous_end = None

    for session in sessions:
        session_start = datetime.fromisoformat(session["session_start"])

        if previous_end is not None and session_start < previous_end:
            # A short gap between sessions, like a real break.
            session_start = previous_end + timedelta(minutes=rng.randint(1, 10))

        if session_start >= simulation_end:
            continue

        session_end = session_start + timedelta(minutes=session["duration_min"])

        session["session_start"] = session_start.isoformat()
        session["session_end"] = session_end.isoformat()

        kept.append(session)
        previous_end = session_end

    return kept


def ordinary_ending(rng):
    """Choose an end reason using the ordinary end-reason weights."""

    return rng.choices(ORDINARY_ENDINGS, weights=ORDINARY_ENDING_WEIGHTS, k=1)[0]


# ------------------------------------------------------------
# Account generators.
# ------------------------------------------------------------

def make_ordinary_user(user_number, rng, simulation_start, used_ids, days=30,
                       sessions_per_day=0.8, play_all_hours=False):
    """Generate ordinary gaming sessions for one fictional user."""

    if user_number < 0:
        raise ValueError("user_number must be nonnegative")

    if days < 1:
        raise ValueError("days must be at least 1")

    # Start the simulation at midnight on the supplied date.
    first_day = simulation_start.replace(hour=0, minute=0, second=0, microsecond=0)
    simulation_end = first_day + timedelta(days=days)

    # ------------------------------------------------------------
    # Account details: generate once and reuse across sessions.
    # ------------------------------------------------------------

    user_id = f"user_{user_number:05d}"

    account_created_at = first_day - timedelta(days=rng.randint(30, 1100))

    country = rng.choice(COUNTRIES)

    # Fictional private-network address, not a real location lookup.
    # Reused for this user's sessions.
    client_ip = new_ip(rng, used_ids)

    payment_hash = new_identifier(rng, used_ids, "payment")

    number_of_devices = rng.randint(1, 2)

    devices = [
        new_identifier(rng, used_ids, "device")
        for _ in range(number_of_devices)
    ]

    rig_type = rng.choices(RIG_TYPES, weights=RIG_WEIGHTS, k=1)[0]

    # Give this user their own typical activity level.
    typical_input_rate = rng.uniform(80, 180)

    # A few favourite games, chosen for most sessions.
    favourites = rng.sample(APP_IDS, k=rng.randint(2, 4))

    # ------------------------------------------------------------
    # Session details: generate separately for each session.
    # ------------------------------------------------------------

    sessions = []

    for day in range(days):

        # A varying number of sessions each day, 0.8 on average by default.
        for _ in range(poisson(rng, sessions_per_day)):

            # Ordinary users mostly play in the evening;
            # night-shift players start sessions at any hour.
            if play_all_hours:
                hour = rng.randint(0, 23)
            else:
                hour = int(clip(rng.gauss(20, 3), 7, 23))

            session_start = first_day + timedelta(
                days=day,
                hours=hour,
                minutes=rng.randint(0, 59),
            )

            # Most sessions are moderate in length, with some longer ones.
            duration_min = clip(rng.lognormvariate(3.7, 0.55), 5.0, 320.0)

            if rng.random() < 0.80:
                app_id = rng.choice(favourites)
            else:
                app_id = rng.choice(APP_IDS)

            sessions.append(make_session(
                user_id=user_id,
                account_created_at=account_created_at,
                session_start=session_start,
                duration_min=duration_min,
                client_ip=client_ip,
                country=country,
                device_id=rng.choice(devices),
                app_id=app_id,
                rig_type=rig_type,
                input_events_per_min=rng.gauss(typical_input_rate, 25),
                payment_hash=payment_hash,
                end_reason=ordinary_ending(rng),
                rng=rng,
            ))

    # Ensure every generated account has at least one session.
    if not sessions:
        session_start = first_day + timedelta(days=rng.randrange(days), hours=20)

        sessions.append(make_session(
            user_id=user_id,
            account_created_at=account_created_at,
            session_start=session_start,
            duration_min=clip(rng.lognormvariate(3.7, 0.55), 5.0, 320.0),
            client_ip=client_ip,
            country=country,
            device_id=rng.choice(devices),
            app_id=rng.choice(favourites),
            rig_type=rig_type,
            input_events_per_min=rng.gauss(typical_input_rate, 25),
            payment_hash=payment_hash,
            end_reason=ordinary_ending(rng),
            rng=rng,
        ))

    return remove_overlaps(sessions, rng, simulation_end)


def make_bot_user(user_number, rng, simulation_start, used_ids, days=30):
    """Generate synthetic sessions with consistent, round-the-clock bot behaviour."""

    if user_number < 0:
        raise ValueError("user_number must be nonnegative")

    if days < 1:
        raise ValueError("days must be at least 1")

    first_day = simulation_start.replace(hour=0, minute=0, second=0, microsecond=0)
    simulation_end = first_day + timedelta(days=days)

    # Account characteristics: generated once per user.
    user_id = f"user_{user_number:05d}"

    # Bot accounts are relatively new.
    account_created_at = first_day - timedelta(days=rng.randint(5, 60))

    country = rng.choice(COUNTRIES)
    client_ip = new_ip(rng, used_ids)
    payment_hash = new_identifier(rng, used_ids, "payment")
    device_id = new_identifier(rng, used_ids, "device")

    # One game on the cheapest rig, every session.
    app_id = rng.choice(APP_IDS)
    rig_type = "basic"

    # Different bots have different baselines.
    # Each bot stays close to its own baselines across sessions.
    typical_input_rate = rng.uniform(180, 240)
    typical_duration = rng.uniform(90, 150)

    sessions = []

    for day in range(days):

        # A new random starting point each day, so over the month
        # sessions begin in every hour of the day.
        phase_minutes = rng.randint(0, 239)

        # Six sessions a day, four hours apart.
        for slot in range(6):

            # Small variation around this bot's typical session length.
            duration_min = clip(rng.gauss(typical_duration, 2.0), 10.0, 230.0)

            session_start = first_day + timedelta(
                days=day,
                minutes=phase_minutes + slot * 240,
            )

            # Bot sessions mostly end normally without anti-cheat detection.
            end_reason = "normal" if rng.random() < 0.95 else "timeout"

            sessions.append(make_session(
                user_id=user_id,
                account_created_at=account_created_at,
                session_start=session_start,
                duration_min=duration_min,
                client_ip=client_ip,
                country=country,
                device_id=device_id,
                app_id=app_id,
                rig_type=rig_type,
                # Small variation around this bot's typical input activity.
                input_events_per_min=rng.gauss(typical_input_rate, 2.5),
                payment_hash=payment_hash,
                end_reason=end_reason,
                rng=rng,
            ))

    # Differs from the reference: a session that would overlap the next one,
    # across a day boundary, is moved to start after it ends.
    return remove_overlaps(sessions, rng, simulation_end)


def account_sharing_users(user_number, rng, simulation_start, used_ids, days=30):
    """Generate an account that is used normally, then resold or rented to strangers."""

    if user_number < 0:
        raise ValueError("user_number must be nonnegative")

    if days < 17:
        raise ValueError("days must be at least 17 for the resale period")

    first_day = simulation_start.replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    # Account details remain the same across all sessions.
    user_id = f"user_{user_number:05d}"

    # Resold accounts are established accounts.
    account_created_at = first_day - timedelta(
        days=rng.randint(200, 1000)
    )

    payment_hash = new_identifier(rng, used_ids, "payment")
    rig_type = rng.choices(RIG_TYPES, weights=RIG_WEIGHTS, k=1)[0]
    typical_input_rate = rng.uniform(80, 180)

    # The original owner's home setup.
    home_country = rng.choice(COUNTRIES)
    home_ip = new_ip(rng, used_ids)
    home_device = new_identifier(rng, used_ids, "device")

    # The renters' setups after the account changes hands.
    renter_countries = rng.sample(COUNTRIES, k=4)

    # Differs from the reference: each renter IP belongs to one country,
    # so a country change always comes with an IP change.
    renter_ips = [
        {"client_ip": new_ip(rng, used_ids), "country": renter_countries[index % 4]}
        for index in range(9)
    ]

    renter_devices = [
        new_identifier(rng, used_ids, "device")
        for _ in range(6)
    ]

    sessions = []

    # Outer loop: days 0–13, ordinary use by the owner.
    for day in range(0, 14):

        for _ in range(poisson(rng, 0.7)):

            hour = int(clip(rng.gauss(20, 3), 8, 23))

            session_start = first_day + timedelta(
                days=day,
                hours=hour,
                minutes=rng.randint(0, 59),
            )

            sessions.append(make_session(
                user_id=user_id,
                account_created_at=account_created_at,
                session_start=session_start,
                duration_min=clip(rng.lognormvariate(3.7, 0.5), 5.0, 300.0),
                client_ip=home_ip,
                country=home_country,
                device_id=home_device,
                app_id=rng.choice(APP_IDS),
                rig_type=rig_type,
                input_events_per_min=rng.gauss(typical_input_rate, 25),
                payment_hash=payment_hash,
                end_reason=ordinary_ending(rng),
                rng=rng,
            ))

    # Days 14–16: the account changes hands and is used heavily,
    # at any hour, from many locations and devices.
    for day in range(14, 17):

        for _ in range(poisson(rng, 4.0)):

            session_start = first_day + timedelta(
                days=day,
                hours=rng.randint(0, 23),
                minutes=rng.randint(0, 59),
            )

            renter_ip = rng.choice(renter_ips)

            sessions.append(make_session(
                user_id=user_id,
                account_created_at=account_created_at,
                session_start=session_start,
                duration_min=clip(rng.lognormvariate(3.5, 0.6), 5.0, 300.0),
                client_ip=renter_ip["client_ip"],
                country=renter_ip["country"],
                device_id=rng.choice(renter_devices),
                app_id=rng.choice(APP_IDS),
                rig_type=rig_type,
                # Different people produce different activity levels.
                input_events_per_min=rng.gauss(130, 40),
                payment_hash=payment_hash,
                end_reason=ordinary_ending(rng),
                rng=rng,
            ))

    # Several renters may play at once, so overlaps are kept.
    sessions.sort(key=lambda session: session["session_start"])

    return sessions


def make_coordinated_accounts(user_numbers, rng, simulation_start, used_ids, days=30):
    """Create a smurf ring: fresh accounts sharing two devices, one payment, and one IP."""

    if len(user_numbers) < 2:
        raise ValueError("A ring needs at least two accounts")

    if days < 6:
        raise ValueError("days must be at least 6 for the ring's activity period")

    first_day = simulation_start.replace(hour=0, minute=0, second=0, microsecond=0)
    simulation_end = first_day + timedelta(days=days)

    # Everything one person behind many accounts would reuse.
    shared_devices = [
        new_identifier(rng, used_ids, "device"),
        new_identifier(rng, used_ids, "device"),
    ]
    shared_payment = new_identifier(rng, used_ids, "payment")
    shared_ip = new_ip(rng, used_ids)
    country = rng.choice(COUNTRIES)

    # Accounts are created within 48 hours of one another, from day 3.
    ring_creation = first_day + timedelta(days=3)

    sessions = []

    for user_number in user_numbers:
        user_id = f"user_{user_number:05d}"
        account_created_at = ring_creation + timedelta(hours=rng.randint(0, 47))

        rig_type = rng.choices(RIG_TYPES, weights=RIG_WEIGHTS, k=1)[0]
        typical_input_rate = rng.uniform(80, 180)

        user_sessions = []

        # Play begins on day 5, after every account exists.
        for day in range(5, days):

            for _ in range(poisson(rng, 1.3)):

                session_start = first_day + timedelta(
                    days=day,
                    hours=rng.randint(8, 23),
                    minutes=rng.randint(0, 59),
                )

                # Ordinary-looking sessions: the signal is the shared entities.
                user_sessions.append(make_session(
                    user_id=user_id,
                    account_created_at=account_created_at,
                    session_start=session_start,
                    duration_min=clip(rng.lognormvariate(3.4, 0.5), 5.0, 200.0),
                    client_ip=shared_ip,
                    country=country,
                    device_id=rng.choice(shared_devices),
                    app_id=rng.choice(APP_IDS),
                    rig_type=rig_type,
                    input_events_per_min=rng.gauss(typical_input_rate, 25),
                    payment_hash=shared_payment,
                    end_reason=ordinary_ending(rng),
                    rng=rng,
                ))

        sessions.extend(remove_overlaps(user_sessions, rng, simulation_end))

    return sessions


def make_cheater_user(user_number, rng, simulation_start, used_ids, days=30):
    """Create an account with frequent anti-cheat terminations on a few games."""

    if user_number < 0:
        raise ValueError("user_number must be nonnegative")

    if days < 1:
        raise ValueError("days must be at least 1")

    first_day = simulation_start.replace(hour=0, minute=0, second=0, microsecond=0)
    simulation_end = first_day + timedelta(days=days)

    # Account details: an independent, established account.
    user_id = f"user_{user_number:05d}"
    account_created_at = first_day - timedelta(days=rng.randint(60, 900))

    country = rng.choice(COUNTRIES)
    client_ip = new_ip(rng, used_ids)
    payment_hash = new_identifier(rng, used_ids, "payment")
    device_id = new_identifier(rng, used_ids, "device")

    rig_type = rng.choices(RIG_TYPES, weights=RIG_WEIGHTS, k=1)[0]
    typical_input_rate = rng.uniform(80, 180)

    sessions = []

    for day in range(days):

        for _ in range(poisson(rng, 1.1)):

            hour = int(clip(rng.gauss(21, 3), 6, 23))

            session_start = first_day + timedelta(
                days=day,
                hours=hour,
                minutes=rng.randint(0, 59),
            )

            # Most sessions are on the games this cheater targets.
            on_cheat_game = rng.random() < 0.75

            if on_cheat_game:
                app_id = rng.choice(CHEAT_APP_IDS)
            else:
                app_id = rng.choice(APP_IDS)

            # On those games, anti-cheat usually ends the session early.
            if on_cheat_game and rng.random() < 0.72:
                end_reason = rng.choices(list(TERMINATED_ENDINGS), weights=[9, 1], k=1)[0]
                duration_min = clip(rng.gauss(18, 8), 3.0, 60.0)
            else:
                end_reason = ordinary_ending(rng)
                duration_min = clip(rng.lognormvariate(3.6, 0.5), 5.0, 250.0)

            sessions.append(make_session(
                user_id=user_id,
                account_created_at=account_created_at,
                session_start=session_start,
                duration_min=duration_min,
                client_ip=client_ip,
                country=country,
                device_id=device_id,
                app_id=app_id,
                rig_type=rig_type,
                input_events_per_min=rng.gauss(typical_input_rate, 25),
                payment_hash=payment_hash,
                end_reason=end_reason,
                rng=rng,
            ))

    if not sessions:
        raise ValueError(f"{user_id} produced no sessions; use a different seed")

    return remove_overlaps(sessions, rng, simulation_end)


def make_shared_network_users(user_numbers, rng, simulation_start, used_ids, days=30):
    """Create innocent dormitory accounts sharing one IP address."""

    if len(user_numbers) < 2:
        raise ValueError("A shared network needs at least two accounts")

    # One address and one location for the whole dormitory.
    shared_ip = new_ip(rng, used_ids)

    # Differs from the reference: every dorm user shares one country,
    # because they sit behind one network address.
    country = rng.choice(COUNTRIES)

    sessions = []

    for user_number in user_numbers:
        user_sessions = make_ordinary_user(user_number, rng, simulation_start, used_ids, days)

        # Devices and payment identifiers stay individual.
        for session in user_sessions:
            session["client_ip"] = shared_ip
            session["country"] = country

        sessions.extend(user_sessions)

    return sessions


# ------------------------------------------------------------
# Innocent lookalike generators.
# Each starts from an ordinary user and adds one coherent behaviour.
# ------------------------------------------------------------

def shorten_session(session, rng):
    """Cut a session short, as an anti-cheat termination would, keeping its bitrate."""

    old_duration = session["duration_min"]
    new_duration = round(max(2.0, old_duration * rng.uniform(0.2, 0.8)), 2)

    session_start = datetime.fromisoformat(session["session_start"])

    session["duration_min"] = new_duration
    session["session_end"] = (session_start + timedelta(minutes=new_duration)).isoformat()

    # Preserve the existing average bitrate when duration changes.
    session["bytes_streamed"] = int(session["bytes_streamed"] * new_duration / old_duration)


def make_traveller_user(user_number, rng, simulation_start, used_ids, days=30):
    """Create an ordinary user who takes one short trip abroad on their own device."""

    if days < 11:
        raise ValueError("days must be at least 11 for a trip")

    sessions = make_ordinary_user(user_number, rng, simulation_start, used_ids, days)

    first_day = simulation_start.replace(hour=0, minute=0, second=0, microsecond=0)
    home_country = sessions[0]["country"]

    # The trip: another country, and one to three hotel or mobile networks there.
    trip_country = rng.choice([country for country in COUNTRIES if country != home_country])
    trip_ips = [new_ip(rng, used_ids) for _ in range(rng.randint(1, 3))]

    # Departure in the afternoon, so the travel day can show two countries.
    trip_start = first_day + timedelta(days=rng.randint(3, days - 8), hours=rng.randint(8, 18))
    trip_end = trip_start + timedelta(days=rng.randint(2, 5))

    # Unlike a resold account, the traveller keeps playing on their own laptop.
    travel_device = sessions[0]["device_id"]

    for session in sessions:
        session_start = datetime.fromisoformat(session["session_start"])

        if trip_start <= session_start < trip_end:
            session["country"] = trip_country
            session["client_ip"] = rng.choice(trip_ips)
            session["device_id"] = travel_device

    return sessions


def make_vpn_users(user_numbers, rng, simulation_start, used_ids, days=30):
    """Create ordinary users who send some sessions through a shared VPN exit."""

    if not user_numbers:
        raise ValueError("At least one VPN user is required")

    # A VPN service exits through a few addresses, each in one country,
    # and many customers share each exit.
    vpn_exits = [
        {"client_ip": new_ip(rng, used_ids), "country": country}
        for country in rng.sample(COUNTRIES, k=VPN_EXIT_COUNT)
    ]

    sessions = []

    for user_number in user_numbers:
        user_sessions = make_ordinary_user(user_number, rng, simulation_start, used_ids, days)
        home_country = user_sessions[0]["country"]

        # Each user picks one exit outside their own country.
        vpn_exit = rng.choice([option for option in vpn_exits if option["country"] != home_country])

        # About a third of sessions go through the VPN; the device does not change.
        for session in user_sessions:
            if rng.random() < 0.30:
                session["client_ip"] = vpn_exit["client_ip"]
                session["country"] = vpn_exit["country"]

        sessions.extend(user_sessions)

    return sessions


def split_into_families(user_numbers, rng):
    """Split account numbers into families of 2–5 accounts, never leaving one alone."""

    if len(user_numbers) < 2:
        raise ValueError("Families need at least two accounts")

    families = []
    remaining = list(user_numbers)

    while remaining:
        if len(remaining) <= 5:
            size = len(remaining)
        else:
            size = rng.randint(2, 5)

            # Avoid leaving a single account without a family.
            if len(remaining) - size == 1:
                size = size - 1 if size > 2 else size + 1

        families.append(remaining[:size])
        remaining = remaining[size:]

    return families


def make_family_accounts(user_numbers, rng, simulation_start, used_ids, days=30):
    """Create families: accounts sharing a payment identifier, home IP, and often a console."""

    sessions = []

    for family in split_into_families(user_numbers, rng):

        # Shared by the whole household.
        home_ip = new_ip(rng, used_ids)
        country = rng.choice(COUNTRIES)
        family_payment = new_identifier(rng, used_ids, "payment")
        console = new_identifier(rng, used_ids, "device")

        # Two people cannot use one console at the same time.
        console_bookings = []

        for user_number in family:

            # Unlike a smurf ring, each account keeps an ordinary creation date,
            # so family members joined at different times.
            user_sessions = make_ordinary_user(user_number, rng, simulation_start, used_ids, days)

            for session in user_sessions:
                session["client_ip"] = home_ip
                session["country"] = country
                session["payment_hash"] = family_payment

                if rng.random() < 0.40:
                    session_start = datetime.fromisoformat(session["session_start"])
                    session_end = datetime.fromisoformat(session["session_end"])

                    console_free = all(
                        session_end <= booked_start or booked_end <= session_start
                        for booked_start, booked_end in console_bookings
                    )

                    # Otherwise the member keeps playing on their own device.
                    if console_free:
                        session["device_id"] = console
                        console_bookings.append((session_start, session_end))

            sessions.extend(user_sessions)

    return sessions


def make_unlucky_user(user_number, rng, simulation_start, used_ids, days=30):
    """Create an ordinary user with several anti-cheat false-positive terminations."""

    sessions = make_ordinary_user(user_number, rng, simulation_start, used_ids, days)

    # Three to five terminations, on whichever sessions happen to be hit,
    # so they are not concentrated on the cheat-prone games.
    hits = min(len(sessions), rng.randint(3, 5))

    for session in rng.sample(sessions, k=hits):
        shorten_session(session, rng)
        session["end_reason"] = "terminated_anticheat"

    return sessions


def make_new_account_user(user_number, rng, simulation_start, used_ids, days=30):
    """Create a legitimate player whose account was created during the month."""

    sessions = make_ordinary_user(user_number, rng, simulation_start, used_ids, days)
    sessions.sort(key=lambda session: session["session_start"])

    first_day = simulation_start.replace(hour=0, minute=0, second=0, microsecond=0)

    # Choose a session from day 1 onwards that leaves at least three sessions,
    # and create the account up to a day and a half before it.
    candidates = [
        index
        for index, session in enumerate(sessions[:-2])
        if datetime.fromisoformat(session["session_start"]) >= first_day + timedelta(days=1)
    ]

    # Too few sessions to place a sign-up date: keep the ordinary account age.
    if not candidates:
        return sessions

    first_session = sessions[rng.choice(candidates)]
    first_start = datetime.fromisoformat(first_session["session_start"])
    account_created_at = max(first_day, first_start - timedelta(minutes=rng.randint(10, 36 * 60)))

    # Sessions before the account existed are removed.
    kept = [
        session
        for session in sessions
        if datetime.fromisoformat(session["session_start"]) >= account_created_at
    ]

    for session in kept:
        session["account_created_at"] = account_created_at.isoformat()

    return kept


def make_night_shift_user(user_number, rng, simulation_start, used_ids, days=30):
    """Create a heavy player who starts sessions at any hour, with human variation."""

    # More sessions per day and no evening preference. Durations and input
    # activity stay human, which is what should separate them from bots.
    return make_ordinary_user(user_number, rng, simulation_start, used_ids, days,
                              sessions_per_day=2.0, play_all_hours=True)


def append_labels(labels, sessions, label, pattern):
    """Record one ground-truth row per account, outside the session data."""

    user_ids = sorted({session["user_id"] for session in sessions})

    for user_id in user_ids:
        labels.append({"user_id": user_id, "label": label, "pattern": pattern})


def make_mixed_dataset(simulation_start, days=30, seed=42):
    """Create every account type with fixed counts and shuffled account numbers."""

    rng = random.Random(seed)
    used_ids = set()

    # Shuffle account numbers so a user_id does not reveal its account type.
    total_accounts = sum(ACCOUNT_COUNTS.values())
    user_numbers = list(range(total_accounts))
    rng.shuffle(user_numbers)

    assigned = {}
    position = 0

    for pattern, count in ACCOUNT_COUNTS.items():
        assigned[pattern] = user_numbers[position:position + count]
        position += count

    sessions = []
    labels = []

    for pattern, numbers in assigned.items():
        label = 1 if pattern in ABUSIVE_PATTERNS else 0

        if pattern == "normal":
            generated = []
            for user_number in numbers:
                generated.extend(make_ordinary_user(user_number, rng, simulation_start, used_ids, days))

        elif pattern == "normal_traveller":
            generated = []
            for user_number in numbers:
                generated.extend(make_traveller_user(user_number, rng, simulation_start, used_ids, days))

        elif pattern == "normal_vpn":
            generated = make_vpn_users(numbers, rng, simulation_start, used_ids, days)

        elif pattern == "normal_family":
            generated = make_family_accounts(numbers, rng, simulation_start, used_ids, days)

        elif pattern == "normal_unlucky":
            generated = []
            for user_number in numbers:
                generated.extend(make_unlucky_user(user_number, rng, simulation_start, used_ids, days))

        elif pattern == "normal_new_account":
            generated = []
            for user_number in numbers:
                generated.extend(make_new_account_user(user_number, rng, simulation_start, used_ids, days))

        elif pattern == "normal_night_shift":
            generated = []
            for user_number in numbers:
                generated.extend(make_night_shift_user(user_number, rng, simulation_start, used_ids, days))

        elif pattern == "dorm_shared_ip":
            generated = make_shared_network_users(numbers, rng, simulation_start, used_ids, days)

        elif pattern == "smurf_ring":
            generated = make_coordinated_accounts(numbers, rng, simulation_start, used_ids, days)

        elif pattern == "account_resale":
            generated = []
            for user_number in numbers:
                generated.extend(account_sharing_users(user_number, rng, simulation_start, used_ids, days))

        elif pattern == "cheater":
            generated = []
            for user_number in numbers:
                generated.extend(make_cheater_user(user_number, rng, simulation_start, used_ids, days))

        elif pattern == "bot_farm":
            generated = []
            for user_number in numbers:
                generated.extend(make_bot_user(user_number, rng, simulation_start, used_ids, days))

        else:
            raise ValueError(f"No generator for account type {pattern}")

        sessions.extend(generated)
        append_labels(labels, generated, label, pattern)

    sessions.sort(key=lambda session: session["session_start"])
    labels.sort(key=lambda row: row["user_id"])

    return sessions, labels, used_ids


def inject_game_incident(sessions, labels, rng, incident_start, used_ids,
                         app_id=INCIDENT_APP_ID, duration_hours=6,
                         surge_sessions=330, termination_probability=0.88):
    """Simulate a game incident by adding a surge of short, mostly terminated sessions."""

    if duration_hours <= 0:
        raise ValueError("duration_hours must be positive")

    if not 0 <= termination_probability <= 1:
        raise ValueError("termination_probability must be between 0 and 1")

    if app_id not in GAME_NAMES:
        raise ValueError(f"{app_id} is not in the game catalogue")

    incident_end = incident_start + timedelta(hours=duration_hours)

    # Group each ordinary user's sessions so participants keep their own details.
    # Only plain "normal" users take part, so lookalikes and the incident stay
    # separate explanations when evaluation reports false positives.
    normal_users = {row["user_id"] for row in labels if row["pattern"] == "normal"}
    sessions_by_user = {}

    for session in sessions:
        if session["user_id"] in normal_users:
            sessions_by_user.setdefault(session["user_id"], []).append(session)

    # Differs from the reference: participants are drawn without replacement,
    # so each has exactly one incident session.
    candidates = sorted(sessions_by_user)
    rng.shuffle(candidates)

    added = []
    participants = []

    for user_id in candidates:
        if len(added) == surge_sessions:
            break

        user_sessions = sessions_by_user[user_id]

        # Differs from the reference: the participant's usual device, IP,
        # country, and payment identifier are reused, not replaced.
        template = rng.choice(user_sessions)

        duration_min = clip(rng.gauss(12, 6), 2.0, 60.0)

        # Try a few start times; skip this user if each one overlaps their play.
        for _ in range(5):
            latest_offset = duration_hours * 60 - math.ceil(duration_min)
            session_start = incident_start + timedelta(minutes=rng.randint(0, latest_offset))
            session_end = session_start + timedelta(minutes=round(duration_min, 2))

            overlaps = any(
                session_start < datetime.fromisoformat(other["session_end"])
                and datetime.fromisoformat(other["session_start"]) < session_end
                for other in user_sessions
            )

            if not overlaps:
                break
        else:
            continue

        # Model a faulty anti-cheat check on this game.
        if rng.random() < termination_probability:
            end_reason = "terminated_anticheat"
        else:
            end_reason = "normal"

        session = make_session(
            user_id=user_id,
            account_created_at=datetime.fromisoformat(template["account_created_at"]),
            session_start=session_start,
            duration_min=duration_min,
            client_ip=template["client_ip"],
            country=template["country"],
            device_id=template["device_id"],
            app_id=app_id,
            rig_type=template["rig_type"],
            input_events_per_min=rng.gauss(template["input_events_per_min"], 25),
            payment_hash=template["payment_hash"],
            end_reason=end_reason,
            rng=rng,
        )

        added.append(session)
        participants.append(user_id)

    if len(added) < surge_sessions:
        raise ValueError("Not enough ordinary users were free during the incident window")

    sessions.extend(added)
    sessions.sort(key=lambda session: session["session_start"])

    # Measure the whole window, including ordinary background traffic.
    # The window includes its start but excludes its end.
    window_sessions = 0
    window_terminated = 0
    other_sessions = 0
    other_terminated = 0

    for session in sessions:
        if session["app_id"] != app_id:
            continue

        session_start = datetime.fromisoformat(session["session_start"])
        terminated = session["end_reason"] in TERMINATED_ENDINGS

        if incident_start <= session_start < incident_end:
            window_sessions += 1
            window_terminated += terminated
        else:
            other_sessions += 1
            other_terminated += terminated

    window_rate = window_terminated / window_sessions if window_sessions else 0.0
    baseline_rate = other_terminated / other_sessions if other_sessions else 0.0

    # Return incident ground truth separately from detection inputs.
    return {
        "app_id": app_id,
        "game_name": GAME_NAMES[app_id],
        "incident_start": incident_start.isoformat(),
        "incident_end": incident_end.isoformat(),
        "termination_probability": termination_probability,
        "surge_sessions": len(added),
        "surge_terminated": sum(session["end_reason"] in TERMINATED_ENDINGS for session in added),
        "window_sessions": window_sessions,
        "window_terminated": window_terminated,
        "window_termination_rate": window_rate,
        "baseline_termination_rate": baseline_rate,
        "rate_ratio": window_rate / baseline_rate if baseline_rate else None,
        "participants": sorted(participants),
    }


def assign_session_ids(sessions):
    """Number each user's sessions in time order: session_<user>_<n>."""

    sessions.sort(key=lambda session: (session["session_start"], session["user_id"]))
    counts = Counter()

    for session in sessions:
        user_number = session["user_id"].split("_")[1]
        session["session_uuid"] = f"session_{user_number}_{counts[session['user_id']]:05d}"
        counts[session["user_id"]] += 1


def check_dataset(sessions, labels):
    """Reject a generated dataset that breaks the documented rules."""

    session_ids = [row["session_uuid"] for row in sessions]
    label_ids = [row["user_id"] for row in labels]

    if len(session_ids) != len(set(session_ids)):
        raise ValueError("Duplicate session IDs found")

    if len(label_ids) != len(set(label_ids)):
        raise ValueError("Duplicate account labels found")

    if {row["user_id"] for row in sessions} != set(label_ids):
        raise ValueError("Session accounts and labelled accounts do not match")

    if Counter(row["pattern"] for row in labels) != Counter(ACCOUNT_COUNTS):
        raise ValueError("Account counts do not match ACCOUNT_COUNTS")

    patterns = {row["user_id"]: row["pattern"] for row in labels}
    dorm_ips = set()
    sessions_by_user = {}

    for session in sessions:
        session_start = datetime.fromisoformat(session["session_start"])
        session_end = datetime.fromisoformat(session["session_end"])
        account_created_at = datetime.fromisoformat(session["account_created_at"])

        if account_created_at > session_start:
            raise ValueError(f"{session['session_uuid']} starts before its account was created")

        if session_end <= session_start:
            raise ValueError(f"{session['session_uuid']} does not end after it starts")

        actual_duration = (session_end - session_start).total_seconds() / 60

        if abs(actual_duration - session["duration_min"]) > 0.01:
            raise ValueError(f"{session['session_uuid']} duration disagrees with its timestamps")

        if patterns[session["user_id"]] == "dorm_shared_ip":
            dorm_ips.add(session["client_ip"])

        sessions_by_user.setdefault(session["user_id"], []).append((session_start, session_end))

    if len(dorm_ips) != 1:
        raise ValueError("Dormitory accounts must share exactly one IP address")

    # One person cannot play two sessions at once.
    for user_id, intervals in sessions_by_user.items():
        if patterns[user_id] in SHARED_USE_PATTERNS:
            continue

        intervals.sort()

        for (_, previous_end), (next_start, _) in zip(intervals, intervals[1:]):
            if next_start < previous_end:
                raise ValueError(f"{user_id} has overlapping sessions")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SAD-Lite session data.")
    parser.add_argument("--seed", type=int, default=42, help="record the seed you used in any write-up")
    args = parser.parse_args()

    simulation_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    days = 30

    # Generate every account type with fixed counts.
    sessions, labels, used_ids = make_mixed_dataset(simulation_start, days=days, seed=args.seed)

    # Day 22 of the simulation (counting from day 0, i.e. 23 January),
    # from 18:00 to midnight UTC.
    incident_start = simulation_start + timedelta(days=22, hours=18)
    incident = inject_game_incident(sessions, labels, random.Random(args.seed + 1), incident_start, used_ids)

    print(f"\nGame incident: {incident['game_name']} ({incident['app_id']})")
    print(f"Sessions in incident window: {incident['window_sessions']}")
    print(f"Terminated in incident window: {incident['window_terminated']}")
    print(f"Termination rate in window: {incident['window_termination_rate']:.1%}")
    print(f"Termination rate for this game outside the window: {incident['baseline_termination_rate']:.1%}")

    assign_session_ids(sessions)

    # Basic checks before writing files.
    check_dataset(sessions, labels)

    # Resolve paths relative to this file, regardless of terminal location.
    output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "incident_truth.json").open("w", encoding="utf-8") as file:
        json.dump(incident, file, indent=2)

    with (output_dir / "sessions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SESSION_FIELDS)
        writer.writeheader()
        writer.writerows(sessions)

    with (output_dir / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["user_id", "label", "pattern"])
        writer.writeheader()
        writer.writerows(labels)

    abusive = sum(row["label"] for row in labels)

    print(f"\nSeed: {args.seed}")
    print(f"Generated {len(sessions):,} sessions for {len(labels):,} accounts")
    print(f"Accounts labelled abusive: {abusive} ({abusive / len(labels):.1%})")
    print(f"Files saved in: {output_dir}")

    print("\nAccounts per scenario:")
    for pattern, count in sorted(Counter(row["pattern"] for row in labels).items()):
        print(f"  {pattern}: {count}")


if __name__ == "__main__":
    main()