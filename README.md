# SAD-Lite

## Cloud gaming

Cloud gaming is a way for a user to play the games they want if they do not have the nessceary hardware to do so. Cloud gaming allows the user to outsource storing/game physics logic and graphics. The user only needs a device that can trasmit the user inputs shuch as buttons pressed/toggled and recvies and decodes the compressed audio/video. The user also needs sufficient bandwidth, low latency, and a stable connection. Ethernet or 5 GHz wifi can help, but more bandwidth does not automatically reduce delay.

## Steps for cloud Gaming

1. Client makes input - mouse/keybaord etc this is packaged into small data and sent across internet to server.

2. Server recieves the package and delivers the input to the game running in the VM. The CPU generally handles game logic and much of the simulation, depending on the engine. The GPU renders the graphics and may perform some other computation.

3. Rendered frames are captured and passed to a video encoder whcih compresses the frames using ultra low latency profiles of codecs such as H.264. The exact capture and encoding process depends on the implementation.

4. The encoded bitstream must now be packaged for real time streaming across the internet. WebRTC is one possible technology, but not every cloud gaming service uses it.

5. Client device recives back package and reassembles the bitstream, decodes the audio/video, and outputs to user on screen.

## Virtual Machine

Virtual Machines - a second mini computer on your comouter that runs a Guest OS, which could be the same as the main computer's OS or different.

### TYPE 2 HYPERVISOR

Application that creates the mini computer - runs on the HOST OS and manages access to resources such as CPU and RAM. The Guest OS runs using virtual hardware presented to it. The hypervisor controls access to the physical resources. Example: VirtualBox.

### TYPE 1 HYPERVISOR

Runs directly on the physical hardware rather than as an application on a conventional Host OS - used by compaines and enterprises. A single server can run multiple Virtual Machines.

### GPU virtualization

GPU virtualization allows multiple VMs to share the same GPU through methods such as time-slicing or partitioning - used in scaling efficeently. It does not always mean splitting the GPU into fixed physical parts.

## Project

This project focuses on devloping a program/helper to help employees notice accounts that have red flags behind them. This could be hackers, bots, etc. Given sessions the prgram goes through and aggregates them to rank accounts that are suspicious and explains why it is susicous and what shodul be looked at by an actual human. The supporting sessions can then be inspected. Separately, the game watcher looks for incidents across many players.

## Things A session will record

| Field | Type | Description |
|---|---|---|
| `session_uuid` | String | Which session this is |
| `user_id` | String | Which account this is |
| `account_created_at` | Timestamp | Account created - time |
| `session_start` | Timestamp | What time did it start |
| `session_end` | Timestamp | What time did it end |
| `duration_min` | Decimal number | Session length in minutes |
| `client_ip` | String | Cliend IP address |
| `country` | String | Country associated with the connection |
| `device_id` | String | Which cliend device was used |
| `app_id` | String | Which games was played |
| `game_name` | String | Game name human readable |
| `rig_type` | Category | The cloud machine - basic, performance, ultimate |
| `input_events_per_min` | Decimal number | Average activity per minute |
| `bytes_streamed` | Integer | Amount of streamed data |
| `payment_hash` | String | An opaque payment identifier that connects accounts using the same payment instrument. Generate fictional identifiers for this project. |
| `end_reason` | Category | normal, timeout, crash, terminated_anticheat, terminated_abuse |

### Validation rules

- Every `session_uuid` is unique.
- A `user_id` can appear in many rows.
- `account_created_at` must be at or before `session_start`.
- `session_end` must be after `session_start`.
- `duration_min` must agree with the start and end timestamps, allowing for rounding.
- Input rate and streamed bytes must be nonnegative.
- Each `app_id` should map to a consistent game name.
- Required fields should not be missing.

## Planned Suspicious Activity Signals

SAD-Lite will examine synthetic session records for patterns that may
warrant investigation. A suspicious signal is not proof of abuse.
Each signal must be considered alongside supporting evidence and
possible innocent explanations.

### 1. Unusually Consistent Input Activity

Measure variation in `input_events_per_min` across an account's sessions.

Very low variation across many sessions may suggest automation,
especially when session lengths are also unusually consistent.
However, similar gameplay can produce similar averages.

This field records a session average, so it cannot reveal the timing
of individual inputs within a session.

### 2. Overlapping Sessions on the Same Account

Check whether the same `user_id` has overlapping sessions on different
devices.

Two sessions overlap when:

```text
session_A.start < session_B.end
AND
session_B.start < session_A.end
```

This may indicate account sharing or resale. Reconnections, delayed
session closure, and logging errors are possible alternative explanations.

A user ID identifies an account, not necessarily one person.

### 3. Multiple Accounts Sharing Devices

Count the distinct `user_id` values associated with each `device_id`.

Many accounts using the same device may indicate coordinated
multi-account activity. The evidence becomes stronger when those
accounts also share a `payment_hash` and were created close together.

Shared family computers and gaming cafés can produce similar patterns,
so device sharing alone should not establish abuse.

### 4. Streaming Volume Compared with Session Duration

Calculate the average streaming bitrate:

```text
average_bitrate_bps =
    bytes_streamed * 8 / (duration_min * 60)
```

Compare this measurement with the assumptions used for each `rig_type`.

Unusual values may reveal inconsistent data or unusual sessions.
Resolution, compression, network conditions, and idle time can also
affect streaming volume.

This will initially serve as a data-quality check. In the reference
generator, streaming volume is derived from duration and rig type,
so it is not independent evidence of abuse.

### 5. Frequent Session Terminations

Calculate the proportion of an account's sessions ending in
`terminated_anticheat` or `terminated_abuse`.

Repeated terminations may warrant investigation, especially when
concentrated on particular games. Rates must be considered alongside
session counts: one termination in one session is weak evidence.

A game or platform fault could also terminate innocent users' sessions.

### 6. Rapid Changes in Devices or Locations

Measure distinct devices, IP addresses, and countries used by an
account within a short period, such as one day.

Rapid changes may suggest account sharing or resale. Travel, VPNs,
and changing networks are possible innocent explanations.

An IP address or country change alone should not establish abuse.

### 7. Unusually Consistent Session Lengths

Measure variation in `duration_min` across many sessions.

Repeated sessions of almost identical length may suggest scripted
behaviour, particularly when input activity is also consistent.

Platform session limits and regular player routines can produce
similar patterns.

### 8. Game-Level Termination Spikes

Monitor termination counts and rates for each `app_id` within a
moving time window.

A spike across many players may indicate a game incident rather than
individual account abuse. Detection will consider:

- The number of terminated sessions.
- The proportion of sessions terminated.
- A minimum session count to avoid conclusions from tiny samples.

A detected incident will produce an alert and a proposed configuration
change for human review.

## Initial Detection Priorities

The first rule-based detector will focus on:

1. High termination rates supported by sufficient session history.
2. Shared devices supported by shared payment information.
3. Many devices or countries used by one account in a short period.
4. Low variation in both input activity and session duration.

Overlapping-session detection is an additional planned check.
Streaming-volume checks will initially validate data consistency.
Game-level monitoring will be implemented separately.

Thresholds will be selected after inspecting the generated data.
Evaluation will examine false positives, missed abuse, and innocent
shared-IP users. Ground-truth labels will remain separate from the
features used by the detectors.