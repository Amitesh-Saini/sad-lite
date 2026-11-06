# SAD-Lite

Suspicious-activity detection and triage for a simulated cloud-gaming service. It covers rule and Isolation Forest scoring with provenance-checked stages, account investigation, evidence-backed enforcement, game-level anomaly monitoring, and a deterministic agent workflow that prepares reviewable configuration changes on local Git branches.

All data is synthetic, produced by `generate_users.py`. No real users, games, or services are involved, and nothing here acts on a real system.

## Requirements

- Python 3.11 or newer (tested on 3.13)
- Git

## Installation

```bash
git clone https://github.com/Amitesh-Saini/sad-lite.git
cd sad-lite
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run `source .venv/bin/activate` in each new terminal.

## Running

Run the whole pipeline:

```bash
./run_all.sh                    # generate, score, evaluate, investigate, preview, monitor, agent preview
./run_all.sh --proposal-demo    # also create a real disarm branch in a disposable Git workspace
./run_all.sh --test             # run the test suite first
./run_all.sh --replace          # needed when results from a previous run exist
./run_all.sh --help
```

Or run each stage yourself, in this order:

```bash
python generate_users.py --seed 42
python features.py
python score_rules.py
python score_model.py --seed 42
python evaluate.py
python investigate.py --user-id <user_id>
python enforce.py ban --report reports/cases/<case>.json --user-id <user_id> \
    --reason <CODE> --reviewer "<name>" --notes "<why>"          # dry run unless --apply
python disarm_watch.py scan
python disarm_watch.py propose --report reports/disarm/<scan>/alerts.json --alert-id <alert>
python agent_sim.py run --dry-run
python agent_sim.py injection-demo
```

## Tests

```bash
python -m pytest tests/ -q
```

## Outputs

Generated stages go to `data/`, reports to `reports/`, and enforcement records to `enforcement/`. All three are ignored by Git.