# =============================================================================
# SAD-Lite — Shared Test Fixture: a Published Pipeline
# =============================================================================
#
# PURPOSE
# Publish every pipeline stage into a directory exactly as the scripts' main()
# functions do, so tests can work on real, verified stages without touching
# the project's own data/.
#
# Used by tests/test_investigate.py and tests/test_enforce.py.
# =============================================================================

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import features  # noqa: E402
import generate_users  # noqa: E402
import provenance  # noqa: E402
import score_model  # noqa: E402
import score_rules  # noqa: E402


def publish_model(data_dir, forest_seed):
    """Publish the model stage from the verified features, with the given forest seed."""

    _, contents = provenance.load_stage(data_dir, "features", ["features.csv"])
    features_bytes = contents["features.csv"]

    model_input = score_model.load_features(features_bytes)
    anomaly_scores, *_ = score_model.score_accounts(model_input, random_state=forest_seed)
    ranked = score_model.rank_and_save_scores(model_input, anomaly_scores)

    provenance.publish_stage(
        data_dir, "model",
        {"model_scores.csv": provenance.frame_to_bytes(ranked),
         "model_settings.json": json.dumps({"random_seed": forest_seed}).encode("utf-8")},
        {"features": provenance.input_record("features", "features.csv", provenance.sha256_bytes(features_bytes))},
        {"random_seed": forest_seed}, score_model.__file__,
    )


def publish_pipeline(data_dir, seed, forest_seed=42):
    """Publish every stage for one generator seed. Returns the labels, for choosing test subjects only."""

    sessions, labels, incident = generate_users.build_dataset(seed=seed)
    sessions_bytes = generate_users.rows_to_csv_bytes(sessions, generate_users.SESSION_FIELDS)
    sessions_digest = provenance.sha256_bytes(sessions_bytes)
    settings = {"seed": seed, "account_counts": generate_users.ACCOUNT_COUNTS}

    provenance.publish_stage(data_dir, "generate", {"sessions.csv": sessions_bytes}, {}, settings, generate_users.__file__)
    provenance.publish_stage(
        data_dir, "truth",
        {"labels.csv": generate_users.rows_to_csv_bytes(labels, ["user_id", "label", "pattern"]),
         "incident_truth.json": json.dumps(incident, indent=2).encode("utf-8")},
        {"sessions": provenance.input_record("generate", "sessions.csv", sessions_digest)},
        settings, generate_users.__file__,
    )

    feature_table = features.build_features(provenance.frame_from_bytes(sessions_bytes))
    features_bytes = provenance.frame_to_bytes(feature_table, na_rep="")
    features_digest = provenance.sha256_bytes(features_bytes)

    provenance.publish_stage(
        data_dir, "features", {"features.csv": features_bytes},
        {"sessions": provenance.input_record("generate", "sessions.csv", sessions_digest)},
        {"min_terminations_for_concentration": features.MIN_TERMINATIONS_FOR_CONCENTRATION}, features.__file__,
    )

    config_path = ROOT / "config" / "rules.yaml"
    config_bytes, config_digest = provenance.read_verified(config_path)
    config = score_rules.yaml.safe_load(config_bytes.decode("utf-8"))
    scores = score_rules.score_features(score_rules.validate_features(provenance.frame_from_bytes(features_bytes)), config)

    provenance.publish_stage(
        data_dir, "rules",
        {"rule_scores.csv": provenance.frame_to_bytes(scores), "rules_config.yaml": config_bytes},
        {"features": provenance.input_record("features", "features.csv", features_digest),
         "config": {"file": config_path.name, "sha256": config_digest}},
        {"config_file": config_path.name}, score_rules.__file__,
    )

    publish_model(data_dir, forest_seed)

    return pd.DataFrame(labels), incident