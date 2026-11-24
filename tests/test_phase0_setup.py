from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_workspace_layout_exists():
    expected = [
        ROOT / "lab",
        ROOT / "securetrace",
        ROOT / "configs",
        ROOT / "scripts",
        ROOT / "dashboard",
        ROOT / "tests",
        ROOT / "data",
    ]
    for path in expected:
        assert path.exists(), f"Missing expected path: {path}"


def test_phase0_entrypoints_exist():
    required = [
        ROOT / "scripts" / "train.py",
        ROOT / "scripts" / "generate_lab_runs.py",
        ROOT / "configs" / "example_experiment.yaml",
        ROOT / "requirements.txt",
    ]
    for path in required:
        assert path.exists(), f"Missing required file: {path}"


def test_env_template_mentions_secrets():
    content = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "MLFLOW_TRACKING_URI" in content
    assert "replace-me" in content
