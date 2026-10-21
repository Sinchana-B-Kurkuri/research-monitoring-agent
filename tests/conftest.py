"""Keep test artifacts and database connections separate from pilot data."""
import os

import pytest

# Tests use explicit mocks; never inherit a developer's live Gemini credential.
# dotenv does not override an existing environment variable, even when empty.
os.environ["GEMINI_API_KEY"] = ""

# Set before application modules are collected. Integration tests require an
# explicitly supplied disposable database, never the application's .env URL.
os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@127.0.0.1:1/research_monitor_test?connect_timeout=1",
)


@pytest.fixture(autouse=True)
def isolated_experiment_outputs(tmp_path, monkeypatch):
    from research_monitor.config import config
    from research_monitor.evaluation import experiments
    from scripts import generate_figures

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    monkeypatch.setattr(experiments, "RESULTS_DIR", result_dir)
    monkeypatch.setattr(generate_figures, "RESULTS_DIR", result_dir)
    monkeypatch.setattr(config, "TEST_ACCESS_LOG_PATH", str(result_dir / "test_access.log"))
