def test_import_research_monitor():
    import research_monitor
    assert research_monitor.__version__ == "0.1.0"

def test_import_config():
    from research_monitor.config import config
    assert config.LOG_LEVEL is not None


def test_summary_can_be_imported_first_in_a_fresh_process():
    import subprocess
    import sys

    result = subprocess.run([
        sys.executable, "-c",
        "from research_monitor.summary import GeminiSummarizer, summarize_decision; "
        "from research_monitor.review import resume_after_review, record_user_feedback; "
        "assert callable(summarize_decision) and callable(resume_after_review) and callable(record_user_feedback)",
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
