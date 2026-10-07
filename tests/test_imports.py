def test_import_research_monitor():
    import research_monitor
    assert research_monitor.__version__ == "0.1.0"

def test_import_config():
    from research_monitor.config import config
    assert config.LOG_LEVEL is not None
