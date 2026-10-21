"""Available-evidence evaluation without invented human labels."""
import numpy as np
import pandas as pd
from tests.test_review_ui import db_session


def test_available_data_report_runs_without_fabricating_missing_labels(db_session, tmp_path, monkeypatch):
    from research_monitor.evaluation import final_evaluation as evaluation
    sample = pd.DataFrame({"paper_id":["a","b","c","d"], "split":["dev","test","test","test"]})
    corpus = pd.DataFrame({"paper_id":["a","b","c","d","ref"], "is_reference":[False]*4+[True], "title":["Title"]*5,"abstract":["Abstract"]*5})
    labels = pd.DataFrame({"paper_id":["a","b","c"],"label":["relevant","relevant","not_relevant"],"user_id":["developer"]*3})
    original_read = pd.read_parquet
    monkeypatch.setattr(pd, "read_parquet", lambda path,*args,**kwargs: sample.copy() if "evaluation_sample" in str(path) else corpus.copy())
    monkeypatch.setattr(evaluation, "load_latest_labels", lambda: labels)
    monkeypatch.setattr(evaluation.np, "load", lambda *args,**kwargs: np.array([[1.]+[0.]*383]*5))
    monkeypatch.setattr(evaluation, "_compute_seed_vector", lambda: np.array([1.]+[0.]*383))
    def calculate(test_df, test_embeddings, seed, **kwargs):
        assert set(test_df.paper_id) == {"b","c"}
        assert set(kwargs["dev_df"].paper_id) == {"a"}
        return pd.DataFrame([{"baseline":"B2","precision":1.}])
    monkeypatch.setattr(evaluation, "run_experiment_e1_gate_quality", calculate)
    report = evaluation.run_final_evaluation_suite(db_session=db_session, results_dir=tmp_path/"results",
        pilot_export_dir=tmp_path/"export", survey_dir=tmp_path/"surveys", allow_partial_labels=True)
    assert report["e1_gate_quality"]["status"] == "COMPLETED_PARTIAL"
    assert report["label_coverage"] == {"total":4,"labeled":3,"missing":1,"scope":"labeled_subset"}
    assert not report["numbers_frozen"]
    assert report["completion_status"] == "COMPLETED_WITH_AVAILABLE_DATA"
    strict = evaluation.run_final_evaluation_suite(db_session=db_session, results_dir=tmp_path/"strict",
        pilot_export_dir=tmp_path/"strict_export", survey_dir=tmp_path/"surveys", allow_partial_labels=False)
    assert strict["e1_gate_quality"]["status"] == "BLOCKED_INCOMPLETE_DATA"


