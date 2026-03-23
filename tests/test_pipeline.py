from core.pipeline_builder import build_pipeline
from sklearn.ensemble import RandomForestClassifier


def test_pipeline_build():
    params = {
        "scaler": "standard",
        "imputer": "mean",
        "feature_selection": False
    }

    model = RandomForestClassifier()
    pipeline = build_pipeline(params, model)

    assert pipeline is not None