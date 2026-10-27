import numpy as np
import pytest

from src import pipeline


class FakeModel:
    classes_ = np.array(["rice", "maize", "jute"])

    def predict_proba(self, row):
        self.seen = row          # remember what the model was given
        return np.array([[0.7, 0.2, 0.1]])


@pytest.fixture
def fake_env(monkeypatch):
    """Replace network calls and the model with fakes."""
    model = FakeModel()
    monkeypatch.setattr(pipeline, "pincode_to_coords", lambda p: (28.6, 77.2))
    monkeypatch.setattr(pipeline, "get_weather",
                        lambda la, lo: {"temperature": 25.0, "humidity": 60.0, "rainfall": 70.0})
    monkeypatch.setattr(pipeline, "get_soil_ph", lambda la, lo: 7.0)
    monkeypatch.setattr(pipeline, "load_model", lambda: model)
    return model


def test_invalid_pincode_format(monkeypatch):
    def boom(p):
        raise AssertionError("network should not be called")
    monkeypatch.setattr(pipeline, "pincode_to_coords", boom)
    assert "error" in pipeline.recommend("abc")
    assert "error" in pipeline.recommend("123")


def test_pincode_not_found(monkeypatch):
    monkeypatch.setattr(pipeline, "pincode_to_coords", lambda p: None)
    assert pipeline.recommend("000000") == {"error": "Pincode not found"}


def test_manual_ph_is_used(fake_env, monkeypatch):
    def boom(la, lo):
        raise AssertionError("SoilGrids should not be called")
    monkeypatch.setattr(pipeline, "get_soil_ph", boom)
    result = pipeline.recommend("110001", manual_ph=7.2)
    assert result["inputs"]["ph"] == 7.2
    assert result["ph_estimated"] is False


def test_missing_ph_falls_back_to_default(fake_env, monkeypatch):
    monkeypatch.setattr(pipeline, "get_soil_ph", lambda la, lo: None)
    result = pipeline.recommend("110001")
    assert result["inputs"]["ph"] == 6.5
    assert result["ph_estimated"] is True


def test_column_order_matches_training(fake_env):
    pipeline.recommend("110001")
    assert list(fake_env.seen.columns) == ["temperature", "humidity", "ph", "rainfall"]


def test_ph_out_of_range(fake_env):
    assert "error" in pipeline.recommend("110001", manual_ph=20)
    assert "error" in pipeline.recommend("110001", manual_ph=-1)


def test_top3_format(fake_env):
    result = pipeline.recommend("110001")
    assert [c for c, _ in result["top3"]] == ["rice", "maize", "jute"]
    assert result["top3"][0][1] == 70.0