import importlib.util
from pathlib import Path


module_path = Path(__file__).resolve().parents[1] / "app" / "pages" / "4_Dashboard_Impacto.py"
spec = importlib.util.spec_from_file_location("dashboard_page", module_path)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


class DummyPredictor:
    def __init__(self):
        self.calls = []

    def _default_meteorologia(self, horizon_hours):
        return {"horizon_hours": horizon_hours}

    def predict(self, *, start_time, horizon_hours, meteorologia, codigo_postal):
        self.calls.append(codigo_postal)
        return {"total_predicted_kwh": float(len(self.calls))}


def test_populate_map_predictions_uses_cache():
    predictor = DummyPredictor()
    cache = {}

    dashboard._populate_map_predictions(predictor, cache)
    first_call_count = len(predictor.calls)

    dashboard._populate_map_predictions(predictor, cache)

    assert len(cache) > 0
    assert len(predictor.calls) == first_call_count


def test_get_solar_panel_reference_points_returns_bizkaia_locations():
    points = dashboard.get_solar_panel_reference_points()

    assert isinstance(points, list)
    assert len(points) > 0
    assert any(point["name"] == "Bilbao" for point in points)
