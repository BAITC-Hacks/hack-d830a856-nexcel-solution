"""Tests for deterministic inference and agent-tool adapters."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.agents.orchestrator import ForecastOrchestrator
from src.agents.tools import get_weather_tool
from src.tools.feature_tool import MODEL_FEATURE_COLUMNS
from src.tools.inference_tool import WindPowerPredictor


class ConstantModel:
    """Small serializable model used to exercise post-processing."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), 1.2)


def weather_frame() -> pd.DataFrame:
    """Build a valid two-hour weather input."""
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T01:00:00Z",
            ],
            "turbine_id": [1, 1],
            "wind_speed_10m": [4.0, 5.0],
            "wind_speed_80m": [7.0, 8.0],
            "wind_speed_100m": [9.0, 26.0],
            "wind_direction_10m": [90.0, 100.0],
            "wind_direction_100m": [90.0, 100.0],
            "temperature_2m": [5.0, 5.5],
        }
    )


class InferenceToolTests(unittest.TestCase):
    """Validate model clipping and the high-wind shutdown rule."""

    def test_predictor_clips_and_applies_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model.pkl"
            joblib.dump(
                {
                    "model": ConstantModel(),
                    "feature_names": MODEL_FEATURE_COLUMNS,
                },
                model_path,
            )
            result = WindPowerPredictor(model_path).predict(weather_frame())

        self.assertEqual(result["predicted_power"].tolist(), [1.0, 0.0])


class AgentToolTests(unittest.TestCase):
    """Validate weather records and safe orchestration serialization."""

    def test_weather_tool_returns_exact_horizon(self) -> None:
        class FakeWeatherService:
            def get_weather(
                self,
                turbine_id: int,
                start_date: object,
                end_date: object,
            ) -> pd.DataFrame:
                return weather_frame()

        with patch(
            "src.agents.tools.HistoricalWeatherService",
            return_value=FakeWeatherService(),
        ):
            result = get_weather_tool(1, "2026-01-01T00:00:00Z", 2)

        self.assertEqual(result["horizon_hours"], 2)
        self.assertEqual(len(result["weather_data"]), 2)

    def test_orchestrator_serializes_tool_result(self) -> None:
        result = ForecastOrchestrator._execute_tool_call(
            {"echo": lambda value: {"value": value}},
            "echo",
            '{"value": 42}',
        )
        self.assertEqual(json.loads(result), {"value": 42})


if __name__ == "__main__":
    unittest.main()
