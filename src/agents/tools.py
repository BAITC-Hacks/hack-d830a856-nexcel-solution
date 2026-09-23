"""Tool adapters between the agent orchestrator and deterministic services."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd

try:
    from src.tools.inference_tool import WindPowerPredictor
    from src.tools.weather_tool import HistoricalWeatherService
except ModuleNotFoundError:
    from tools.inference_tool import WindPowerPredictor
    from tools.weather_tool import HistoricalWeatherService

MAX_FORECAST_HORIZON_HOURS = 48

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather_tool",
            "description": "Получить архивный прогноз погоды по координатам турбины на дату",
            "parameters": {
                "type": "object",
                "properties": {
                    "turbine_id": {
                        "type": "integer",
                        "enum": [1, 2],
                        "description": "Идентификатор турбины.",
                    },
                    "target_date": {
                        "type": "string",
                        "description": (
                            "Время начала в ISO 8601. Если передана только дата, "
                            "используется 00:00 UTC."
                        ),
                    },
                    "horizon_hours": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 48,
                        "default": 48,
                    },
                },
                "required": ["turbine_id", "target_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_model_tool",
            "description": "Запустить модель прогноза мощности по погодным данным",
            "parameters": {
                "type": "object",
                "properties": {
                    "weather_data": {
                        "type": "array",
                        "description": (
                            "Массив weather_data, который вернул get_weather_tool. "
                            "Передавай его без изменения."
                        ),
                        "items": {"type": "object"},
                    }
                },
                "required": ["weather_data"],
            },
        },
    },
]


def _parse_turbine_id(turbine_id: int | str) -> int:
    """Validate a turbine ID without coercing arbitrary user input."""
    if isinstance(turbine_id, bool):
        raise TypeError("turbine_id must be an integer")
    try:
        parsed = int(turbine_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("turbine_id must be 1 or 2") from exc
    if parsed not in (1, 2):
        raise ValueError("turbine_id must be 1 or 2")
    return parsed


def _parse_target_timestamp(target_date: str) -> pd.Timestamp:
    """Parse a requested UTC forecast start timestamp."""
    timestamp = pd.to_datetime(target_date, errors="coerce", utc=True)
    if pd.isna(timestamp):
        raise ValueError("target_date must be an ISO 8601 date or timestamp")
    return pd.Timestamp(timestamp)


def get_weather_tool(
    turbine_id: int | str,
    target_date: str,
    horizon_hours: int = MAX_FORECAST_HORIZON_HOURS,
) -> dict[str, Any]:
    """Get hourly weather records for one historical forecast horizon."""
    turbine = _parse_turbine_id(turbine_id)
    if isinstance(horizon_hours, bool) or not isinstance(horizon_hours, int):
        raise TypeError("horizon_hours must be an integer from 1 to 48")
    if not 1 <= horizon_hours <= MAX_FORECAST_HORIZON_HOURS:
        raise ValueError("horizon_hours must be an integer from 1 to 48")

    start = _parse_target_timestamp(target_date)
    end = start + timedelta(hours=horizon_hours - 1)
    weather_service = HistoricalWeatherService()
    weather = weather_service.get_weather(turbine, start.date(), end.date())
    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"],
        errors="raise",
        utc=True,
    )
    selected = weather.loc[
        weather["timestamp"].between(start, end, inclusive="both")
    ].copy()
    if len(selected) != horizon_hours:
        raise RuntimeError(
            "Weather service did not return complete hourly coverage for "
            f"{start.isoformat()} through {end.isoformat()}"
        )

    selected["timestamp"] = selected["timestamp"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return {
        "turbine_id": turbine,
        "start_timestamp": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_hours": horizon_hours,
        "weather_data": selected.to_dict(orient="records"),
    }


def run_model_tool(
    weather_data: list[dict[str, Any]] | dict[str, Any],
) -> dict[str, Any]:
    """Run local model inference for weather returned by get_weather_tool."""
    records = (
        weather_data.get("weather_data")
        if isinstance(weather_data, dict)
        else weather_data
    )
    if not isinstance(records, list):
        raise TypeError("weather_data must be a list of hourly weather records")

    predictor = WindPowerPredictor()
    predictions = predictor.predict(pd.DataFrame(records))
    predictions["timestamp"] = predictions["timestamp"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    model_metadata = predictor.metadata()
    return {
        "model": {
            "objective": model_metadata["objective"],
            "validation_metrics": model_metadata["validation_metrics"],
        },
        "forecast": predictions.to_dict(orient="records"),
    }
