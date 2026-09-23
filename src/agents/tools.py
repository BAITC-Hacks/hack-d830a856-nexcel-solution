TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather_tool",
            "description": "Получить архивный прогноз погоды по координатам турбины на дату",
            "parameters": {
                "type": "object",
                "properties": {
                    "turbine_id": {"type": "string"},
                    "target_date": {"type": "string"},
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
                "properties": {"weather_data": {"type": "object"}},
                "required": ["weather_data"],
            },
        },
    },
]


def get_weather_tool(*args, **kwargs):
    raise NotImplementedError


def run_model_tool(*args, **kwargs):
    raise NotImplementedError
