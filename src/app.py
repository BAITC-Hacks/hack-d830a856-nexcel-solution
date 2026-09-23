import html
import logging
from datetime import date
from functools import partial

import gradio as gr

try:
    from src.agents.orchestrator import ForecastOrchestrator
    from src.agents.pipeline import ForecastResult, activity_markdown, plot_frame
except ModuleNotFoundError:
    from agents.orchestrator import ForecastOrchestrator
    from agents.pipeline import ForecastResult, activity_markdown, plot_frame

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CUSTOM_CSS = r"""
:root {
    --app-bg: #0B0F14;
    --panel: #11161D;
    --panel-secondary: #151B23;
    --input-bg: #0F141B;
    --border: #252C36;
    --text: #F1F5F9;
    --muted: #94A3B8;
    --accent: #F97316;
    --accent-hover: #EA580C;
    --success: #22C55E;
}

body, .gradio-container {
    background: var(--app-bg) !important;
    color: var(--text) !important;
}

.gradio-container {
    max-width: 1500px !important;
    margin: 0 auto !important;
    min-height: 100vh !important;
    padding: 24px 28px 32px !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

#app-shell { gap: 18px !important; }

#app-header {
    align-items: center !important;
    margin: 0 2px 2px !important;
    padding: 4px 0 12px !important;
}

.app-brand {
    color: var(--text);
    font-size: 25px;
    font-weight: 700;
    letter-spacing: -0.035em;
}

.app-subtitle {
    margin-top: 5px;
    color: var(--muted);
    font-size: 13px;
}

.status-wrap {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    height: 100%;
}

.agent-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border: 1px solid rgba(34, 197, 94, 0.18);
    border-radius: 999px;
    background: rgba(34, 197, 94, 0.08);
    color: #86EFAC;
    font-size: 12px;
    font-weight: 600;
}

.agent-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.08);
}

#dashboard-layout { align-items: stretch !important; }

.dashboard-panel {
    height: 100%;
    min-height: 630px;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    background: var(--panel) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.16) !important;
}

.dashboard-panel > .wrap,
.dashboard-panel > .block { background: transparent !important; }

.panel-heading {
    padding: 17px 18px 14px;
    border-bottom: 1px solid var(--border);
}

.panel-title {
    color: var(--text);
    font-size: 14px;
    font-weight: 700;
}

.panel-subtitle {
    margin-top: 4px;
    color: var(--muted);
    font-size: 12px;
}

#chatbot {
    min-height: 425px !important;
    border: 0 !important;
    background: transparent !important;
}

#chatbot .message {
    max-width: 82% !important;
    border: 1px solid var(--border) !important;
    border-radius: 15px !important;
    box-shadow: none !important;
    font-size: 14px !important;
    line-height: 1.55 !important;
}

#chatbot .message.user {
    border-color: #2A3441 !important;
    background: #1C2430 !important;
}

#chatbot .message.bot,
#chatbot .message.assistant { background: var(--panel-secondary) !important; }

.quick-actions {
    gap: 8px !important;
    padding: 0 18px 10px !important;
}

.quick-action {
    min-width: auto !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    background: transparent !important;
    color: var(--muted) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
}

.quick-action:hover {
    border-color: #4B5563 !important;
    background: var(--panel-secondary) !important;
    color: var(--text) !important;
}

#input-row {
    gap: 10px !important;
    margin: 0 !important;
    padding: 14px 18px 18px !important;
    border-top: 1px solid var(--border);
}

#chat-input textarea,
#chat-input input {
    min-height: 52px !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    background: var(--input-bg) !important;
    color: var(--text) !important;
    box-shadow: none !important;
    font-size: 14px !important;
}

#chat-input textarea:focus,
#chat-input input:focus {
    border-color: rgba(249, 115, 22, 0.65) !important;
    box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.08) !important;
}

#chat-input textarea::placeholder,
#chat-input input::placeholder { color: #64748B !important; }

#send-button {
    height: 52px !important;
    min-width: 110px !important;
    max-width: 140px !important;
    border: 1px solid #FB923C !important;
    border-radius: 14px !important;
    background: var(--accent) !important;
    color: white !important;
    box-shadow: none !important;
    font-size: 13px !important;
    font-weight: 700 !important;
}

#send-button:hover { background: var(--accent-hover) !important; }

.forecast-placeholder {
    display: flex;
    min-height: 380px;
    align-items: center;
    justify-content: center;
    margin: 16px 18px 0;
    border: 1px dashed #334155;
    border-radius: 14px;
    background: var(--input-bg);
    color: var(--muted);
    text-align: center;
}

.forecast-placeholder-title {
    color: var(--text);
    font-size: 14px;
    font-weight: 650;
}

.forecast-placeholder-text {
    margin-top: 7px;
    font-size: 12px;
}

#forecast-controls {
    gap: 8px !important;
    padding: 14px 18px 0 !important;
    align-items: end !important;
}

#forecast-chart { margin: 12px 18px 0 !important; }

#agent-activity {
    margin: 0 18px 18px !important;
    color: var(--muted);
    font-size: 12px;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    margin: 12px 18px 18px;
}

.metric-card {
    padding: 13px 14px;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--input-bg);
}

.metric-label {
    color: var(--muted);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.metric-value {
    margin-top: 7px;
    color: var(--text);
    font-size: 16px;
    font-weight: 650;
}

footer { display: none !important; }

@media (max-width: 900px) {
    .gradio-container { padding: 16px !important; }
    .dashboard-panel { min-height: auto; }
    #chatbot { min-height: 360px !important; }
}

@media (max-width: 600px) {
    .gradio-container { padding: 12px !important; }
    .app-brand { font-size: 21px; }
    .status-wrap { justify-content: flex-start; margin-top: 10px; }
    .forecast-placeholder { min-height: 260px; }
    .metrics-grid { grid-template-columns: 1fr; }
    #send-button { min-width: 90px !important; }
}
"""

CHAT_HEADER_HTML = """
<div class="panel-heading">
    <div class="panel-title">AI Assistant</div>
    <div class="panel-subtitle">Спросите о прогнозе, погоде или состоянии турбины</div>
</div>
"""

FORECAST_HEADER_HTML = """
<div class="panel-heading">
    <div class="panel-title">Прогноз</div>
    <div class="panel-subtitle">Почасовая выработка · 24–48 часов</div>
</div>
"""

FORECAST_PLACEHOLDER_HTML = """
<div class="forecast-placeholder">
    <div>
        <div class="forecast-placeholder-title">Прогноз ещё не построен</div>
        <div class="forecast-placeholder-text">График появится после построения прогноза</div>
    </div>
</div>
"""

TURBINE_CHOICES = {"Турбина 1": (1,), "Турбина 2": (2,), "Обе турбины": (1, 2)}
DEFAULT_ISSUE_DATE = "2026-01-31"
ACTIVITY_PLACEHOLDER = "Этапы агента появятся после построения прогноза."


def _metric_card(label: str, value: str) -> str:
    return (
        f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div></div>'
    )


def metrics_html(result: ForecastResult | None) -> str:
    """Real metrics of the latest forecast; nothing hardcoded."""
    if result is None:
        cards = [_metric_card("Статус", "Прогноз не построен")]
    else:
        metrics = (result.model_info or {}).get("validation_metrics") or {}
        mae = f"MAE {metrics['mae']:.3f}" if "mae" in metrics else "нет метрик"
        cards = [
            _metric_card("Выпуск · горизонт", f"{result.issue_date} · {result.horizon_hours} ч"),
            _metric_card("Статус", result.status),
            _metric_card("Модель (январь 2026)", mae if result.model_info else "не загружена"),
        ]
        for name, stats in result.turbines.items():
            cards.append(_metric_card(
                name.replace("turbine_", "Турбина "),
                f"средн. {stats['mean_power']:.2f} · пик {stats['peak_power']:.2f} "
                f"в {stats['peak_time_utc'][5:16].replace('T', ' ')} UTC",
            ))
    return f'<div class="metrics-grid">{"".join(cards)}</div>'


INITIAL_HISTORY = [
    {
        "role": "assistant",
        "content": (
            "Здравствуйте. Я AI-агент прогнозирования ВЭС. Могу построить прогноз "
            "выработки, проанализировать погодные условия или объяснить результат модели."
        ),
    }
]

orchestrator = ForecastOrchestrator()
agent = orchestrator.agent  # one ForecastAgent for the button and the chat tools
NO_CHANGE = (gr.update(), gr.update(), gr.update(), gr.update())


def forecast_views(result: ForecastResult | None):
    """Plot, placeholder, activity and metrics for a forecast result."""
    has_forecast = result is not None and result.forecast is not None
    return (
        gr.update(value=plot_frame(result), visible=has_forecast),
        gr.update(visible=not has_forecast),
        activity_markdown(result) if result else ACTIVITY_PLACEHOLDER,
        metrics_html(result),
    )


def build_forecast(issue_date: str, turbines: str, horizon: str, *, update_only: bool = False):
    """Direct path without LLM: Gradio -> ForecastAgent -> plot/metrics."""
    try:
        parsed_date = date.fromisoformat(issue_date.strip())
    except ValueError:
        error = f"❌ Дата должна быть в формате YYYY-MM-DD, получено: `{issue_date}`"
        return gr.update(), gr.update(), gr.update(), error, gr.update()
    method = agent.check_update if update_only else agent.run
    result = method(parsed_date, int(horizon), TURBINE_CHOICES[turbines])
    return result.summary(), *forecast_views(result)


def send_message(message: str, history: list, forecast_context: dict | None):
    history = list(history or [])
    if not message.strip():
        return history, "", forecast_context, *NO_CHANGE

    before = agent.last_result
    history.append({"role": "user", "content": message})
    try:
        answer = orchestrator.chat(message, history[:-1], forecast_context)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        LOGGER.exception("Unable to process chat message")
        answer = f"Ошибка: {error}"
    history.append({"role": "assistant", "content": answer})

    result = agent.last_result
    if result is before:  # the chat answered without building a new forecast
        return history, "", forecast_context, *NO_CHANGE
    return history, "", result.summary(), *forecast_views(result)


with gr.Blocks(title="NEXCEL Wind AI") as demo:
    with gr.Column(elem_id="app-shell"):
        with gr.Row(elem_id="app-header"):
            with gr.Column(scale=8):
                gr.HTML(
                    """
                    <div class="app-brand">NEXCEL Wind AI</div>
                    <div class="app-subtitle">Агентная система прогнозирования выработки ВЭС · HackAlem AI</div>
                    """
                )
            with gr.Column(scale=2, min_width=150):
                gr.HTML(
                    """
                    <div class="status-wrap"><div class="agent-status">
                        <span class="agent-status-dot"></span>Agent online
                    </div></div>
                    """
                )

        with gr.Row(elem_id="dashboard-layout", equal_height=True):
            with gr.Column(scale=1, min_width=400):  # noqa: SIM117
                with gr.Group(elem_classes="dashboard-panel"):
                    gr.HTML(CHAT_HEADER_HTML)
                    chatbot = gr.Chatbot(
                        value=INITIAL_HISTORY,
                        label="Чат",
                        show_label=False,
                        elem_id="chatbot",
                        height=425,
                    )
                    with gr.Row(elem_classes="quick-actions"):
                        quick_24 = gr.Button("Прогноз на 24 часа", elem_classes="quick-action")
                        quick_48 = gr.Button("Прогноз на 48 часов", elem_classes="quick-action")
                        quick_model = gr.Button("Состояние модели", elem_classes="quick-action")
                    with gr.Row(elem_id="input-row"):
                        textbox = gr.Textbox(
                            placeholder="Спросите о прогнозе ВЭС...",
                            label="Сообщение",
                            show_label=False,
                            elem_id="chat-input",
                            scale=8,
                        )
                        send_button = gr.Button(
                            "Отправить",
                            variant="primary",
                            elem_id="send-button",
                            scale=1,
                        )

            with gr.Column(scale=1, min_width=400):  # noqa: SIM117
                with gr.Group(elem_classes="dashboard-panel"):
                    gr.HTML(FORECAST_HEADER_HTML)
                    with gr.Row(elem_id="forecast-controls"):
                        issue_date = gr.Textbox(value=DEFAULT_ISSUE_DATE, label="Дата выпуска", scale=2)
                        turbines = gr.Dropdown(
                            list(TURBINE_CHOICES), value="Обе турбины", label="Турбины", scale=2
                        )
                        horizon = gr.Radio(["24", "48"], value="48", label="Горизонт, ч", scale=2)
                    with gr.Row(elem_classes="quick-actions"):
                        forecast_button = gr.Button("Построить прогноз", variant="primary")
                        update_button = gr.Button("Проверить обновление погоды", elem_classes="quick-action")
                    forecast_plot = gr.LinePlot(
                        value=plot_frame(None),
                        x="target_time",
                        y="predicted_power",
                        color="turbine",
                        y_lim=[0, 1],
                        x_title="Время, UTC",
                        y_title="Нормализованная мощность",
                        show_label=False,
                        visible=False,
                        elem_id="forecast-chart",
                    )
                    placeholder = gr.HTML(FORECAST_PLACEHOLDER_HTML)
                    metrics = gr.HTML(metrics_html(None))
                    activity = gr.Markdown(ACTIVITY_PLACEHOLDER, elem_id="agent-activity")

    forecast_state = gr.State(None)
    views = [forecast_plot, placeholder, activity, metrics]

    chat_inputs = [textbox, chatbot, forecast_state]
    chat_outputs = [chatbot, textbox, forecast_state, *views]
    send_button.click(send_message, inputs=chat_inputs, outputs=chat_outputs)
    textbox.submit(send_message, inputs=chat_inputs, outputs=chat_outputs)
    forecast_inputs = [issue_date, turbines, horizon]
    forecast_button.click(build_forecast, inputs=forecast_inputs, outputs=[forecast_state, *views])
    update_button.click(
        partial(build_forecast, update_only=True), inputs=forecast_inputs, outputs=[forecast_state, *views]
    )
    quick_24.click(lambda d: f"Построй прогноз обеих турбин на 24 часа от {d}", inputs=issue_date, outputs=textbox)
    quick_48.click(lambda d: f"Построй прогноз обеих турбин на 48 часов от {d}", inputs=issue_date, outputs=textbox)
    quick_model.click(lambda: "Покажи состояние модели прогноза", outputs=textbox)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", css=CUSTOM_CSS, theme=gr.themes.Base())
