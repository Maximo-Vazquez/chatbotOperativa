"""Constantes de modelos DeepSeek.

Fuente: https://api-docs.deepseek.com/quick_start/pricing
Confirmado en 2026-04: DeepSeek V4 tiene ventana de contexto de 1M tokens
y salida maxima de 384K tokens. Los IDs `deepseek-chat` y `deepseek-reasoner`
son aliases legacy de deepseek-v4-flash en modo non-thinking/thinking.
"""

# IDs de la API (campo `model` en la request).
MODEL_DEEPSEEK_FLASH = "deepseek-v4-flash"
MODEL_DEEPSEEK_PRO = "deepseek-v4-pro"

# Ventana de contexto por modelo, en tokens.
# Doc oficial: 1M tokens (1_048_576) para v4-flash y v4-pro.
CONTEXT_WINDOW_BY_MODEL = {
    MODEL_DEEPSEEK_FLASH: 1_048_576,
    MODEL_DEEPSEEK_PRO: 1_048_576,
    # Aliases legacy todavia aceptados por el endpoint de DeepSeek.
    "deepseek-chat": 1_048_576,
    "deepseek-reasoner": 1_048_576,
}

DEFAULT_CONTEXT_WINDOW = 1_048_576


def get_context_window(model: str) -> int:
    return CONTEXT_WINDOW_BY_MODEL.get(model, DEFAULT_CONTEXT_WINDOW)


# Parametros de inferencia por modelo.
# - non-thinking: respuestas conversacionales acotadas, temperatura baja.
# - thinking: DeepSeek ignora temperature/top_p; max_tokens incluye razonamiento
#   y respuesta final.
MODEL_PARAMS = {
    MODEL_DEEPSEEK_FLASH: {"temperature": 0.4, "max_tokens": 8192, "thinking": False},
    MODEL_DEEPSEEK_PRO: {"temperature": None, "max_tokens": 32768, "thinking": True},
    "deepseek-chat": {"temperature": 0.4, "max_tokens": 8192, "thinking": False},
    "deepseek-reasoner": {"temperature": None, "max_tokens": 32768, "thinking": True},
}

DEFAULT_PARAMS = {"temperature": 0.4, "max_tokens": 8192, "thinking": False}


def get_model_params(model: str) -> dict:
    return MODEL_PARAMS.get(model, DEFAULT_PARAMS)


def is_thinking_model(model: str) -> bool:
    return get_model_params(model).get("thinking", False)
