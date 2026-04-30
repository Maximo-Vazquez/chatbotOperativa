"""Constantes de modelos DeepSeek.

Fuente: https://api-docs.deepseek.com/quick_start/pricing
Confirmado en 2026-04: ambos modelos V4 tienen ventana de contexto de 1M tokens
y salida máxima de 384K tokens. Los IDs `deepseek-chat` y `deepseek-reasoner`
quedan deprecados (mapeados a v4-flash y v4-pro respectivamente, retiro 2026-07-24).
"""

# IDs de la API (campo `model` en la request).
MODEL_DEEPSEEK_FLASH = "deepseek-v4-flash"
MODEL_DEEPSEEK_PRO = "deepseek-v4-pro"

# Ventana de contexto por modelo, en tokens.
# Doc oficial: 1M tokens (1_048_576) para v4-flash y v4-pro.
CONTEXT_WINDOW_BY_MODEL = {
    MODEL_DEEPSEEK_FLASH: 1_048_576,
    MODEL_DEEPSEEK_PRO: 1_048_576,
    # Aliases legacy todavía aceptados por el endpoint de DeepSeek.
    "deepseek-chat": 1_048_576,
    "deepseek-reasoner": 1_048_576,
}

DEFAULT_CONTEXT_WINDOW = 1_048_576


def get_context_window(model: str) -> int:
    return CONTEXT_WINDOW_BY_MODEL.get(model, DEFAULT_CONTEXT_WINDOW)


# Parámetros de inferencia por modelo.
# - flash (non-thinking): respuestas conversacionales acotadas, temperatura baja.
# - pro (thinking): necesita más tokens de salida porque incluye razonamiento;
#   DeepSeek recomienda temperatura ~1.0 para el modo thinking.
MODEL_PARAMS = {
    MODEL_DEEPSEEK_FLASH:    {"temperature": 0.6, "max_tokens": 4096,  "thinking": False},
    MODEL_DEEPSEEK_PRO:      {"temperature": 1.0, "max_tokens": 16384, "thinking": True},
    "deepseek-chat":         {"temperature": 0.6, "max_tokens": 4096,  "thinking": False},
    "deepseek-reasoner":     {"temperature": 1.0, "max_tokens": 16384, "thinking": True},
}

DEFAULT_PARAMS = {"temperature": 0.6, "max_tokens": 4096, "thinking": False}


def get_model_params(model: str) -> dict:
    return MODEL_PARAMS.get(model, DEFAULT_PARAMS)


def is_thinking_model(model: str) -> bool:
    return get_model_params(model).get("thinking", False)
