TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "pacf",
        "description": "Calcula la funcion de autocorrelacion parcial PACF usando statsmodels.",
        "parameters": {
            "type": "object",
            "properties": {
                "valores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Valores numericos ordenados temporalmente.",
                },
                "lags": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Cantidad maxima de rezagos a calcular.",
                },
            },
            "required": ["valores"],
        },
    },
}

TOOL_META = {
    "label": "PACF",
    "description": "Autocorrelacion parcial por rezagos",
    "icon": "bx-chart",
    "color": "#f59e0b",
}


def _ejecutar_pacf(valores: list, lags: int | None = None) -> dict:
    if len(valores) < 4:
        return {"error": "Se necesitan al menos 4 valores para calcular PACF."}

    try:
        from statsmodels.tsa.stattools import pacf as statsmodels_pacf
    except ImportError:
        return {"error": "Falta instalar statsmodels para usar esta herramienta."}

    max_allowed_lags = max(1, (len(valores) // 2) - 1)
    max_lags = min(lags if lags is not None else min(20, max_allowed_lags), max_allowed_lags)
    valores_pacf = statsmodels_pacf(valores, nlags=max_lags, method="ywm")

    return {
        "lags": list(range(max_lags + 1)),
        "pacf": [round(float(v), 6) for v in valores_pacf],
    }


TOOL_FUNCTION = _ejecutar_pacf
