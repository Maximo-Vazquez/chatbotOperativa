TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "acf",
        "description": "Calcula la funcion de autocorrelacion ACF de una serie temporal usando statsmodels.",
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
    "label": "ACF",
    "description": "Autocorrelacion por rezagos",
    "icon": "bx-bar-chart-alt-2",
    "color": "#38bdf8",
}


def _ejecutar_acf(valores: list, lags: int | None = None) -> dict:
    if len(valores) < 3:
        return {"error": "Se necesitan al menos 3 valores para calcular ACF."}

    try:
        from statsmodels.tsa.stattools import acf as statsmodels_acf
    except ImportError:
        return {"error": "Falta instalar statsmodels para usar esta herramienta."}

    max_lags = min(lags if lags is not None else min(20, len(valores) - 1), len(valores) - 1)
    valores_acf = statsmodels_acf(valores, nlags=max_lags, fft=False, missing="drop")

    return {
        "lags": list(range(max_lags + 1)),
        "acf": [round(float(v), 6) for v in valores_acf],
    }


TOOL_FUNCTION = _ejecutar_acf
