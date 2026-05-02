import math


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "acf",
        "description": (
            "Calcula la Funcion de Autocorrelacion (ACF) sobre una serie "
            "preferentemente estacionaria. Devuelve los coeficientes rho_k, "
            "las bandas de confianza +-1.96/sqrt(n) y los rezagos "
            "estadisticamente significativos (utiles para identificar el "
            "orden q de un modelo MA/ARIMA)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "valores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Serie temporal (idealmente estacionaria).",
                },
                "lags": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Numero maximo de rezagos k a evaluar. "
                        "Default: min(20, n//4)."
                    ),
                },
                "nivel_confianza": {
                    "type": "number",
                    "minimum": 0.8,
                    "maximum": 0.999,
                    "description": "Nivel de confianza para las bandas (default 0.95).",
                },
            },
            "required": ["valores"],
        },
    },
}

TOOL_META = {
    "label": "ACF",
    "description": "Autocorrelacion simple con bandas de confianza",
    "icon": "bx-bar-chart-alt-2",
    "color": "#38bdf8",
}


_Z_SCORES = {0.90: 1.6449, 0.95: 1.96, 0.99: 2.5758}


def _z_score(nivel):
    closest = min(_Z_SCORES.keys(), key=lambda k: abs(k - nivel))
    return _Z_SCORES[closest]


def _ejecutar_acf(
    valores: list,
    lags: int | None = None,
    nivel_confianza: float = 0.95,
) -> dict:
    n = len(valores)
    if n < 3:
        return {"error": "Se necesitan al menos 3 valores para calcular ACF."}

    try:
        from statsmodels.tsa.stattools import acf as statsmodels_acf
    except ImportError:
        return {"error": "Falta instalar statsmodels para usar esta herramienta."}

    max_lags = min(
        lags if lags is not None else min(20, max(1, n // 4)),
        n - 1,
    )
    valores_acf = statsmodels_acf(valores, nlags=max_lags, fft=False, missing="drop")

    z = _z_score(nivel_confianza)
    banda = z / math.sqrt(n)

    coeficientes = [round(float(v), 6) for v in valores_acf]
    rezagos_significativos = [
        {"lag": k, "rho": coeficientes[k]}
        for k in range(1, max_lags + 1)
        if abs(coeficientes[k]) > banda
    ]

    # Sugerencia de q: ultimo lag significativo bajo, antes del primer "corte"
    q_sugerido = 0
    for k in range(1, max_lags + 1):
        if abs(coeficientes[k]) > banda:
            q_sugerido = k
        else:
            break

    return {
        "n_observaciones": n,
        "lags": list(range(max_lags + 1)),
        "acf": coeficientes,
        "nivel_confianza": nivel_confianza,
        "banda_confianza": round(banda, 6),
        "rezagos_significativos": rezagos_significativos,
        "q_sugerido": q_sugerido,
        "interpretacion": (
            f"Bandas +-{round(banda, 4)}. "
            f"{len(rezagos_significativos)} rezago(s) significativo(s). "
            f"Sugerencia de orden MA(q)={q_sugerido} (corte de la ACF)."
        ),
    }


TOOL_FUNCTION = _ejecutar_acf
