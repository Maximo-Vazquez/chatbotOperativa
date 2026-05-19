import math
import statistics


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "pacf",
        "description": (
            "Calcula la Funcion de Autocorrelacion Parcial (PACF) sobre una "
            "serie preferentemente estacionaria. Devuelve los coeficientes "
            "phi_kk, las bandas de confianza +-1.96/sqrt(n) y los rezagos "
            "estadisticamente significativos (utiles para identificar el "
            "orden p de un modelo AR/ARIMA)."
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
    "label": "PACF",
    "description": "Autocorrelacion parcial con bandas de confianza",
    "icon": "bx-chart",
    "color": "#f59e0b",
}


_Z_SCORES = {0.90: 1.6449, 0.95: 1.96, 0.99: 2.5758}


def _z_score(nivel):
    closest = min(_Z_SCORES.keys(), key=lambda k: abs(k - nivel))
    return _Z_SCORES[closest]


def _ejecutar_pacf(
    valores: list,
    lags: int | None = None,
    nivel_confianza: float = 0.95,
) -> dict:
    n = len(valores)
    if n < 4:
        return {"error": "Se necesitan al menos 4 valores para calcular PACF."}

    varianza = statistics.pvariance(valores)
    if varianza == 0:
        return {
            "n_observaciones": n,
            "lags": [0],
            "pacf": [1.0],
            "nivel_confianza": nivel_confianza,
            "banda_confianza": round(_z_score(nivel_confianza) / math.sqrt(n), 6),
            "rezagos_significativos": [],
            "p_sugerido": 0,
            "serie_constante": True,
            "interpretacion": (
                "La serie es constante y tiene varianza nula. La PACF no aporta "
                "estructura autorregresiva identificable; no se sugiere AR(p)."
            ),
        }

    try:
        from statsmodels.tsa.stattools import pacf as statsmodels_pacf
    except ImportError:
        return {"error": "Falta instalar statsmodels para usar esta herramienta."}

    max_allowed = max(1, (n // 2) - 1)
    max_lags = min(
        lags if lags is not None else min(20, max_allowed),
        max_allowed,
    )
    valores_pacf = statsmodels_pacf(valores, nlags=max_lags, method="ywm")

    z = _z_score(nivel_confianza)
    banda = z / math.sqrt(n)

    coeficientes = [round(float(v), 6) for v in valores_pacf]
    rezagos_significativos = [
        {"lag": k, "phi": coeficientes[k]}
        for k in range(1, max_lags + 1)
        if abs(coeficientes[k]) > banda
    ]

    # Sugerencia de p: ultimo lag significativo en racha desde k=1
    p_sugerido = 0
    for k in range(1, max_lags + 1):
        if abs(coeficientes[k]) > banda:
            p_sugerido = k
        else:
            break

    return {
        "n_observaciones": n,
        "lags": list(range(max_lags + 1)),
        "pacf": coeficientes,
        "nivel_confianza": nivel_confianza,
        "banda_confianza": round(banda, 6),
        "rezagos_significativos": rezagos_significativos,
        "p_sugerido": p_sugerido,
        "interpretacion": (
            f"Bandas +-{round(banda, 4)}. "
            f"{len(rezagos_significativos)} rezago(s) significativo(s). "
            f"Sugerencia de orden AR(p)={p_sugerido} (corte de la PACF)."
        ),
    }


TOOL_FUNCTION = _ejecutar_pacf
