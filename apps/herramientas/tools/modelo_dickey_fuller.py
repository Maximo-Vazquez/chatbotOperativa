TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "modelo_dickey_fuller",
        "description": (
            "Aplica la prueba aumentada de Dickey-Fuller (ADF) para determinar "
            "si una serie temporal es estacionaria. H0: la serie tiene raiz "
            "unitaria (no estacionaria). Si p-valor <= significancia, se "
            "rechaza H0 y la serie se considera estacionaria."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "valores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Serie temporal Y_t a evaluar.",
                },
                "significancia": {
                    "type": "number",
                    "minimum": 0.001,
                    "maximum": 0.2,
                    "description": "Nivel de significancia alfa (default 0.05).",
                },
                "regresion": {
                    "type": "string",
                    "enum": ["c", "ct", "ctt", "n"],
                    "description": (
                        "Componentes deterministas: 'c' constante (default), "
                        "'ct' constante+tendencia, 'ctt' con tendencia "
                        "cuadratica, 'n' sin constante."
                    ),
                },
            },
            "required": ["valores"],
        },
    },
}

TOOL_META = {
    "label": "Dickey-Fuller",
    "description": "Prueba ADF de estacionariedad",
    "icon": "bx-check-shield",
    "color": "#a855f7",
}


def _ejecutar_modelo_dickey_fuller(
    valores: list,
    significancia: float = 0.05,
    regresion: str = "c",
) -> dict:
    if len(valores) < 8:
        return {"error": "Se recomiendan al menos 8 valores para aplicar Dickey-Fuller."}

    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return {"error": "Falta instalar statsmodels para usar esta herramienta."}

    estadistico, p_value, used_lag, n_obs, critical_values, icbest = adfuller(
        valores, autolag="AIC", regression=regresion
    )
    p_value = float(p_value)
    estacionaria = bool(p_value < significancia)

    if estacionaria:
        decision = (
            f"Se rechaza H0 (p-valor={p_value:.4f} < {significancia}). "
            "La serie es estacionaria; se puede pasar al analisis ACF/PACF."
        )
    else:
        decision = (
            f"No se rechaza H0 (p-valor={p_value:.4f} >= {significancia}). "
            "La serie no es estacionaria; se recomienda aplicar diferenciacion "
            "y/o transformacion logaritmica antes de modelar."
        )

    return {
        "estadistico_adf": round(float(estadistico), 6),
        "p_value": round(float(p_value), 6),
        "lags_usados": int(used_lag),
        "n_observaciones": int(n_obs),
        "criterio_aic": round(float(icbest), 6),
        "valores_criticos": {k: round(float(v), 6) for k, v in critical_values.items()},
        "significancia": significancia,
        "regresion": regresion,
        "es_estacionaria": estacionaria,
        "decision": decision,
    }


TOOL_FUNCTION = _ejecutar_modelo_dickey_fuller
