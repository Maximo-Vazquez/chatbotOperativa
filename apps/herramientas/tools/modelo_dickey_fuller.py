TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "modelo_dickey_fuller",
        "description": (
            "Aplica la prueba aumentada de Dickey-Fuller para evaluar "
            "estacionariedad en una serie temporal."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "valores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Valores numericos ordenados temporalmente.",
                },
                "significancia": {
                    "type": "number",
                    "minimum": 0.001,
                    "maximum": 0.2,
                    "description": "Nivel de significancia para decidir estacionariedad.",
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


def _ejecutar_modelo_dickey_fuller(valores: list, significancia: float = 0.05) -> dict:
    if len(valores) < 8:
        return {"error": "Se recomiendan al menos 8 valores para aplicar Dickey-Fuller."}

    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return {"error": "Falta instalar statsmodels para usar esta herramienta."}

    estadistico, p_value, used_lag, n_obs, critical_values, icbest = adfuller(valores, autolag="AIC")
    estacionaria = p_value < significancia

    return {
        "estadistico_adf": round(float(estadistico), 6),
        "p_value": round(float(p_value), 6),
        "lags_usados": int(used_lag),
        "n_observaciones": int(n_obs),
        "criterio_aic": round(float(icbest), 6),
        "valores_criticos": {k: round(float(v), 6) for k, v in critical_values.items()},
        "significancia": significancia,
        "es_estacionaria": estacionaria,
        "interpretacion": (
            "Se rechaza la hipotesis nula de raiz unitaria; la serie parece estacionaria."
            if estacionaria
            else "No se rechaza la hipotesis nula de raiz unitaria; la serie no parece estacionaria."
        ),
    }


TOOL_FUNCTION = _ejecutar_modelo_dickey_fuller
