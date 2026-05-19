import statistics


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
    n = len(valores)
    if n < 4:
        return {"error": "Se necesitan al menos 4 valores para evaluar estacionariedad."}

    media = statistics.mean(valores)
    varianza = statistics.pvariance(valores)
    if varianza == 0:
        return {
            "estadistico_adf": None,
            "p_value": None,
            "lags_usados": 0,
            "n_observaciones": n,
            "criterio_aic": None,
            "valores_criticos": {},
            "significancia": significancia,
            "regresion": regresion,
            "es_estacionaria": True,
            "serie_constante": True,
            "muestra_corta": n < 8,
            "decision": (
                "La serie es constante: media constante, varianza nula y sin "
                "tendencia. Se considera estacionaria en sentido practico/"
                "degenerado; no corresponde aplicar ADF formal."
            ),
        }

    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return {"error": "Falta instalar statsmodels para usar esta herramienta."}

    muestra_corta = n < 8
    try:
        if muestra_corta:
            estadistico, p_value, used_lag, n_obs, critical_values = adfuller(
                valores, maxlag=0, autolag=None, regression=regresion
            )
            icbest = None
        else:
            estadistico, p_value, used_lag, n_obs, critical_values, icbest = adfuller(
                valores, autolag="AIC", regression=regresion
            )
    except Exception:
        pendiente = (
            (valores[-1] - valores[0]) / (n - 1)
            if n > 1
            else 0
        )
        desv = statistics.pstdev(valores)
        umbral_pendiente = max(desv * 0.2, abs(media) * 0.01, 1e-9)
        estacionaria_operativa = abs(pendiente) <= umbral_pendiente
        return {
            "estadistico_adf": None,
            "p_value": None,
            "lags_usados": 0,
            "n_observaciones": n,
            "criterio_aic": None,
            "valores_criticos": {},
            "significancia": significancia,
            "regresion": regresion,
            "es_estacionaria": estacionaria_operativa,
            "muestra_corta": muestra_corta,
            "diagnostico_operativo": True,
            "pendiente_aproximada": round(float(pendiente), 6),
            "decision": (
                "No se pudo aplicar ADF formal de manera estable con esta "
                "muestra corta. Por inspeccion operativa, la serie parece "
                + (
                    "aproximadamente estacionaria: no presenta tendencia clara."
                    if estacionaria_operativa
                    else "no estacionaria: presenta una tendencia clara."
                )
            ),
        }

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
        "criterio_aic": round(float(icbest), 6) if icbest is not None else None,
        "valores_criticos": {k: round(float(v), 6) for k, v in critical_values.items()},
        "significancia": significancia,
        "regresion": regresion,
        "es_estacionaria": estacionaria,
        "muestra_corta": muestra_corta,
        "decision": decision,
    }


TOOL_FUNCTION = _ejecutar_modelo_dickey_fuller
