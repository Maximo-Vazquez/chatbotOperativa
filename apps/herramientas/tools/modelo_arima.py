import numpy as np


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "modelo_arima",
        "description": (
            "Ajusta un modelo ARIMA(p, d, q) sobre una serie temporal: "
            "p ordenes autorregresivos (PACF), d diferenciaciones (Test ADF), "
            "q ordenes de medias moviles (ACF). Devuelve coeficientes, AIC/BIC, "
            "validacion de residuos como ruido blanco (Ljung-Box) y un "
            "pronostico para los siguientes pasos."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "valores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Serie temporal original Y_t.",
                },
                "p": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Orden autorregresivo (sugerido por la PACF).",
                },
                "d": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 2,
                    "description": "Grado de integracion (diferenciaciones) segun ADF.",
                },
                "q": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Orden de medias moviles (sugerido por la ACF).",
                },
                "pasos_pronostico": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Cantidad de pasos a pronosticar (default 1).",
                },
                "con_constante": {
                    "type": "boolean",
                    "description": (
                        "Si True, incluye constante/drift cuando d>0 "
                        "(util para ARIMA(0,1,0) con drift)."
                    ),
                },
            },
            "required": ["valores", "p", "d", "q"],
        },
    },
}

TOOL_META = {
    "label": "Modelo ARIMA",
    "description": "Ajuste ARIMA(p,d,q) con pronostico y validacion de residuos",
    "icon": "bx-pulse",
    "color": "#ec4899",
}


def _ejecutar_modelo_arima(
    valores: list,
    p: int,
    d: int,
    q: int,
    pasos_pronostico: int = 1,
    con_constante: bool = True,
) -> dict:
    n = len(valores)
    minimo = p + d + q + 3
    if n < minimo:
        return {
            "error": (
                f"Se necesitan al menos {minimo} observaciones para ajustar "
                f"ARIMA({p},{d},{q})."
            )
        }

    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError:
        return {"error": "Falta instalar statsmodels para usar esta herramienta."}

    # statsmodels rechaza terminos de tendencia de orden < d+D.
    # d=0: 'c' (constante) o 'n'. d=1: 't' (drift) o 'n'. d>=2: 'n'.
    if not con_constante:
        trend = "n"
    elif d == 0:
        trend = "c"
    elif d == 1:
        trend = "t"
    else:
        trend = "n"

    modelo = ARIMA(valores, order=(p, d, q), trend=trend)
    ajuste = modelo.fit()

    params = ajuste.params
    nombres_params = list(getattr(params, "index", [])) or [f"param_{i}" for i in range(len(params))]
    coeficientes = {
        str(nombre): round(float(valor), 6)
        for nombre, valor in zip(nombres_params, np.asarray(params))
    }

    residuos = np.asarray(ajuste.resid, dtype=float)
    # Cuando d>0, los primeros d residuos pueden ser inestables; los descartamos para diagnostico.
    residuos_diag = residuos[d:] if d > 0 and len(residuos) > d else residuos
    mse = float(np.mean(residuos_diag ** 2))

    lb_lags = max(1, min(10, len(residuos_diag) // 5))
    lb = acorr_ljungbox(residuos_diag, lags=[lb_lags], return_df=True)
    lb_stat = float(lb["lb_stat"].iloc[0])
    lb_pvalue = float(lb["lb_pvalue"].iloc[0])
    es_ruido_blanco = bool(lb_pvalue > 0.05)

    pronostico = ajuste.forecast(steps=pasos_pronostico)
    pronostico_lista = [round(float(v), 6) for v in np.asarray(pronostico)]

    return {
        "modelo": (
            f"ARIMA({p},{d},{q})"
            + (" con constante" if trend == "c" else "")
            + (" con drift" if trend == "t" else "")
        ),
        "orden": {"p": p, "d": d, "q": q},
        "n_observaciones": n,
        "coeficientes": coeficientes,
        "aic": round(float(ajuste.aic), 6),
        "bic": round(float(ajuste.bic), 6),
        "mse_residuos": round(mse, 6),
        "media_residuos": round(float(np.mean(residuos_diag)), 6),
        "varianza_residuos": round(float(np.var(residuos_diag, ddof=1)), 6),
        "ljung_box": {
            "lags": lb_lags,
            "estadistico": round(lb_stat, 6),
            "p_value": round(lb_pvalue, 6),
            "es_ruido_blanco": es_ruido_blanco,
            "interpretacion": (
                "Los residuos se comportan como ruido blanco (modelo valido)."
                if es_ruido_blanco
                else "Los residuos NO son ruido blanco: el modelo no captura toda la estructura."
            ),
        },
        "pasos_pronostico": pasos_pronostico,
        "pronostico": pronostico_lista,
    }


TOOL_FUNCTION = _ejecutar_modelo_arima
