import numpy as np


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "modelo_ar",
        "description": (
            "Ajusta un modelo Autorregresivo AR(p) sobre una serie "
            "estacionaria. Estima la constante c y los coeficientes phi_1..phi_p, "
            "calcula MSE/AIC/BIC, valida los residuos como ruido blanco "
            "(Ljung-Box) y genera un pronostico para los siguientes pasos."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "valores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Serie temporal estacionaria.",
                },
                "p": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Orden autorregresivo (sugerido por la PACF).",
                },
                "pasos_pronostico": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Cantidad de pasos a pronosticar (default 1).",
                },
            },
            "required": ["valores", "p"],
        },
    },
}

TOOL_META = {
    "label": "Modelo AR",
    "description": "Ajuste de un modelo Autorregresivo AR(p)",
    "icon": "bx-trending-up",
    "color": "#0ea5e9",
}


def _ejecutar_modelo_ar(
    valores: list,
    p: int,
    pasos_pronostico: int = 1,
) -> dict:
    n = len(valores)
    if n < p + 3:
        return {"error": f"Se necesitan al menos {p + 3} observaciones para ajustar AR({p})."}

    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError:
        return {"error": "Falta instalar statsmodels para usar esta herramienta."}

    modelo = ARIMA(valores, order=(p, 0, 0))
    ajuste = modelo.fit()

    params = ajuste.params
    nombres_params = list(getattr(params, "index", [])) or [f"param_{i}" for i in range(len(params))]
    coeficientes = {
        str(nombre): round(float(valor), 6)
        for nombre, valor in zip(nombres_params, np.asarray(params))
    }

    residuos = np.asarray(ajuste.resid, dtype=float)
    mse = float(np.mean(residuos ** 2))

    # Test de Ljung-Box (ruido blanco). lags acotado por la longitud de los residuos.
    lb_lags = max(1, min(10, len(residuos) // 5))
    lb = acorr_ljungbox(residuos, lags=[lb_lags], return_df=True)
    lb_stat = float(lb["lb_stat"].iloc[0])
    lb_pvalue = float(lb["lb_pvalue"].iloc[0])
    es_ruido_blanco = bool(lb_pvalue > 0.05)

    pronostico = ajuste.forecast(steps=pasos_pronostico)
    pronostico_lista = [round(float(v), 6) for v in np.asarray(pronostico)]

    return {
        "modelo": f"AR({p})",
        "orden_p": p,
        "n_observaciones": n,
        "coeficientes": coeficientes,
        "aic": round(float(ajuste.aic), 6),
        "bic": round(float(ajuste.bic), 6),
        "mse_residuos": round(mse, 6),
        "media_residuos": round(float(np.mean(residuos)), 6),
        "varianza_residuos": round(float(np.var(residuos, ddof=1)), 6),
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


TOOL_FUNCTION = _ejecutar_modelo_ar
