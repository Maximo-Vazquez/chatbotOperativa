import statistics

import numpy as np


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "descomposicion_visualizacion_serie",
        "description": (
            "Analiza una serie temporal: calcula estadisticos descriptivos "
            "(media, varianza, desvio, min, max, rango, cuartiles, coeficiente "
            "de variacion) y descompone la serie en tendencia, estacionalidad "
            "y residuo (modelo aditivo o multiplicativo)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "valores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Valores numericos Y_t ordenados temporalmente.",
                },
                "frecuencia": {
                    "type": "integer",
                    "minimum": 2,
                    "description": (
                        "Periodicidad m de la serie (m=12 mensual, m=4 trimestral, "
                        "m=7 semanal). Define el ciclo estacional."
                    ),
                },
                "modelo": {
                    "type": "string",
                    "enum": ["aditivo", "multiplicativo"],
                    "description": (
                        "Aditivo si la amplitud estacional es constante; "
                        "multiplicativo si crece con el nivel de la serie."
                    ),
                },
                "nombre_serie": {
                    "type": "string",
                    "description": "Nombre descriptivo de la serie (opcional).",
                },
            },
            "required": ["valores", "frecuencia"],
        },
    },
}

TOOL_META = {
    "label": "Descomposicion y Estadisticos",
    "description": "Estadisticos descriptivos + tendencia, estacionalidad y residuo",
    "icon": "bx-analyse",
    "color": "#14b8a6",
}


def _round_or_none(value):
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except TypeError:
        return None
    return round(float(value), 6)


def _quantiles_safe(valores):
    n = len(valores)
    if n < 2:
        return None, None, None
    ordenados = sorted(valores)
    if n >= 4:
        q1, q2, q3 = statistics.quantiles(valores, n=4, method="inclusive")
    else:
        q2 = statistics.median(ordenados)
        q1 = ordenados[0]
        q3 = ordenados[-1]
    return q1, q2, q3


def _ejecutar_descomposicion_visualizacion_serie(
    valores: list,
    frecuencia: int,
    modelo: str = "aditivo",
    nombre_serie: str = "serie",
) -> dict:
    n = len(valores)
    if n < 3:
        return {"error": "Se necesitan al menos 3 valores para analizar la serie."}

    # ── Estadisticos descriptivos ─────────────────────────────────
    media = statistics.mean(valores)
    mediana = statistics.median(valores)
    varianza = statistics.variance(valores) if n > 1 else 0.0
    desv = statistics.stdev(valores) if n > 1 else 0.0
    minimo = min(valores)
    maximo = max(valores)
    rango = maximo - minimo
    cv = (desv / media * 100) if media != 0 else None
    q1, q2, q3 = _quantiles_safe(valores)

    # Tendencia simple por regresion lineal (pendiente via numpy)
    pendiente = float(np.polyfit(np.arange(n), valores, 1)[0])
    if abs(pendiente) < (desv * 0.05 if desv else 1e-9):
        tendencia_txt = "estable (sin tendencia clara)"
    elif pendiente > 0:
        tendencia_txt = f"creciente (pendiente {pendiente:.4f} por periodo)"
    else:
        tendencia_txt = f"decreciente (pendiente {pendiente:.4f} por periodo)"

    outliers = (
        [round(v, 4) for v in valores if desv and abs(v - media) > 2 * desv]
        if desv
        else []
    )

    estadisticos = {
        "n_observaciones": n,
        "media": round(media, 6),
        "mediana": round(mediana, 6),
        "varianza": round(varianza, 6),
        "desv_estandar": round(desv, 6),
        "minimo": round(minimo, 6),
        "maximo": round(maximo, 6),
        "rango": round(rango, 6),
        "Q1": _round_or_none(q1),
        "Q2": _round_or_none(q2),
        "Q3": _round_or_none(q3),
        "coef_variacion_pct": round(cv, 2) if cv is not None else None,
        "tendencia_lineal": tendencia_txt,
        "outliers_potenciales": outliers,
    }

    # ── Descomposicion estocastica ────────────────────────────────
    descomposicion = None
    advertencia_descomp = None

    if n < frecuencia * 2:
        advertencia_descomp = (
            f"Se requieren al menos {frecuencia * 2} observaciones (2 ciclos) "
            "para descomponer la serie. Se devuelven solo los estadisticos."
        )
    elif modelo == "multiplicativo" and any(v <= 0 for v in valores):
        advertencia_descomp = (
            "El modelo multiplicativo requiere valores > 0. "
            "Se devuelven solo los estadisticos."
        )
    else:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
        except ImportError:
            advertencia_descomp = "Falta statsmodels: no se pudo descomponer la serie."
        else:
            stats_model = "multiplicative" if modelo == "multiplicativo" else "additive"
            result = seasonal_decompose(valores, model=stats_model, period=frecuencia)
            descomposicion = {
                "modelo": modelo,
                "tendencia": [_round_or_none(v) for v in result.trend],
                "estacionalidad": [_round_or_none(v) for v in result.seasonal],
                "residuo": [_round_or_none(v) for v in result.resid],
            }

    salida = {
        "nombre_serie": nombre_serie,
        "frecuencia": frecuencia,
        "serie": [_round_or_none(v) for v in valores],
        "estadisticos": estadisticos,
        "descomposicion": descomposicion,
    }
    if advertencia_descomp:
        salida["advertencia_descomposicion"] = advertencia_descomp
    return salida


TOOL_FUNCTION = _ejecutar_descomposicion_visualizacion_serie
