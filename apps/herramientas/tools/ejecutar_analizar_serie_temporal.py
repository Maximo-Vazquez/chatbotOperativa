import statistics


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "analizar_serie_temporal",
        "description": (
            "Analiza estadisticamente una serie temporal numerica. Calcula media, "
            "mediana, desvio estandar, minimo, maximo, coeficiente de variacion "
            "y detecta una tendencia simple."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "valores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Lista de valores numericos de la serie temporal.",
                },
                "nombre_serie": {
                    "type": "string",
                    "description": "Nombre descriptivo de la serie.",
                },
                "periodo": {
                    "type": "string",
                    "enum": ["diario", "semanal", "mensual", "trimestral", "anual", "otro"],
                    "description": "Frecuencia de los datos.",
                },
            },
            "required": ["valores", "nombre_serie"],
        },
    },
}

TOOL_META = {
    "label": "Analizador de Serie",
    "description": "Estadisticas y tendencia de una lista de valores",
    "icon": "bx-line-chart",
    "color": "#818cf8",
}


def _ejecutar_analizar_serie_temporal(valores: list, nombre_serie: str, periodo: str = "otro") -> dict:
    n = len(valores)
    if n < 3:
        return {"error": "Se necesitan al menos 3 valores para analizar la serie."}

    media = statistics.mean(valores)
    mediana = statistics.median(valores)
    desv = statistics.stdev(valores) if n > 1 else 0
    minimo = min(valores)
    maximo = max(valores)
    rango = maximo - minimo
    cv = (desv / media * 100) if media != 0 else None

    x_mean = (n - 1) / 2
    numerador = sum((i - x_mean) * (v - media) for i, v in enumerate(valores))
    denominador = sum((i - x_mean) ** 2 for i in range(n))
    pendiente = numerador / denominador if denominador != 0 else 0

    if abs(pendiente) < desv * 0.05:
        tendencia = "estable (sin tendencia clara)"
    elif pendiente > 0:
        tendencia = f"creciente (pendiente aprox. {pendiente:.4f} por periodo)"
    else:
        tendencia = f"decreciente (pendiente aprox. {pendiente:.4f} por periodo)"

    outliers = [round(v, 4) for v in valores if desv and abs(v - media) > 2 * desv]

    return {
        "nombre_serie": nombre_serie,
        "periodo": periodo,
        "n_observaciones": n,
        "media": round(media, 4),
        "mediana": round(mediana, 4),
        "desv_estandar": round(desv, 4),
        "minimo": round(minimo, 4),
        "maximo": round(maximo, 4),
        "rango": round(rango, 4),
        "coef_variacion_pct": round(cv, 2) if cv is not None else None,
        "tendencia": tendencia,
        "outliers_potenciales": outliers,
    }


TOOL_FUNCTION = _ejecutar_analizar_serie_temporal
