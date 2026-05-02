TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "descomposicion_visualizacion_serie",
        "description": (
            "Descompone una serie temporal en tendencia, componente estacional "
            "y residuo. Devuelve datos listos para visualizar."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "valores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Valores numericos ordenados temporalmente.",
                },
                "periodo_estacional": {
                    "type": "integer",
                    "minimum": 2,
                    "description": "Cantidad de observaciones por ciclo estacional.",
                },
                "modelo": {
                    "type": "string",
                    "enum": ["aditivo", "multiplicativo"],
                    "description": "Tipo de descomposicion.",
                },
            },
            "required": ["valores", "periodo_estacional"],
        },
    },
}

TOOL_META = {
    "label": "Descomposicion",
    "description": "Tendencia, estacionalidad y residuo para graficar",
    "icon": "bx-analyse",
    "color": "#14b8a6",
}


def _round_or_none(value):
    if value is None:
        return None
    try:
        if value != value:
            return None
    except TypeError:
        return None
    return round(float(value), 6)


def _ejecutar_descomposicion_visualizacion_serie(
    valores: list,
    periodo_estacional: int,
    modelo: str = "aditivo",
) -> dict:
    if len(valores) < periodo_estacional * 2:
        return {
            "error": (
                "Se necesitan al menos dos ciclos completos: "
                f"{periodo_estacional * 2} observaciones."
            )
        }

    if modelo == "multiplicativo" and any(v <= 0 for v in valores):
        return {"error": "El modelo multiplicativo requiere valores mayores que cero."}

    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        return {"error": "Falta instalar statsmodels para usar esta herramienta."}

    stats_model = "multiplicative" if modelo == "multiplicativo" else "additive"
    result = seasonal_decompose(valores, model=stats_model, period=periodo_estacional)

    return {
        "modelo": modelo,
        "periodo_estacional": periodo_estacional,
        "serie": [_round_or_none(v) for v in valores],
        "tendencia": [_round_or_none(v) for v in result.trend],
        "estacionalidad": [_round_or_none(v) for v in result.seasonal],
        "residuo": [_round_or_none(v) for v in result.resid],
    }


TOOL_FUNCTION = _ejecutar_descomposicion_visualizacion_serie
