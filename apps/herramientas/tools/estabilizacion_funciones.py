TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "estabilizacion_funciones",
        "description": (
            "Aplica transformaciones basicas para estabilizar una serie temporal: "
            "diferenciacion, logaritmo, raiz cuadrada o cambio porcentual."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "valores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Valores numericos ordenados temporalmente.",
                },
                "metodo": {
                    "type": "string",
                    "enum": ["diferenciacion", "log", "sqrt", "pct_change"],
                    "description": "Transformacion a aplicar.",
                },
                "orden": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                    "description": "Orden de diferenciacion si metodo=diferenciacion.",
                },
            },
            "required": ["valores", "metodo"],
        },
    },
}

TOOL_META = {
    "label": "Estabilizacion",
    "description": "Transformaciones para estabilizar la serie",
    "icon": "bx-transfer-alt",
    "color": "#22c55e",
}


def _diff(values):
    return [values[i] - values[i - 1] for i in range(1, len(values))]


def _ejecutar_estabilizacion_funciones(valores: list, metodo: str, orden: int = 1) -> dict:
    if len(valores) < 3:
        return {"error": "Se necesitan al menos 3 valores para estabilizar la serie."}

    if metodo == "diferenciacion":
        transformada = [float(v) for v in valores]
        for _ in range(orden):
            transformada = _diff(transformada)
        nombre = f"diferenciacion orden {orden}"
    elif metodo == "log":
        if any(v <= 0 for v in valores):
            return {"error": "La transformacion log requiere valores mayores que cero."}
        import math

        transformada = [math.log(v) for v in valores]
        nombre = "logaritmo natural"
    elif metodo == "sqrt":
        if any(v < 0 for v in valores):
            return {"error": "La transformacion sqrt requiere valores no negativos."}
        import math

        transformada = [math.sqrt(v) for v in valores]
        nombre = "raiz cuadrada"
    elif metodo == "pct_change":
        if any(valores[i - 1] == 0 for i in range(1, len(valores))):
            return {"error": "pct_change no puede dividir por valores previos iguales a cero."}
        transformada = [
            (valores[i] - valores[i - 1]) / valores[i - 1]
            for i in range(1, len(valores))
        ]
        nombre = "cambio porcentual"
    else:
        return {"error": f"Metodo '{metodo}' no soportado."}

    return {
        "metodo": metodo,
        "transformacion": nombre,
        "n_original": len(valores),
        "n_transformada": len(transformada),
        "valores_transformados": [round(float(v), 6) for v in transformada],
    }


TOOL_FUNCTION = _ejecutar_estabilizacion_funciones
