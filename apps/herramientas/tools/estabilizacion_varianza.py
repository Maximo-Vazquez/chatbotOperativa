import numpy as np


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "estabilizacion_varianza",
        "description": (
            "Aplica una transformacion logaritmica natural z_t = ln(y_t) "
            "para estabilizar la varianza de una serie temporal cuando la "
            "amplitud de las oscilaciones crece con el nivel "
            "(heterocedasticidad). Requiere valores estrictamente positivos."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "valores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Serie original Y_t con valores > 0.",
                },
            },
            "required": ["valores"],
        },
    },
}

TOOL_META = {
    "label": "Estabilizar Varianza",
    "description": "Transformacion logaritmica para estabilizar la dispersion",
    "icon": "bx-log-in",
    "color": "#22c55e",
}


def _ejecutar_estabilizacion_varianza(valores: list) -> dict:
    if len(valores) < 3:
        return {"error": "Se necesitan al menos 3 valores para aplicar la transformacion."}

    arr = np.asarray(valores, dtype=float)
    if np.any(arr <= 0):
        return {"error": "La transformacion logaritmica requiere valores estrictamente > 0."}

    transformada = np.log(arr)

    return {
        "transformacion": "logaritmo natural (ln)",
        "ecuacion": "z_t = ln(y_t)",
        "n_original": int(arr.size),
        "n_transformada": int(transformada.size),
        "varianza_original": round(float(np.var(arr, ddof=1)), 6),
        "varianza_transformada": round(float(np.var(transformada, ddof=1)), 6),
        "valores_transformados": [round(float(v), 6) for v in transformada],
    }


TOOL_FUNCTION = _ejecutar_estabilizacion_varianza
