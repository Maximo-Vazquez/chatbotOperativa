import numpy as np


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "estabilizacion_media",
        "description": (
            "Aplica diferenciacion regular de orden d a una serie temporal "
            "para eliminar la tendencia y estabilizar la media: "
            "w_t = (1-B)^d Y_t. Cada diferencia reduce en 1 la longitud "
            "de la serie."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "valores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Serie original Y_t (o Z_t si ya se aplico log).",
                },
                "orden": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                    "description": "Orden d de la diferenciacion (default 1).",
                },
            },
            "required": ["valores"],
        },
    },
}

TOOL_META = {
    "label": "Estabilizar Media",
    "description": "Diferenciacion regular para eliminar tendencia",
    "icon": "bx-transfer-alt",
    "color": "#22c55e",
}


def _ejecutar_estabilizacion_media(valores: list, orden: int = 1) -> dict:
    n = len(valores)
    if n < orden + 2:
        return {
            "error": (
                f"Se necesitan al menos {orden + 2} valores para diferenciar "
                f"con orden d={orden}."
            )
        }

    arr = np.asarray(valores, dtype=float)
    transformada = np.diff(arr, n=orden)

    return {
        "transformacion": f"diferenciacion regular de orden {orden}",
        "ecuacion": f"w_t = (1-B)^{orden} Y_t",
        "orden": orden,
        "n_original": int(arr.size),
        "n_transformada": int(transformada.size),
        "media_original": round(float(np.mean(arr)), 6),
        "media_transformada": round(float(np.mean(transformada)), 6),
        "varianza_transformada": round(float(np.var(transformada, ddof=1)), 6),
        "valores_transformados": [round(float(v), 6) for v in transformada],
    }


TOOL_FUNCTION = _ejecutar_estabilizacion_media
