"""
Definición y ejecución de herramientas disponibles para el chatbot.

Para agregar una herramienta nueva:
1. Definir su schema en TOOL_DEFINITIONS (formato OpenAI function calling).
2. Agregar su implementación en TOOL_REGISTRY con el mismo nombre.
"""

import statistics
import math

# ─────────────────────────────────────────────
# DEFINICIONES (schemas que se envían a la API)
# ─────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "analizar_serie_temporal",
            "description": (
                "Analiza estadísticamente una serie temporal numérica. "
                "Calcula media, mediana, desvío estándar, mínimo, máximo, "
                "coeficiente de variación y detecta tendencia simple. "
                "Úsala cuando el usuario quiera analizar o explorar una lista de valores numéricos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "valores": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Lista de valores numéricos de la serie temporal (al menos 3).",
                    },
                    "nombre_serie": {
                        "type": "string",
                        "description": "Nombre descriptivo de la serie (ej: 'ventas mensuales', 'temperatura diaria').",
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
    },
]

# Mapeo rápido para el frontend: nombre → etiqueta y descripción corta
TOOL_META = {
    "analizar_serie_temporal": {
        "label": "Analizador de Serie",
        "description": "Estadísticas y tendencia de una lista de valores",
        "icon": "bx-line-chart",
        "color": "#818cf8",
    },
}


# ─────────────────────────────────────────────
# IMPLEMENTACIONES
# ─────────────────────────────────────────────

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

    # Tendencia simple: regresión lineal mínimos cuadrados
    x_mean = (n - 1) / 2
    numerador = sum((i - x_mean) * (v - media) for i, v in enumerate(valores))
    denominador = sum((i - x_mean) ** 2 for i in range(n))
    pendiente = numerador / denominador if denominador != 0 else 0

    if abs(pendiente) < desv * 0.05:
        tendencia = "estable (sin tendencia clara)"
    elif pendiente > 0:
        tendencia = f"creciente (pendiente ≈ {pendiente:.4f} por período)"
    else:
        tendencia = f"decreciente (pendiente ≈ {pendiente:.4f} por período)"

    # Detección básica de outliers (más de 2 desvíos)
    outliers = [round(v, 4) for v in valores if abs(v - media) > 2 * desv]

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


# ─────────────────────────────────────────────
# DISPATCHER
# ─────────────────────────────────────────────

TOOL_REGISTRY = {
    "analizar_serie_temporal": _ejecutar_analizar_serie_temporal,
}


def ejecutar_herramienta(nombre: str, argumentos: dict) -> dict:
    """Ejecuta la herramienta con el nombre dado y devuelve el resultado."""
    fn = TOOL_REGISTRY.get(nombre)
    if fn is None:
        return {"error": f"Herramienta '{nombre}' no encontrada."}
    try:
        return fn(**argumentos)
    except TypeError as e:
        return {"error": f"Argumentos inválidos para '{nombre}': {e}"}
    except Exception as e:
        return {"error": f"Error al ejecutar '{nombre}': {e}"}
