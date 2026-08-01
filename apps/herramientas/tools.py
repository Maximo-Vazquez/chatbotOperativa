"""
Carga y ejecución de herramientas disponibles para el chatbot.

Cada archivo Python dentro de ``apps/herramientas/tools/`` representa una
herramienta y debe exponer:
- TOOL_DEFINITION: esquema para function calling de OpenAI.
- TOOL_META: metadatos para el frontend.
- TOOL_FUNCTION: función ejecutable de la herramienta.
"""

import importlib.util
import math
from pathlib import Path


CARPETA_HERRAMIENTAS = Path(__file__).with_name("tools")


def _cargar_modulos_de_herramientas():
    modulos = []
    if not CARPETA_HERRAMIENTAS.exists():
        return modulos

    for ruta in sorted(CARPETA_HERRAMIENTAS.glob("*.py")):
        if ruta.name.startswith("_"):
            continue

        nombre_modulo = f"apps.herramientas.herramientas_dinamicas.{ruta.stem}"
        especificacion = importlib.util.spec_from_file_location(nombre_modulo, ruta)
        if especificacion is None or especificacion.loader is None:
            continue

        modulo = importlib.util.module_from_spec(especificacion)
        especificacion.loader.exec_module(modulo)
        modulos.append(modulo)

    return modulos


_MODULOS_HERRAMIENTAS = _cargar_modulos_de_herramientas()

TOOL_DEFINITIONS = [
    modulo.TOOL_DEFINITION
    for modulo in _MODULOS_HERRAMIENTAS
    if hasattr(modulo, "TOOL_DEFINITION")
]

TOOL_META = {
    modulo.TOOL_DEFINITION["function"]["name"]: modulo.TOOL_META
    for modulo in _MODULOS_HERRAMIENTAS
    if hasattr(modulo, "TOOL_DEFINITION") and hasattr(modulo, "TOOL_META")
}

TOOL_REGISTRY = {
    modulo.TOOL_DEFINITION["function"]["name"]: modulo.TOOL_FUNCTION
    for modulo in _MODULOS_HERRAMIENTAS
    if hasattr(modulo, "TOOL_DEFINITION") and hasattr(modulo, "TOOL_FUNCTION")
}


def _to_json_safe(value):
    """Convierte recursivamente tipos numpy/pandas a tipos nativos JSON serializables."""
    # bool primero: en Python bool es subclase de int, hay que chequearlo antes
    if isinstance(value, bool):
        return bool(value)
    if value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # numpy/pandas escalares exponen .item()
    if hasattr(value, "item") and not isinstance(value, (list, tuple, dict)):
        try:
            return _to_json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]
    # numpy arrays
    if hasattr(value, "tolist"):
        try:
            return _to_json_safe(value.tolist())
        except Exception:
            pass
    return str(value)


def ejecutar_herramienta(nombre: str, argumentos: dict) -> dict:
    """Ejecuta la herramienta con el nombre dado y devuelve el resultado."""
    fn = TOOL_REGISTRY.get(nombre)
    if fn is None:
        return {"error": f"Herramienta '{nombre}' no encontrada."}
    try:
        resultado = fn(**argumentos)
    except TypeError as e:
        return {"error": f"Argumentos inválidos para '{nombre}': {e}"}
    except Exception as e:
        return {"error": f"Error al ejecutar '{nombre}': {e}"}
    return _to_json_safe(resultado)
