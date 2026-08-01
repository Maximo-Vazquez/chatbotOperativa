"""Fechas y frecuencia temporal opcionales del nucleo de pronostico.

Todo lo relacionado con ``pandas.DatetimeIndex`` (parseo, alias de
frecuencia, inferencia, deteccion de periodos faltantes y generacion de
fechas futuras) vive aca. ``engine.py`` sigue siendo enteramente numerico
-no conoce fechas-; este modulo valida las fechas por separado y, al final,
las traduce a ``fechas_pronostico`` ya formateadas en ISO para la respuesta
JSON de ``tools/modelo_arima.py``.
"""

from typing import Optional, Sequence

import pandas as pd

from .exceptions import (
    DateLengthMismatchError,
    DateValidationError,
    DuplicateDatesError,
    FrequencyValidationError,
    InconsistentFrequencyError,
    UnsortedDatesError,
)

# Alias pedagogicos -> codigo de frecuencia de pandas. Centralizado aca para
# que la normalizacion de `frecuencia` (entrada del usuario) y los mensajes
# de error usen siempre el mismo mapeo.
ALIAS_FRECUENCIA = {
    "diaria": "D",
    "semanal": "W",
    "mensual": "MS",
    "trimestral": "QS",
    "anual": "YS",
    "horaria": "H",
}

# Codigos de pandas admitidos ademas de los alias en espanol.
FRECUENCIAS_VALIDAS = {"D", "W", "MS", "M", "QS", "Q", "YS", "Y", "H"}

# Frecuencias con componente horario: se formatean con hora, no solo fecha.
FRECUENCIAS_SUBDIARIAS = {"H"}

# Cuando la fecha original no tiene fechas suficientes, `pd.infer_freq`
# requiere al menos 3 puntos; con menos, no tiene sentido intentarlo.
MINIMO_OBSERVACIONES_PARA_INFERIR = 3

# Tope de periodos faltantes que se listan en la respuesta, para no inflarla
# si la serie es muy irregular.
LIMITE_PERIODOS_FALTANTES_MOSTRADOS = 20


def normalizar_frecuencia(frecuencia: Optional[str]) -> Optional[str]:
    """Traduce un alias pedagogico o valida un codigo de pandas ya soportado.

    No infiere ni adivina: si ``frecuencia`` no es ``None`` y no es
    reconocible, lanza :class:`FrequencyValidationError`.
    """
    if frecuencia is None:
        return None
    if not isinstance(frecuencia, str):
        raise FrequencyValidationError("'frecuencia' debe ser una cadena de texto.")

    candidato = frecuencia.strip()
    alias = ALIAS_FRECUENCIA.get(candidato.lower())
    if alias is not None:
        return alias

    candidato_normalizado = candidato.upper()
    if candidato_normalizado in FRECUENCIAS_VALIDAS:
        return candidato_normalizado

    raise FrequencyValidationError(
        f"'{frecuencia}' no es una frecuencia reconocida. Use un alias en "
        f"espanol ({', '.join(sorted(ALIAS_FRECUENCIA))}) o un codigo de "
        f"pandas ({', '.join(sorted(FRECUENCIAS_VALIDAS))})."
    )


def validar_fechas(fechas: Sequence, n_valores: int) -> pd.DatetimeIndex:
    """Valida y parsea ``fechas``: longitud, ausencia de nulos, orden y duplicados.

    No reordena la serie si las fechas vienen desordenadas: se rechaza
    explicitamente para no alterar la relacion entre posicion y valor sin
    que el usuario lo note.
    """
    if not isinstance(fechas, (list, tuple)):
        raise DateValidationError("'fechas' debe ser una lista de fechas.")
    if len(fechas) != n_valores:
        raise DateLengthMismatchError(
            f"'fechas' debe tener la misma longitud que 'valores' "
            f"({len(fechas)} vs {n_valores})."
        )
    if any(fecha is None for fecha in fechas):
        raise DateValidationError("'fechas' no puede contener valores nulos.")

    try:
        indice = pd.DatetimeIndex(pd.to_datetime(list(fechas), errors="raise"))
    except (ValueError, TypeError) as exc:
        raise DateValidationError(
            "No fue posible interpretar una o mas fechas. Use un formato "
            "ISO 8601 (YYYY-MM-DD, YYYY-MM o YYYY-MM-DDTHH:MM:SS)."
        ) from exc

    if indice.has_duplicates:
        raise DuplicateDatesError("La serie contiene fechas duplicadas.")

    if not indice.is_monotonic_increasing:
        raise UnsortedDatesError(
            "Las fechas no estan en orden cronologico estrictamente "
            "creciente. Reordene los datos antes de enviarlos: no se "
            "reordenan automaticamente."
        )

    return indice


def inferir_frecuencia(indice: pd.DatetimeIndex) -> Optional[str]:
    """Intenta inferir una frecuencia regular a partir de las fechas provistas.

    No inventa una frecuencia si pandas no puede determinarla con certeza
    (serie demasiado corta o irregular): devuelve ``None`` en ese caso.
    """
    if len(indice) < MINIMO_OBSERVACIONES_PARA_INFERIR:
        return None
    try:
        return pd.infer_freq(indice)
    except (ValueError, TypeError):
        return None


def _frecuencia_base(codigo: str) -> str:
    """Codigo de frecuencia sin el ancla (p. ej. 'W-SUN' -> 'W')."""
    return codigo.split("-")[0]


def verificar_frecuencia_compatible(indice: pd.DatetimeIndex, frecuencia: str) -> bool:
    """True si cada fecha de ``indice`` cae sobre la grilla esperada de ``frecuencia``.

    No exige que no falten periodos (eso es responsabilidad de
    :func:`detectar_periodos_faltantes`): solo que las fechas presentes sean
    compatibles con esa periodicidad.
    """
    if len(indice) < 2:
        return True

    inferida = pd.infer_freq(indice)
    if inferida is not None:
        return _frecuencia_base(inferida) == _frecuencia_base(frecuencia)

    esperado = pd.date_range(start=indice.min(), end=indice.max(), freq=frecuencia)
    return bool(indice.isin(esperado).all())


def formatear_fecha_iso(fecha: pd.Timestamp, frecuencia: Optional[str] = None) -> str:
    """Formatea un ``Timestamp`` a ISO: solo fecha, salvo frecuencias sub-diarias."""
    if frecuencia in FRECUENCIAS_SUBDIARIAS:
        return fecha.isoformat()
    return fecha.strftime("%Y-%m-%d")


def detectar_periodos_faltantes(
    indice: pd.DatetimeIndex,
    frecuencia: str,
    limite_mostrado: int = LIMITE_PERIODOS_FALTANTES_MOSTRADOS,
) -> list[str]:
    """Compara ``indice`` contra el rango completo esperado a la frecuencia dada.

    Devuelve las fechas ISO faltantes, acotadas a ``limite_mostrado`` para no
    inflar la respuesta si son demasiadas. No completa los valores: solo
    informa cuales periodos faltan.
    """
    esperado = pd.date_range(start=indice.min(), end=indice.max(), freq=frecuencia)
    faltantes = esperado.difference(indice)
    formateadas = [formatear_fecha_iso(fecha, frecuencia) for fecha in faltantes]
    return formateadas[:limite_mostrado]


def generar_fechas_pronostico(
    indice: Optional[pd.DatetimeIndex],
    frecuencia_utilizada: Optional[str],
    pasos: int,
) -> Optional[list[str]]:
    """Genera ``pasos`` fechas futuras a partir de la ultima fecha observada.

    Devuelve ``None`` si no hay fechas o no se determino una frecuencia: no
    se inventan fechas numericas como si fueran fechas reales.
    """
    if indice is None or frecuencia_utilizada is None:
        return None

    ultima_fecha = indice[-1]
    futuras = pd.date_range(start=ultima_fecha, periods=pasos + 1, freq=frecuencia_utilizada)[1:]
    return [formatear_fecha_iso(fecha, frecuencia_utilizada) for fecha in futuras]


def construir_informacion_temporal(
    fechas: Optional[Sequence],
    frecuencia: Optional[str],
    n_valores: int,
) -> tuple[dict, list[dict], Optional[pd.DatetimeIndex], Optional[str]]:
    """Valida fechas/frecuencia y arma ``informacion_temporal`` mas advertencias.

    Devuelve ``(informacion_temporal, advertencias, indice_fechas,
    frecuencia_utilizada)``. ``indice_fechas`` y ``frecuencia_utilizada`` son
    ``None`` cuando no se proporcionaron fechas o la frecuencia no pudo
    determinarse (ni fue provista ni se pudo inferir).

    Decision de diseno: cuando no hay fechas, no se agrega ninguna
    advertencia de primer nivel del tipo "SERIE_SIN_FECHAS" (generaria ruido
    en practicamente todas las respuestas sin fechas); la ausencia queda
    autodocumentada en ``informacion_temporal.fechas_proporcionadas = False``.
    """
    if fechas is None:
        informacion_temporal = {
            "fechas_proporcionadas": False,
            "frecuencia_solicitada": None,
            "frecuencia_inferida": None,
            "frecuencia_utilizada": None,
            "serie_regular": None,
            "periodos_faltantes": [],
        }
        return informacion_temporal, [], None, None

    indice = validar_fechas(fechas, n_valores)
    frecuencia_solicitada = normalizar_frecuencia(frecuencia)
    frecuencia_inferida = inferir_frecuencia(indice)

    advertencias = []

    if frecuencia_solicitada is not None:
        if not verificar_frecuencia_compatible(indice, frecuencia_solicitada):
            raise InconsistentFrequencyError(
                "Las fechas proporcionadas no son compatibles con la "
                f"frecuencia '{frecuencia_solicitada}'."
            )
        frecuencia_utilizada = frecuencia_solicitada
    else:
        frecuencia_utilizada = frecuencia_inferida
        if frecuencia_utilizada is None:
            advertencias.append({
                "codigo": "FRECUENCIA_NO_INFERIBLE",
                "mensaje": "No fue posible determinar una frecuencia regular a partir de las fechas.",
                "severidad": "advertencia",
            })

    if frecuencia_utilizada is not None:
        periodos_faltantes = detectar_periodos_faltantes(indice, frecuencia_utilizada)
        serie_regular = len(periodos_faltantes) == 0
        if periodos_faltantes:
            ejemplos = ", ".join(periodos_faltantes[:5])
            advertencias.append({
                "codigo": "PERIODOS_FALTANTES",
                "mensaje": f"La serie omite periodo(s): {ejemplos}.",
                "severidad": "advertencia",
            })
    else:
        periodos_faltantes = []
        serie_regular = False

    informacion_temporal = {
        "fechas_proporcionadas": True,
        "frecuencia_solicitada": frecuencia_solicitada,
        "frecuencia_inferida": frecuencia_inferida,
        "frecuencia_utilizada": frecuencia_utilizada,
        "serie_regular": serie_regular,
        "periodos_faltantes": periodos_faltantes,
    }
    return informacion_temporal, advertencias, indice, frecuencia_utilizada
