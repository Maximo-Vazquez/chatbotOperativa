"""Validacion y analisis de componentes estacionales (P, D, Q, s).

Extraido en fase 7 desde `tools/modelo_sarima.py` (donde vivia como logica
privada de esa unica fachada) para que `modelo_sarima` y el nuevo
`modelo_sarimax` lo reutilicen sin duplicar nada. No conoce ARIMA/exogenas:
solo validacion de ordenes/periodicidad y analisis de ciclos/coherencia con
la frecuencia, reutilizable por cualquier fachada con componente estacional.
"""

from typing import Optional

from .exceptions import InvalidOrderError, InvalidSeasonalOrderError, InvalidSeasonalPeriodError

# Limites pedagogicos propios de las herramientas con componente estacional
# (no de ARIMA/MA, que no fijan techo a p/q): la combinacion de componentes
# regulares y estacionales crece mucho mas rapido en complejidad.
P_REGULAR_MAXIMO = 5
Q_REGULAR_MAXIMO = 5
P_ESTACIONAL_MAXIMO = 3
Q_ESTACIONAL_MAXIMO = 3
# D<=1 (no 2): D=2 es rarisimo en la practica y exige 2*s observaciones
# extra solo para la diferenciacion estacional, demasiado para series
# tipicamente cortas en el uso academico de estas herramientas.
D_ESTACIONAL_MAXIMO = 1
S_MAXIMO = 366

# Combinaciones frecuencia-periodicidad consideradas habituales en la
# practica. Se comparan por la base del codigo de frecuencia (sin anclaje,
# p. ej. "QS-OCT" -> "QS").
COMBINACIONES_HABITUALES = {
    ("D", 7): "un ciclo semanal sobre una serie diaria",
    ("W", 52): "un ciclo anual sobre una serie semanal",
    ("MS", 12): "un ciclo anual sobre una serie mensual",
    ("M", 12): "un ciclo anual sobre una serie mensual",
    ("QS", 4): "un ciclo anual sobre una serie trimestral",
    ("Q", 4): "un ciclo anual sobre una serie trimestral",
    ("H", 24): "un ciclo diario sobre una serie horaria",
}


def validar_techo_ordenes_regulares(
    p: int, q: int, p_maximo: int = P_REGULAR_MAXIMO, q_maximo: int = Q_REGULAR_MAXIMO
) -> None:
    """Limite pedagogico de p/q para fachadas con componente estacional opcional.

    No son ordenes "estacionales": un limite excedido aca usa el codigo
    generico de orden invalido, no `ORDEN_ESTACIONAL_INVALIDO` (reservado a P/D/Q).
    """
    if p > p_maximo:
        raise InvalidOrderError(f"El orden p={p} supera el maximo admitido por esta herramienta ({p_maximo}).")
    if q > q_maximo:
        raise InvalidOrderError(f"El orden q={q} supera el maximo admitido por esta herramienta ({q_maximo}).")


def validar_ordenes_estacionales(P: int, D: int, Q: int) -> None:
    """Valida tipo, signo y techo pedagogico de P, D, Q."""
    for nombre, valor in (("P", P), ("D", D), ("Q", Q)):
        if isinstance(valor, bool) or not isinstance(valor, int):
            raise InvalidSeasonalOrderError(
                f"Los ordenes estacionales P, D y Q deben ser enteros no negativos "
                f"('{nombre}' recibido: {valor!r})."
            )
        if valor < 0:
            raise InvalidSeasonalOrderError(
                f"Los ordenes estacionales P, D y Q deben ser enteros no negativos "
                f"('{nombre}' recibido: {valor})."
            )
    if P > P_ESTACIONAL_MAXIMO:
        raise InvalidSeasonalOrderError(f"El orden estacional P={P} supera el maximo admitido ({P_ESTACIONAL_MAXIMO}).")
    if Q > Q_ESTACIONAL_MAXIMO:
        raise InvalidSeasonalOrderError(f"El orden estacional Q={Q} supera el maximo admitido ({Q_ESTACIONAL_MAXIMO}).")
    if D > D_ESTACIONAL_MAXIMO:
        raise InvalidSeasonalOrderError(
            f"El orden de diferenciacion estacional D={D} supera el maximo admitido ({D_ESTACIONAL_MAXIMO})."
        )


def validar_periodicidad(s) -> None:
    """Valida tipo, signo y techo pedagogico de `s`. No decide si `s` es obligatorio:
    esa regla depende de si hay componente estacional y la resuelve el llamador."""
    if isinstance(s, bool) or not isinstance(s, int):
        raise InvalidSeasonalPeriodError(f"La periodicidad 's' debe ser un entero, se recibio {s!r}.")
    if s < 2:
        raise InvalidSeasonalPeriodError("La periodicidad 's' debe ser mayor o igual que 2.")
    if s > S_MAXIMO:
        raise InvalidSeasonalPeriodError(f"La periodicidad s={s} supera el maximo admitido ({S_MAXIMO}).")


def advertencia_ciclos_insuficientes() -> dict:
    return {
        "codigo": "CICLOS_ESTACIONALES_INSUFICIENTES",
        "mensaje": (
            "La serie contiene menos de dos ciclos estacionales completos. Dos "
            "ciclos son un minimo tecnico orientativo, no una garantia "
            "estadistica: se recomiendan al menos tres o mas ciclos completos "
            "para una estimacion mas estable."
        ),
        "severidad": "advertencia_alta",
    }


def analizar_ciclos(n_observaciones: int, s: int) -> tuple[dict, list]:
    """Calcula `n_ciclos_aproximados` y advierte si son insuficientes/limitados."""
    n_ciclos = n_observaciones / s
    advertencias = []
    if n_observaciones < 2 * s:
        advertencias.append(advertencia_ciclos_insuficientes())
    elif n_observaciones < 3 * s:
        advertencias.append({
            "codigo": "CICLOS_ESTACIONALES_LIMITADOS",
            "mensaje": (
                f"La serie cubre {n_ciclos:.1f} ciclos estacionales. Se recomiendan "
                "al menos tres o mas ciclos completos para un diagnostico mas confiable."
            ),
            "severidad": "advertencia",
        })
    info = {"n_observaciones": n_observaciones, "periodicidad": s, "n_ciclos_aproximados": round(n_ciclos, 2)}
    return info, advertencias


def clasificar_coherencia_estacional(frecuencia_utilizada: Optional[str], s: int) -> tuple[dict, list]:
    """Clasifica la combinacion frecuencia/periodicidad como habitual, poco
    habitual o sin informacion (nunca "invalida": eso ya lo bloquean otras
    validaciones, como la frecuencia explicita incompatible con las fechas)."""
    if frecuencia_utilizada is None:
        return (
            {
                "frecuencia": None, "periodicidad": s, "clasificacion": "sin_informacion",
                "interpretacion": "No hay una frecuencia determinada para evaluar la coherencia con 's'.",
            },
            [],
        )

    base = frecuencia_utilizada.split("-")[0]
    descripcion = COMBINACIONES_HABITUALES.get((base, s))
    if descripcion:
        return (
            {
                "frecuencia": frecuencia_utilizada, "periodicidad": s, "clasificacion": "habitual",
                "interpretacion": f"La serie usa {descripcion}.",
            },
            [],
        )

    coherencia = {
        "frecuencia": frecuencia_utilizada, "periodicidad": s, "clasificacion": "poco_habitual",
        "interpretacion": (
            f"La combinacion de frecuencia '{frecuencia_utilizada}' y periodicidad "
            f"s={s} es matematicamente valida pero menos habitual; verifique que "
            "representa el ciclo estacional que espera."
        ),
    }
    advertencia = [{
        "codigo": "FRECUENCIA_Y_PERIODICIDAD_INUSUALES",
        "mensaje": coherencia["interpretacion"],
        "severidad": "informacion",
    }]
    return coherencia, advertencia
