"""Validaciones de entrada del nucleo de pronostico.

Cada funcion valida un unico aspecto de la solicitud (serie, orden, horizonte,
nivel de confianza, configuracion de tendencia) y lanza una excepcion de
``exceptions.py`` con un mensaje comprensible cuando la entrada no es
utilizable. No dependen de statsmodels: son chequeos previos al ajuste.
"""

from typing import Sequence

import numpy as np

from .exceptions import (
    InsufficientDataError,
    InvalidConfidenceLevelError,
    InvalidForecastHorizonError,
    InvalidOrderError,
    InvalidSeriesError,
)

# Limites del contrato publico actual de `modelo_arima` (TOOL_DEFINITION).
D_MAXIMO = 2
PASOS_PRONOSTICO_MAXIMO = 50
NIVEL_CONFIANZA_MINIMO = 0.80
NIVEL_CONFIANZA_MAXIMO = 0.999


def validar_serie(valores: Sequence) -> np.ndarray:
    """Valida que ``valores`` sea una lista de numeros finitos y la convierte a ``ndarray``.

    No valida cantidad minima de observaciones ni varianza: esas son
    responsabilidad de :func:`validar_muestra_minima` y
    :func:`validar_serie_no_constante`, que dependen del orden solicitado.
    """
    if not isinstance(valores, (list, tuple)):
        raise InvalidSeriesError("La serie debe ser una lista de valores numericos.")
    if len(valores) == 0:
        raise InvalidSeriesError("La serie no puede estar vacia.")

    numeros = []
    for indice, valor in enumerate(valores):
        if isinstance(valor, bool):
            raise InvalidSeriesError(
                f"El valor en la posicion {indice} es booleano; se esperaba un numero."
            )
        if not isinstance(valor, (int, float)):
            raise InvalidSeriesError(
                f"El valor en la posicion {indice} no es numerico: {valor!r}."
            )
        numeros.append(float(valor))

    serie = np.asarray(numeros, dtype=float)

    if np.isnan(serie).any():
        raise InvalidSeriesError("La serie contiene valores NaN.")
    if np.isinf(serie).any():
        raise InvalidSeriesError("La serie contiene valores infinitos.")

    return serie


def validar_serie_no_constante(serie: np.ndarray) -> None:
    """Rechaza series con varianza nula: no corresponde ajustar ARIMA sobre ellas."""
    if serie.size > 0 and np.ptp(serie) == 0:
        raise InvalidSeriesError(
            "La serie es constante (varianza nula); no corresponde ajustar un "
            "modelo de esta familia (ARIMA/MA). Use la herramienta de "
            "descomposicion o ADF para diagnosticar series constantes."
        )


def validar_orden_arima(p: int, d: int, q: int) -> None:
    """Valida tipos y rango de p, d, q. No valida tamano de muestra."""
    for nombre, valor in (("p", p), ("d", d), ("q", q)):
        if isinstance(valor, bool) or not isinstance(valor, int):
            raise InvalidOrderError(
                f"El orden '{nombre}' debe ser un entero, se recibio {valor!r}."
            )
        if valor < 0:
            raise InvalidOrderError(
                f"El orden '{nombre}' no puede ser negativo (valor recibido: {valor})."
            )
    if d > D_MAXIMO:
        raise InvalidOrderError(
            f"El grado de integracion d={d} supera el maximo admitido ({D_MAXIMO})."
        )


def calcular_minimo_observaciones(p: int, d: int, q: int) -> int:
    """Minimo de observaciones exigido para ajustar ARIMA(p,d,q).

    Heuristica conservadora ya usada por la implementacion anterior: al menos
    3 observaciones por encima de la cantidad de parametros estructurales.
    """
    return p + d + q + 3


def validar_muestra_minima(n_observaciones: int, p: int, d: int, q: int) -> int:
    """Valida que la serie tenga al menos el minimo de observaciones para el orden dado."""
    minimo = calcular_minimo_observaciones(p, d, q)
    if n_observaciones < minimo:
        raise InsufficientDataError(
            f"Se necesitan al menos {minimo} observaciones para ajustar "
            f"ARIMA({p},{d},{q}); se recibieron {n_observaciones}."
        )
    return minimo


def calcular_minimo_observaciones_general(
    p: int,
    d: int,
    q: int,
    P: int = 0,
    D: int = 0,
    Q: int = 0,
    s: int = 0,
    n_exogenas: int = 0,
    con_constante: bool = True,
    margen: int = 3,
) -> int:
    """Generalizacion de `calcular_minimo_observaciones` para configuraciones que
    combinan componentes regulares, estacionales y/o variables exogenas
    (usada por SARIMA/ARIMAX/SARIMAX). Con los valores por defecto
    (`P=D=Q=s=n_exogenas=0`) el calculo de "perdidas por diferenciacion +
    parametros + margen" coincide con el criterio ya usado en las fachadas
    de MA/SARIMA/ARIMAX de fases anteriores (que si sumaban +1 por la
    constante); no reemplaza `calcular_minimo_observaciones` (que no suma la
    constante) para no alterar el contrato ya probado de ARIMA.
    """
    perdidas_por_diferenciacion = d + D * s
    parametros = p + q + P + Q + n_exogenas + (1 if con_constante else 0)
    return perdidas_por_diferenciacion + parametros + margen


def advertencia_configuracion_compleja(
    parametros_estimados: int,
    n_observaciones: int,
    codigo: str,
    multiplicador: int = 5,
) -> list[dict]:
    """Advertencia no bloqueante cuando el modelo puede ajustarse pero la
    relacion parametros/observaciones es pobre (menos de `multiplicador`
    observaciones por parametro estimado). Reutilizada por las fachadas que
    combinan varios componentes (SARIMA, SARIMAX); no reemplaza el error
    duro por muestra tecnicamente insuficiente, que se valida aparte.
    """
    if parametros_estimados > 0 and n_observaciones < parametros_estimados * multiplicador:
        return [{
            "codigo": codigo,
            "mensaje": (
                f"El modelo puede ajustarse, pero tiene {parametros_estimados} "
                f"parametros estimados para {n_observaciones} observaciones: el "
                "diagnostico residual y las metricas pueden ser poco confiables "
                "con esta relacion parametros/muestra."
            ),
            "severidad": "advertencia",
        }]
    return []


def validar_horizonte_pronostico(pasos_pronostico: int) -> None:
    """Valida que el horizonte de pronostico sea un entero dentro del contrato actual."""
    if isinstance(pasos_pronostico, bool) or not isinstance(pasos_pronostico, int):
        raise InvalidForecastHorizonError("'pasos_pronostico' debe ser un entero.")
    if pasos_pronostico < 1:
        raise InvalidForecastHorizonError("'pasos_pronostico' debe ser al menos 1.")
    if pasos_pronostico > PASOS_PRONOSTICO_MAXIMO:
        raise InvalidForecastHorizonError(
            f"'pasos_pronostico' no puede superar {PASOS_PRONOSTICO_MAXIMO}."
        )


def validar_nivel_confianza(nivel_confianza) -> float:
    """Valida y normaliza el nivel de confianza para los intervalos de prediccion."""
    if isinstance(nivel_confianza, bool) or not isinstance(nivel_confianza, (int, float)):
        raise InvalidConfidenceLevelError("'nivel_confianza' debe ser numerico.")
    nivel = float(nivel_confianza)
    if not (NIVEL_CONFIANZA_MINIMO <= nivel <= NIVEL_CONFIANZA_MAXIMO):
        raise InvalidConfidenceLevelError(
            f"'nivel_confianza' debe estar entre {NIVEL_CONFIANZA_MINIMO} y "
            f"{NIVEL_CONFIANZA_MAXIMO} (recibido: {nivel})."
        )
    return nivel


def resolver_tendencia(d: int, con_constante: bool, D: int = 0) -> tuple[str, str]:
    """Determina el codigo de tendencia de statsmodels y su descripcion en texto plano.

    statsmodels rechaza terminos deterministicos de orden menor que `d + D`
    (confirmado empiricamente: el mensaje de error de
    ``statsmodels.tsa.arima.model.ARIMA`` habla explicitamente de "integration
    (d > 0) or seasonal integration (D > 0)" como una unica restriccion
    conjunta). Con `d+D=0` solo admite constante ('c'); con `d+D=1`, drift
    lineal ('t'); con `d+D>=2`, ningun termino determinista es admisible via
    el parametro `trend`. `D` es 0 por defecto para no afectar a ARIMA/MA
    (sin componente estacional).
    """
    if not con_constante:
        return "n", "Sin termino deterministico (constante o drift deshabilitados por el usuario)."

    d_total = d + D
    if d_total == 0:
        return "c", "Constante aplicada sobre un modelo sin diferenciar (nivel medio distinto de cero)."
    if d_total == 1:
        origenes = []
        if d >= 1:
            origenes.append("diferenciacion regular")
        if D >= 1:
            origenes.append("diferenciacion estacional")
        via = " y ".join(origenes)
        return "t", f"Drift lineal aplicado sobre un modelo integrado de orden 1 (via {via})."
    return "n", (
        f"Sin termino deterministico: statsmodels no admite tendencia con "
        f"d+D={d_total}>=2."
    )
