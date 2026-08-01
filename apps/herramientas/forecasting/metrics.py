"""Metricas de error para pronosticos: MAE, RMSE y MAPE.

Funciones puras que comparan valores reales contra valores pronosticados.
Aceptan listas, tuplas o arreglos de NumPy y siempre devuelven tipos
nativos de Python (float/int/bool/None), listos para serializar a JSON. No
conocen statsmodels ni el motor de ajuste (`engine.py`): las usa tanto el
diagnostico in-sample como la evaluacion fuera de muestra (`evaluation.py`),
y podran reutilizarse sin cambios desde MA/SARIMA/ARIMAX/SARIMAX.
"""

from typing import Sequence

import numpy as np

from .exceptions import MetricCalculationError, MetricLengthMismatchError

# Valores reales con |y| menor a esta tolerancia se consideran "cercanos a
# cero" a efectos de advertir sobre la inestabilidad de MAPE. No se excluyen
# del calculo (solo se excluyen los que son exactamente cero): la tolerancia
# es para advertir, no para alterar silenciosamente los datos.
TOLERANCIA_MAPE_CERCANO_A_CERO = 1e-6


def _a_arreglo_numerico(valores: Sequence, nombre: str) -> np.ndarray:
    """Convierte ``valores`` a ``ndarray[float]``, validando tipo y finitud."""
    try:
        lista = list(valores)
    except TypeError as exc:
        raise MetricCalculationError(f"'{nombre}' debe ser una secuencia de numeros.") from exc

    if len(lista) == 0:
        raise MetricCalculationError(f"'{nombre}' no puede estar vacio.")

    numeros = []
    for indice, valor in enumerate(lista):
        if isinstance(valor, bool):
            raise MetricCalculationError(
                f"El valor en la posicion {indice} de '{nombre}' es booleano; "
                "se esperaba un numero."
            )
        if not isinstance(valor, (int, float, np.integer, np.floating)):
            raise MetricCalculationError(
                f"El valor en la posicion {indice} de '{nombre}' no es numerico: {valor!r}."
            )
        numeros.append(float(valor))

    arreglo = np.asarray(numeros, dtype=float)
    if np.isnan(arreglo).any():
        raise MetricCalculationError(f"'{nombre}' contiene valores NaN.")
    if np.isinf(arreglo).any():
        raise MetricCalculationError(f"'{nombre}' contiene valores infinitos.")
    return arreglo


def _validar_par(
    valores_reales: Sequence, valores_pronosticados: Sequence
) -> tuple[np.ndarray, np.ndarray]:
    reales = _a_arreglo_numerico(valores_reales, "valores_reales")
    pronosticados = _a_arreglo_numerico(valores_pronosticados, "valores_pronosticados")
    if reales.shape != pronosticados.shape:
        raise MetricLengthMismatchError(
            "'valores_reales' y 'valores_pronosticados' tienen longitudes "
            f"distintas ({reales.size} vs {pronosticados.size})."
        )
    return reales, pronosticados


def calcular_mae(valores_reales: Sequence, valores_pronosticados: Sequence) -> float:
    """Error absoluto medio: misma unidad que la serie, sensible por igual a todo error."""
    reales, pronosticados = _validar_par(valores_reales, valores_pronosticados)
    return round(float(np.mean(np.abs(reales - pronosticados))), 6)


def calcular_rmse(valores_reales: Sequence, valores_pronosticados: Sequence) -> float:
    """Raiz del error cuadratico medio: misma unidad que la serie, penaliza mas los errores grandes."""
    reales, pronosticados = _validar_par(valores_reales, valores_pronosticados)
    return round(float(np.sqrt(np.mean((reales - pronosticados) ** 2))), 6)


def calcular_mape(
    valores_reales: Sequence,
    valores_pronosticados: Sequence,
    tolerancia_cero: float = TOLERANCIA_MAPE_CERCANO_A_CERO,
) -> dict:
    """Error porcentual absoluto medio, excluyendo del calculo los reales exactamente cero.

    Devuelve ``{"mape": float|None, "mape_detalle": {...}}``. Si todos los
    valores reales son cero, ``mape`` es ``None`` y ``mape_detalle.calculado``
    es ``False`` (no se devuelve infinito ni un numero artificialmente
    grande). Los valores reales negativos no se rechazan, pero generan una
    advertencia conceptual: MAPE es dificil de interpretar con negativos.
    """
    reales, pronosticados = _validar_par(valores_reales, valores_pronosticados)

    observaciones_totales = int(reales.size)
    es_cero = reales == 0.0
    observaciones_excluidas = int(np.sum(es_cero))

    if observaciones_excluidas == observaciones_totales:
        return {
            "mape": None,
            "mape_detalle": {
                "calculado": False,
                "observaciones_totales": observaciones_totales,
                "observaciones_utilizadas": 0,
                "observaciones_excluidas_por_cero": observaciones_excluidas,
                "motivo": "No puede calcularse MAPE porque todos los valores reales son cero.",
                "advertencias": [],
            },
        }

    reales_utiles = reales[~es_cero]
    pronosticados_utiles = pronosticados[~es_cero]
    errores_porcentuales = np.abs((reales_utiles - pronosticados_utiles) / reales_utiles)
    mape = round(float(100.0 * np.mean(errores_porcentuales)), 6)

    advertencias = []
    if observaciones_excluidas > 0:
        plural = observaciones_excluidas != 1
        advertencias.append({
            "codigo": "MAPE_VALORES_CERO_EXCLUIDOS",
            "mensaje": (
                f"Se {'excluyeron' if plural else 'excluyo'} {observaciones_excluidas} "
                f"observacion{'es' if plural else ''} cuyo valor real era cero."
            ),
            "severidad": "advertencia",
        })

    cercanos_a_cero = int(
        np.sum((np.abs(reales_utiles) > 0) & (np.abs(reales_utiles) < tolerancia_cero))
    )
    if cercanos_a_cero > 0:
        advertencias.append({
            "codigo": "MAPE_VALORES_CERCANOS_A_CERO",
            "mensaje": (
                f"{cercanos_a_cero} valor(es) real(es) estan muy cerca de cero "
                f"(< {tolerancia_cero}); MAPE puede ser inestable en esas observaciones."
            ),
            "severidad": "advertencia",
        })

    if np.any(reales_utiles < 0):
        advertencias.append({
            "codigo": "MAPE_VALORES_NEGATIVOS",
            "mensaje": (
                "MAPE puede resultar dificil de interpretar cuando la serie "
                "contiene valores reales negativos."
            ),
            "severidad": "advertencia",
        })

    return {
        "mape": mape,
        "mape_detalle": {
            "calculado": True,
            "observaciones_totales": observaciones_totales,
            "observaciones_utilizadas": observaciones_totales - observaciones_excluidas,
            "observaciones_excluidas_por_cero": observaciones_excluidas,
            "advertencias": advertencias,
        },
    }
