"""Evaluacion temporal fuera de muestra (holdout cronologico).

Divide la serie en entrenamiento/prueba conservando el orden temporal
(nunca aleatorio), delega el ajuste y pronostico en una funcion provista por
el llamador y calcula MAE/RMSE/MAPE sobre el conjunto de prueba. Este modulo
no conoce ARIMA ni ningun modelo especifico: recibe una funcion
``funcion_pronostico(entrenamiento, pasos) -> valores_pronosticados`` que la
herramienta publica arma envolviendo ``engine.ajustar_arima`` (o, en fases
futuras, el motor de MA/SARIMA/ARIMAX/SARIMAX), por lo que la division
cronologica y las metricas se reutilizan sin duplicarse.
"""

from typing import Callable, Optional, Sequence

import numpy as np

from . import metrics
from .exceptions import ForecastingError, InvalidEvaluationConfigurationError
from .temporal import formatear_fecha_iso

# Un porcentaje de prueba debe ser estrictamente interior a este rango: 0%
# no tiene sentido (no habria prueba) y >=50% dejaria muy poco entrenamiento
# para series cortas, que es el caso tipico de este chatbot academico.
PORCENTAJE_PRUEBA_MINIMO = 0.0
PORCENTAJE_PRUEBA_MAXIMO = 0.5

# Default documentado cuando el usuario activa `evaluar_modelo` sin indicar
# `cantidad_prueba` ni `porcentaje_prueba`: 20% de la serie, con un piso de 1
# observacion y dejando siempre al menos una para entrenamiento.
PORCENTAJE_PRUEBA_DEFAULT = 0.2


def determinar_tamano_prueba(
    n_observaciones: int,
    cantidad_prueba: Optional[int],
    porcentaje_prueba: Optional[float],
) -> int:
    """Determina cuantas observaciones finales se usan como conjunto de prueba.

    Prioridad: ``cantidad_prueba`` > ``porcentaje_prueba`` > default
    documentado (20% de la serie). El resultado siempre deja al menos una
    observacion para entrenamiento y nunca es cero.
    """
    if cantidad_prueba is not None:
        if isinstance(cantidad_prueba, bool) or not isinstance(cantidad_prueba, int):
            raise InvalidEvaluationConfigurationError("'cantidad_prueba' debe ser un entero.")
        if cantidad_prueba < 1:
            raise InvalidEvaluationConfigurationError("'cantidad_prueba' debe ser al menos 1.")
        if cantidad_prueba >= n_observaciones:
            raise InvalidEvaluationConfigurationError(
                "'cantidad_prueba' no puede consumir toda la serie: debe ser "
                f"menor que la cantidad total de observaciones ({n_observaciones})."
            )
        return cantidad_prueba

    if porcentaje_prueba is not None:
        if isinstance(porcentaje_prueba, bool) or not isinstance(porcentaje_prueba, (int, float)):
            raise InvalidEvaluationConfigurationError("'porcentaje_prueba' debe ser numerico.")
        porcentaje = float(porcentaje_prueba)
        if not (PORCENTAJE_PRUEBA_MINIMO < porcentaje < PORCENTAJE_PRUEBA_MAXIMO):
            raise InvalidEvaluationConfigurationError(
                "'porcentaje_prueba' debe estar estrictamente entre "
                f"{PORCENTAJE_PRUEBA_MINIMO} y {PORCENTAJE_PRUEBA_MAXIMO} "
                f"(recibido: {porcentaje})."
            )
        cantidad = max(1, round(n_observaciones * porcentaje))
        return min(cantidad, n_observaciones - 1)

    cantidad = max(1, round(n_observaciones * PORCENTAJE_PRUEBA_DEFAULT))
    return min(cantidad, n_observaciones - 1)


def evaluar_holdout_temporal(
    serie: np.ndarray,
    minimo_observaciones_entrenamiento: int,
    funcion_pronostico: Callable[[np.ndarray, int], Sequence[float]],
    cantidad_prueba: Optional[int] = None,
    porcentaje_prueba: Optional[float] = None,
    fechas_indice=None,
) -> dict:
    """Ejecuta un holdout temporal: ajusta solo con entrenamiento, evalua contra prueba.

    No reajusta el modelo con la serie completa: esa es una operacion
    distinta que el llamador (``tools/modelo_arima.py``) hace por separado
    para producir el pronostico futuro final.

    Decision documentada: una configuracion invalida de
    ``cantidad_prueba``/``porcentaje_prueba`` se trata como error duro (se
    propaga la excepcion) porque es un dato de entrada incorrecto, igual que
    un orden ARIMA invalido. En cambio, si la configuracion es valida pero no
    queda suficiente entrenamiento (o el ajuste de evaluacion falla), se
    omite unicamente la evaluacion (``{"ejecutada": False, "motivo": ...}``)
    y se permite que el llamador continue con el ajuste final sobre toda la
    serie, salvo que este exija explicitamente que la evaluacion sea
    obligatoria.
    """
    n_total = int(serie.size)
    n_prueba = determinar_tamano_prueba(n_total, cantidad_prueba, porcentaje_prueba)
    n_entrenamiento = n_total - n_prueba

    if n_entrenamiento < minimo_observaciones_entrenamiento:
        return {
            "ejecutada": False,
            "motivo": (
                "No existen suficientes observaciones para separar entrenamiento "
                f"y prueba: se necesitan al menos {minimo_observaciones_entrenamiento} "
                f"observaciones de entrenamiento y solo quedarian {n_entrenamiento} "
                f"de {n_total} totales."
            ),
        }

    entrenamiento = serie[:n_entrenamiento]
    prueba = serie[n_entrenamiento:]

    try:
        pronosticados = [float(v) for v in funcion_pronostico(entrenamiento, n_prueba)]
    except ForecastingError as exc:
        return {
            "ejecutada": False,
            "motivo": f"No fue posible completar la evaluacion fuera de muestra: {exc}",
        }

    if len(pronosticados) != n_prueba:
        return {
            "ejecutada": False,
            "motivo": "El pronostico generado para la evaluacion no tiene la longitud esperada.",
        }

    reales = [round(float(valor), 6) for valor in prueba]
    pronosticados_redondeados = [round(valor, 6) for valor in pronosticados]

    mae = metrics.calcular_mae(reales, pronosticados_redondeados)
    rmse = metrics.calcular_rmse(reales, pronosticados_redondeados)
    resultado_mape = metrics.calcular_mape(reales, pronosticados_redondeados)

    evaluacion = {
        "ejecutada": True,
        "estrategia": "holdout_temporal",
        "n_observaciones_totales": n_total,
        "n_entrenamiento": n_entrenamiento,
        "n_prueba": n_prueba,
        "indice_inicio_prueba": n_entrenamiento,
        "valores_reales": reales,
        "valores_pronosticados": pronosticados_redondeados,
        "metricas_prueba": {
            "mae": mae,
            "rmse": rmse,
            "mape": resultado_mape["mape"],
        },
        "mape_detalle": resultado_mape["mape_detalle"],
        "advertencias": list(resultado_mape["mape_detalle"].get("advertencias", [])),
    }

    if fechas_indice is not None:
        fechas_entrenamiento = fechas_indice[:n_entrenamiento]
        fechas_prueba = fechas_indice[n_entrenamiento:]
        evaluacion["periodo_entrenamiento"] = {
            "inicio": formatear_fecha_iso(fechas_entrenamiento[0]),
            "fin": formatear_fecha_iso(fechas_entrenamiento[-1]),
        }
        evaluacion["periodo_prueba"] = {
            "inicio": formatear_fecha_iso(fechas_prueba[0]),
            "fin": formatear_fecha_iso(fechas_prueba[-1]),
        }
        evaluacion["fechas_prueba"] = [formatear_fecha_iso(fecha) for fecha in fechas_prueba]

    return evaluacion
