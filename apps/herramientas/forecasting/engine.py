"""Motor de ajuste y pronostico ARIMA sobre statsmodels.

Responsable de construir el modelo, ajustarlo capturando advertencias del
optimizador, y generar el pronostico con sus intervalos de prediccion. No
valida entradas (eso es responsabilidad de ``validation.py``, ejecutado
antes por la herramienta publica) ni conoce el formato JSON de salida (eso
lo arma ``tools/modelo_arima.py`` a partir del ``ResultadoAjusteARIMA``
devuelto aca).
"""

import warnings

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from statsmodels.tsa.arima.model import ARIMA

from . import diagnostics
from .exceptions import ForecastGenerationError, ModelFitError, NumericalError
from .schemas import ParametroEstimado, PronosticoPaso, ResultadoAjusteARIMA
from .validation import resolver_tendencia

# statsmodels fuerza estacionariedad/invertibilidad por defecto: si el
# optimizador no encuentra una solucion que las respete, el ajuste falla en
# vez de devolver un modelo invalido silenciosamente. En esta fase se
# conservan los valores por defecto (no se exponen como parametro publico).
ENFORCE_STATIONARITY = True
ENFORCE_INVERTIBILITY = True


def _valor_finito(valor) -> float | None:
    if valor is None:
        return None
    numero = float(valor)
    return round(numero, 6) if np.isfinite(numero) else None


def _construir_parametros(ajuste) -> list[ParametroEstimado]:
    nombres = list(ajuste.param_names)
    coeficientes = np.asarray(ajuste.params, dtype=float)
    errores = np.asarray(ajuste.bse, dtype=float)
    p_values = np.asarray(ajuste.pvalues, dtype=float)
    t_values = np.asarray(ajuste.tvalues, dtype=float)
    intervalos_confianza = np.asarray(ajuste.conf_int(), dtype=float)

    parametros = []
    for indice, nombre in enumerate(nombres):
        error_estandar = (
            float(errores[indice])
            if indice < len(errores) and np.isfinite(errores[indice])
            else None
        )
        p_value = (
            float(p_values[indice])
            if indice < len(p_values) and np.isfinite(p_values[indice])
            else None
        )

        # Se prioriza el estadistico t expuesto por statsmodels; solo se
        # recalcula manualmente si no esta disponible o no es finito.
        estadistico = None
        if indice < len(t_values) and np.isfinite(t_values[indice]):
            estadistico = float(t_values[indice])
        elif error_estandar not in (None, 0):
            estadistico = float(coeficientes[indice] / error_estandar)

        intervalo = None
        if indice < len(intervalos_confianza) and np.all(np.isfinite(intervalos_confianza[indice])):
            intervalo = (
                round(float(intervalos_confianza[indice, 0]), 6),
                round(float(intervalos_confianza[indice, 1]), 6),
            )

        parametros.append(
            ParametroEstimado(
                nombre=str(nombre),
                tipo=diagnostics.clasificar_parametro(str(nombre)),
                coeficiente=round(float(coeficientes[indice]), 6),
                error_estandar=round(error_estandar, 6) if error_estandar is not None else None,
                estadistico=round(estadistico, 6) if estadistico is not None else None,
                p_value=round(p_value, 6) if p_value is not None else None,
                intervalo_confianza=intervalo,
                significativo_005=(p_value < 0.05) if p_value is not None else None,
            )
        )
    return parametros


def ajustar_arima(
    serie: np.ndarray,
    p: int,
    d: int,
    q: int,
    con_constante: bool,
    pasos_pronostico: int,
    nivel_confianza: float,
    seasonal_order: tuple[int, int, int, int] | None = None,
    exog: np.ndarray | pd.DataFrame | None = None,
    exog_futuro: np.ndarray | pd.DataFrame | None = None,
) -> ResultadoAjusteARIMA:
    """Ajusta ARIMA(p,d,q) sobre ``serie`` y genera el pronostico con intervalos.

    Si se provee ``seasonal_order=(P,D,Q,s)``, ajusta un SARIMA(p,d,q)(P,D,Q)_s:
    ``statsmodels.tsa.arima.model.ARIMA`` es subclase de
    ``statsmodels.tsa.statespace.sarimax.SARIMAX`` y acepta `seasonal_order`
    de forma nativa (confirmado empiricamente en fase 5), asi que no hace
    falta una segunda funcion de ajuste ni instanciar `SARIMAX` por separado
    para soportar componentes estacionales.

    Si se provee ``exog`` (ARIMAX, fase 6), se ajusta una regresion dinamica:
    ``ARIMA`` acepta `exog` de forma nativa (confirmado empiricamente) y, si
    se le pasa un `pandas.DataFrame` con columnas nombradas, preserva esos
    nombres en `param_names` (p. ej. "temperatura", "promocion") en vez de
    genericos "x1"/"x2" -- por eso las herramientas publicas deben pasar un
    DataFrame, no un array plano, cuando les interese conservar los nombres.
    ``exog_futuro`` se usa unicamente para el pronostico
    (`get_forecast(..., exog=exog_futuro)`), nunca para el ajuste.

    Se asume que ``serie``, los ordenes, el horizonte y las exogenas ya
    fueron validados por ``validation.py``/``exogenous.py`` (o por la
    validacion especifica de la herramienta llamadora, para los ordenes
    estacionales). Lanza excepciones de ``exceptions.py`` ante fallos de
    ajuste, errores numericos o pronosticos no finitos.
    """
    P, D, Q, s = seasonal_order if seasonal_order is not None else (0, 0, 0, 0)
    trend, descripcion_tendencia = resolver_tendencia(d, con_constante, D=D)

    modelo = ARIMA(
        serie,
        exog=exog,
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        trend=trend,
        enforce_stationarity=ENFORCE_STATIONARITY,
        enforce_invertibility=ENFORCE_INVERTIBILITY,
    )

    try:
        with warnings.catch_warnings(record=True) as capturadas:
            warnings.simplefilter("always")
            ajuste = modelo.fit()
    except LinAlgError as exc:
        raise NumericalError(
            f"Error numerico al ajustar ARIMA({p},{d},{q}): la matriz resultante "
            "es singular o esta mal condicionada para esta serie."
        ) from exc
    except Exception as exc:
        raise ModelFitError(
            f"No fue posible ajustar el modelo ARIMA({p},{d},{q}) con los datos "
            "proporcionados."
        ) from exc

    advertencias = diagnostics.clasificar_advertencias(list(capturadas))

    convergio = None
    mle_retvals = getattr(ajuste, "mle_retvals", None)
    if isinstance(mle_retvals, dict):
        convergio = mle_retvals.get("converged")
    if convergio is False and not any(a["codigo"] == "CONVERGENCIA_NO_ALCANZADA" for a in advertencias):
        advertencias.append({
            "codigo": "CONVERGENCIA_NO_ALCANZADA",
            "mensaje": "El optimizador no confirmo convergencia del ajuste.",
            "categoria": "ConvergenceWarning",
            "severidad": "advertencia",
        })

    parametros = _construir_parametros(ajuste)
    residuos = np.asarray(ajuste.resid, dtype=float)

    try:
        pronostico_obj = ajuste.get_forecast(steps=pasos_pronostico, exog=exog_futuro)
        media_pronostico = np.asarray(pronostico_obj.predicted_mean, dtype=float)
        intervalos = np.asarray(pronostico_obj.conf_int(alpha=1 - nivel_confianza), dtype=float)
    except Exception as exc:
        raise ForecastGenerationError(
            f"No fue posible generar el pronostico para ARIMA({p},{d},{q})."
        ) from exc

    if media_pronostico.size != pasos_pronostico or intervalos.shape[0] != pasos_pronostico:
        raise ForecastGenerationError(
            "El pronostico generado no tiene la cantidad de pasos solicitada."
        )
    if not np.all(np.isfinite(media_pronostico)) or not np.all(np.isfinite(intervalos)):
        raise ForecastGenerationError(
            "El pronostico generado contiene valores no finitos; el modelo no es "
            "utilizable con esta configuracion de ordenes."
        )
    if np.any(intervalos[:, 0] > intervalos[:, 1]):
        raise ForecastGenerationError(
            "El intervalo de prediccion generado es inconsistente (limite inferior "
            "mayor que el superior)."
        )

    pronosticos = [
        PronosticoPaso(
            paso=indice + 1,
            pronostico=round(float(media_pronostico[indice]), 6),
            limite_inferior=round(float(intervalos[indice, 0]), 6),
            limite_superior=round(float(intervalos[indice, 1]), 6),
        )
        for indice in range(pasos_pronostico)
    ]

    n_observaciones_efectivas = None
    nobs = getattr(ajuste, "nobs", None)
    if nobs is not None and np.isfinite(nobs):
        n_observaciones_efectivas = int(nobs)

    return ResultadoAjusteARIMA(
        modelo=f"ARIMA({p},{d},{q})",
        orden=(p, d, q),
        orden_estacional=(P, D, Q, s) if seasonal_order is not None else None,
        trend=trend,
        descripcion_tendencia=descripcion_tendencia,
        n_observaciones=int(serie.size),
        n_observaciones_efectivas=n_observaciones_efectivas,
        parametros=parametros,
        aic=_valor_finito(getattr(ajuste, "aic", None)),
        bic=_valor_finito(getattr(ajuste, "bic", None)),
        log_likelihood=_valor_finito(getattr(ajuste, "llf", None)),
        convergio=convergio,
        enforce_stationarity=ENFORCE_STATIONARITY,
        enforce_invertibility=ENFORCE_INVERTIBILITY,
        residuos=residuos,
        pronosticos=pronosticos,
        nivel_confianza=nivel_confianza,
        advertencias=advertencias,
    )
