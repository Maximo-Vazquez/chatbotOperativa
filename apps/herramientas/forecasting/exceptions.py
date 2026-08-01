"""Excepciones de dominio del nucleo de pronostico.

Todas heredan de ``ForecastingError`` para que la frontera publica de cada
herramienta (``tools/*.py``) pueda atraparlas con un unico ``except`` y
convertirlas en una respuesta serializable sin perder informacion sobre
el tipo de fallo (``codigo_error``).
"""


class ForecastingError(Exception):
    """Error base del nucleo de pronostico."""

    codigo_error = "ERROR_PRONOSTICO"


class InvalidSeriesError(ForecastingError):
    """La serie de entrada no es utilizable (vacia, no numerica, NaN, etc.)."""

    codigo_error = "SERIE_INVALIDA"


class InvalidOrderError(ForecastingError):
    """El orden del modelo (p, d, q, ...) es invalido o esta fuera de rango."""

    codigo_error = "ORDEN_INVALIDA"


class InvalidSeasonalOrderError(InvalidOrderError):
    """Los ordenes estacionales P, D o Q son invalidos (tipo, signo o maximo pedagogico)."""

    codigo_error = "ORDEN_ESTACIONAL_INVALIDO"


class InvalidSeasonalPeriodError(InvalidOrderError):
    """La periodicidad `s` es invalida (tipo, `s<2` o maximo pedagogico)."""

    codigo_error = "PERIODICIDAD_INVALIDA"


class SeasonalPeriodRequiredError(InvalidSeasonalPeriodError):
    """Existe algun componente estacional (P, D o Q > 0) pero no se proveyo `s`.

    Subclase de `InvalidSeasonalPeriodError` (fase 7, `modelo_sarimax`): es un
    caso distinto de "s invalido" (aca directamente falta), pero conceptualmente
    sigue siendo un problema de periodicidad, no de los ordenes P/D/Q en si.
    """

    codigo_error = "PERIODICIDAD_ESTACIONAL_REQUERIDA"


class InvalidMAOrderError(InvalidOrderError):
    """El orden q de un modelo MA publico no cumple la regla q >= 1 o excede el maximo.

    Subclase de `InvalidOrderError`: los problemas genericos de tipo/signo de
    `q` (booleano, no entero, negativo) ya los cubre la validacion compartida
    (`validation.validar_orden_arima`) con el codigo generico `ORDEN_INVALIDA`.
    Este codigo especifico es solo para la regla propia de la herramienta MA
    (q debe ser >=1, no 0, y no exceder el maximo pedagogico admitido).
    """

    codigo_error = "ORDEN_MA_INVALIDO"


class InsufficientDataError(ForecastingError):
    """La serie no tiene observaciones suficientes para el orden solicitado."""

    codigo_error = "MUESTRA_INSUFICIENTE"


class InvalidForecastHorizonError(ForecastingError):
    """El horizonte de pronostico (pasos) es invalido."""

    codigo_error = "HORIZONTE_INVALIDO"


class InvalidConfidenceLevelError(ForecastingError):
    """El nivel de confianza solicitado esta fuera del rango admitido."""

    codigo_error = "NIVEL_CONFIANZA_INVALIDO"


class ModelFitError(ForecastingError):
    """El modelo no pudo ajustarse (fallo del optimizador de statsmodels)."""

    codigo_error = "ERROR_AJUSTE"


class ModelConvergenceError(ForecastingError):
    """El ajuste no convergio y el resultado no es utilizable con confianza."""

    codigo_error = "ERROR_CONVERGENCIA"


class NumericalError(ForecastingError):
    """Error numerico durante el ajuste (matriz singular, mal condicionada, etc.)."""

    codigo_error = "ERROR_NUMERICO"


class ForecastGenerationError(ForecastingError):
    """El pronostico no pudo generarse o el resultado no es finito."""

    codigo_error = "ERROR_PRONOSTICO_GENERACION"


# ── Metricas (fase 3) ──────────────────────────────────────────────────────


class MetricCalculationError(ForecastingError):
    """Los valores provistos a una metrica (MAE/RMSE/MAPE) no son utilizables."""

    codigo_error = "METRICAS_ENTRADA_INVALIDA"


class MetricLengthMismatchError(MetricCalculationError):
    """Los valores reales y los pronosticados tienen distinta longitud."""

    codigo_error = "METRICAS_LONGITUD_INCOMPATIBLE"


# ── Evaluacion temporal fuera de muestra (fase 3) ──────────────────────────


class InvalidEvaluationConfigurationError(ForecastingError):
    """`cantidad_prueba`/`porcentaje_prueba` son invalidos o incompatibles entre si."""

    codigo_error = "CONFIGURACION_PRUEBA_INVALIDA"


class InsufficientTrainingDataError(ForecastingError):
    """No queda suficiente entrenamiento tras separar el conjunto de prueba.

    No se usa para abortar el ajuste: `evaluation.py` la reserva para casos
    donde el llamador decida tratar una evaluacion imposible como un error
    duro en vez de omitirla (ver decision documentada en `evaluation.py`).
    """

    codigo_error = "ENTRENAMIENTO_INSUFICIENTE"


# ── Fechas y frecuencia (fase 3) ────────────────────────────────────────────


class DateValidationError(ForecastingError):
    """Una fecha no pudo interpretarse o la entrada de fechas es invalida."""

    codigo_error = "FECHA_INVALIDA"


class DateLengthMismatchError(DateValidationError):
    """`fechas` no tiene la misma longitud que `valores`."""

    codigo_error = "FECHAS_LONGITUD_INCOMPATIBLE"


class DuplicateDatesError(DateValidationError):
    """La serie contiene fechas repetidas."""

    codigo_error = "FECHAS_DUPLICADAS"


class UnsortedDatesError(DateValidationError):
    """Las fechas no estan en orden cronologico estrictamente creciente."""

    codigo_error = "FECHAS_DESORDENADAS"


class FrequencyValidationError(ForecastingError):
    """La frecuencia solicitada no es un alias ni un codigo de pandas reconocido."""

    codigo_error = "FRECUENCIA_INVALIDA"


class InconsistentFrequencyError(ForecastingError):
    """Las fechas provistas no son compatibles con la frecuencia solicitada."""

    codigo_error = "FRECUENCIA_INCONSISTENTE"


# ── Variables exogenas / ARIMAX (fase 6) ───────────────────────────────────


class ExogenousRequiredError(ForecastingError):
    """No se proporciono ninguna variable exogena historica."""

    codigo_error = "EXOGENAS_HISTORICAS_REQUERIDAS"


class FutureExogenousRequiredError(ForecastingError):
    """Se necesita pronosticar pero no se proveyeron variables exogenas futuras."""

    codigo_error = "EXOGENAS_FUTURAS_REQUERIDAS"


class ExogenousValueError(ForecastingError):
    """Una variable exogena no es una estructura numerica valida (tipo, booleanos,
    estructuras anidadas, nombres invalidos o duplicados)."""

    codigo_error = "EXOGENA_NO_NUMERICA"


class ExogenousNonFiniteError(ForecastingError):
    """Una variable exogena contiene valores NaN o infinitos."""

    codigo_error = "EXOGENA_NO_FINITA"


class ExogenousLengthMismatchError(ForecastingError):
    """Las variables exogenas no tienen la longitud esperada (entre si, contra
    `valores`, o contra `pasos_pronostico` en el caso de las futuras)."""

    codigo_error = "EXOGENA_LONGITUD_INCOMPATIBLE"


class ExogenousColumnMismatchError(ForecastingError):
    """Las variables exogenas futuras no coinciden exactamente con las historicas."""

    codigo_error = "EXOGENAS_COLUMNAS_INCOMPATIBLES"


class ExogenousDuplicateError(ForecastingError):
    """Dos variables exogenas historicas son exactamente identicas."""

    codigo_error = "EXOGENAS_DUPLICADAS"


class ExogenousSingularMatrixError(ForecastingError):
    """La matriz de variables exogenas (con la constante, si aplica) no tiene rango completo."""

    codigo_error = "EXOGENAS_MATRIZ_DEGENERADA"


class ExogenousDateMismatchError(ForecastingError):
    """Las fechas de las variables exogenas (historicas o futuras) no estan alineadas
    con las fechas de la serie objetivo o del pronostico."""

    codigo_error = "FECHAS_EXOGENAS_DESALINEADAS"


class ForecastHorizonExogenousMismatchError(ForecastingError):
    """La cantidad de valores futuros de una variable exogena no coincide con `pasos_pronostico`."""

    codigo_error = "HORIZONTE_EXOGENAS_INCOMPATIBLE"
