"""Pruebas del nucleo compartido de pronostico: validaciones, excepciones de
dominio y serializacion JSON. No dependen de statsmodels ni ajustan modelos:
verifican unicamente `apps/herramientas/forecasting/validation.py` y las
excepciones de `exceptions.py`.
"""

import json
import math

import numpy as np
from django.test import SimpleTestCase

from apps.herramientas.forecasting import validation
from apps.herramientas.forecasting.exceptions import (
    ForecastingError,
    InsufficientDataError,
    InvalidConfidenceLevelError,
    InvalidForecastHorizonError,
    InvalidOrderError,
    InvalidSeriesError,
)


class ValidarSerieTests(SimpleTestCase):
    def test_serie_correcta_se_convierte_a_ndarray_float(self):
        serie = validation.validar_serie([1, 2, 3.5, 4])
        self.assertIsInstance(serie, np.ndarray)
        self.assertEqual(serie.dtype, np.float64)
        self.assertEqual(list(serie), [1.0, 2.0, 3.5, 4.0])

    def test_serie_vacia_lanza_invalid_series_error(self):
        with self.assertRaises(InvalidSeriesError):
            validation.validar_serie([])

    def test_serie_con_texto_lanza_invalid_series_error(self):
        with self.assertRaises(InvalidSeriesError):
            validation.validar_serie([1, 2, "tres", 4])

    def test_serie_con_booleano_lanza_invalid_series_error(self):
        # bool es subclase de int en Python: debe rechazarse explicitamente,
        # no colarse como 0/1.
        with self.assertRaises(InvalidSeriesError):
            validation.validar_serie([1, 2, True, 4])

    def test_serie_con_nan_lanza_invalid_series_error(self):
        with self.assertRaises(InvalidSeriesError):
            validation.validar_serie([1, 2, float("nan"), 4])

    def test_serie_con_infinito_lanza_invalid_series_error(self):
        with self.assertRaises(InvalidSeriesError):
            validation.validar_serie([1, 2, float("inf"), 4])

    def test_serie_no_lista_lanza_invalid_series_error(self):
        with self.assertRaises(InvalidSeriesError):
            validation.validar_serie("1,2,3")


class ValidarSerieNoConstanteTests(SimpleTestCase):
    def test_serie_constante_lanza_invalid_series_error(self):
        serie = validation.validar_serie([7, 7, 7, 7, 7])
        with self.assertRaises(InvalidSeriesError):
            validation.validar_serie_no_constante(serie)

    def test_serie_variable_no_lanza_error(self):
        serie = validation.validar_serie([1, 2, 3, 2, 1])
        validation.validar_serie_no_constante(serie)  # no debe lanzar


class ValidarOrdenArimaTests(SimpleTestCase):
    def test_orden_valido_no_lanza(self):
        validation.validar_orden_arima(1, 1, 1)

    def test_p_negativo_lanza_invalid_order_error(self):
        with self.assertRaises(InvalidOrderError):
            validation.validar_orden_arima(-1, 0, 0)

    def test_d_negativo_lanza_invalid_order_error(self):
        with self.assertRaises(InvalidOrderError):
            validation.validar_orden_arima(0, -1, 0)

    def test_q_negativo_lanza_invalid_order_error(self):
        with self.assertRaises(InvalidOrderError):
            validation.validar_orden_arima(0, 0, -1)

    def test_orden_no_entero_lanza_invalid_order_error(self):
        with self.assertRaises(InvalidOrderError):
            validation.validar_orden_arima(1.5, 0, 0)

    def test_orden_booleano_lanza_invalid_order_error(self):
        with self.assertRaises(InvalidOrderError):
            validation.validar_orden_arima(True, 0, 0)

    def test_d_supera_maximo_admitido_lanza_invalid_order_error(self):
        with self.assertRaises(InvalidOrderError):
            validation.validar_orden_arima(0, 3, 0)

    def test_d_en_el_limite_maximo_no_lanza(self):
        validation.validar_orden_arima(0, 2, 0)


class ValidarMuestraMinimaTests(SimpleTestCase):
    def test_muestra_suficiente_no_lanza(self):
        minimo = validation.validar_muestra_minima(10, p=1, d=1, q=1)
        self.assertEqual(minimo, 6)

    def test_muestra_insuficiente_lanza_insufficient_data_error(self):
        with self.assertRaises(InsufficientDataError):
            validation.validar_muestra_minima(3, p=2, d=1, q=2)


class ValidarHorizontePronosticoTests(SimpleTestCase):
    def test_horizonte_valido_no_lanza(self):
        validation.validar_horizonte_pronostico(5)

    def test_horizonte_cero_lanza_invalid_forecast_horizon_error(self):
        with self.assertRaises(InvalidForecastHorizonError):
            validation.validar_horizonte_pronostico(0)

    def test_horizonte_negativo_lanza_invalid_forecast_horizon_error(self):
        with self.assertRaises(InvalidForecastHorizonError):
            validation.validar_horizonte_pronostico(-3)

    def test_horizonte_excede_maximo_lanza_invalid_forecast_horizon_error(self):
        with self.assertRaises(InvalidForecastHorizonError):
            validation.validar_horizonte_pronostico(51)

    def test_horizonte_no_entero_lanza_invalid_forecast_horizon_error(self):
        with self.assertRaises(InvalidForecastHorizonError):
            validation.validar_horizonte_pronostico(2.5)


class ValidarNivelConfianzaTests(SimpleTestCase):
    def test_nivel_valido_se_normaliza_a_float(self):
        self.assertEqual(validation.validar_nivel_confianza(0.95), 0.95)

    def test_nivel_debajo_del_minimo_lanza_invalid_confidence_level_error(self):
        with self.assertRaises(InvalidConfidenceLevelError):
            validation.validar_nivel_confianza(0.5)

    def test_nivel_encima_del_maximo_lanza_invalid_confidence_level_error(self):
        with self.assertRaises(InvalidConfidenceLevelError):
            validation.validar_nivel_confianza(1.0)

    def test_nivel_no_numerico_lanza_invalid_confidence_level_error(self):
        with self.assertRaises(InvalidConfidenceLevelError):
            validation.validar_nivel_confianza("0.95")


class ResolverTendenciaTests(SimpleTestCase):
    def test_d0_con_constante_usa_trend_c(self):
        trend, descripcion = validation.resolver_tendencia(0, True)
        self.assertEqual(trend, "c")
        self.assertIn("Constante", descripcion)

    def test_d1_con_constante_usa_trend_t_drift(self):
        trend, descripcion = validation.resolver_tendencia(1, True)
        self.assertEqual(trend, "t")
        self.assertIn("Drift", descripcion)

    def test_d2_no_admite_tendencia(self):
        trend, _descripcion = validation.resolver_tendencia(2, True)
        self.assertEqual(trend, "n")

    def test_sin_constante_siempre_usa_trend_n(self):
        for d in (0, 1, 2):
            trend, _descripcion = validation.resolver_tendencia(d, False)
            self.assertEqual(trend, "n")


class ExcepcionesDeDominioTests(SimpleTestCase):
    def test_todas_heredan_de_forecasting_error(self):
        subclases = [
            InvalidSeriesError,
            InvalidOrderError,
            InsufficientDataError,
            InvalidForecastHorizonError,
            InvalidConfidenceLevelError,
        ]
        for excepcion in subclases:
            self.assertTrue(issubclass(excepcion, ForecastingError))

    def test_cada_excepcion_tiene_codigo_error_propio_y_distinto(self):
        codigos = {
            InvalidSeriesError().codigo_error,
            InvalidOrderError().codigo_error,
            InsufficientDataError().codigo_error,
            InvalidForecastHorizonError().codigo_error,
            InvalidConfidenceLevelError().codigo_error,
        }
        self.assertEqual(len(codigos), 5)


class SerializacionJSONTests(SimpleTestCase):
    def test_serie_validada_es_json_serializable_via_tolist(self):
        serie = validation.validar_serie([1, 2, 3])
        # np.ndarray no es serializable directo; el patron correcto es .tolist().
        json.dumps(serie.tolist())

    def test_valores_no_finitos_no_deben_llegar_a_json(self):
        with self.assertRaises(InvalidSeriesError):
            validation.validar_serie([1, 2, math.inf])
