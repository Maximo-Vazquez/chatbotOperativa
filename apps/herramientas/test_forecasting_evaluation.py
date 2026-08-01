"""Pruebas de `apps/herramientas/forecasting/evaluation.py`: seleccion del
tamano de prueba y el holdout temporal generico (division cronologica +
metricas). Usa una `funcion_pronostico` de prueba (no ARIMA real) para
verificar el flujo de division/metrica de forma aislada y rapida; las
pruebas de evaluacion con ARIMA real estan en `test_modelo_arima.py`.
"""

import numpy as np
from django.test import SimpleTestCase

from apps.herramientas.forecasting import evaluation
from apps.herramientas.forecasting.exceptions import InvalidEvaluationConfigurationError


class DeterminarTamanoPruebaTests(SimpleTestCase):
    def test_cantidad_prueba_valida(self):
        self.assertEqual(evaluation.determinar_tamano_prueba(20, 4, None), 4)

    def test_porcentaje_prueba_valido(self):
        self.assertEqual(evaluation.determinar_tamano_prueba(20, None, 0.25), 5)

    def test_prioridad_de_cantidad_sobre_porcentaje(self):
        self.assertEqual(evaluation.determinar_tamano_prueba(20, 3, 0.4), 3)

    def test_cantidad_cero_lanza_error(self):
        with self.assertRaises(InvalidEvaluationConfigurationError):
            evaluation.determinar_tamano_prueba(20, 0, None)

    def test_cantidad_negativa_lanza_error(self):
        with self.assertRaises(InvalidEvaluationConfigurationError):
            evaluation.determinar_tamano_prueba(20, -2, None)

    def test_cantidad_mayor_que_la_serie_lanza_error(self):
        with self.assertRaises(InvalidEvaluationConfigurationError):
            evaluation.determinar_tamano_prueba(20, 20, None)

    def test_cantidad_igual_a_la_serie_lanza_error(self):
        with self.assertRaises(InvalidEvaluationConfigurationError):
            evaluation.determinar_tamano_prueba(10, 10, None)

    def test_porcentaje_cero_lanza_error(self):
        with self.assertRaises(InvalidEvaluationConfigurationError):
            evaluation.determinar_tamano_prueba(20, None, 0.0)

    def test_porcentaje_negativo_lanza_error(self):
        with self.assertRaises(InvalidEvaluationConfigurationError):
            evaluation.determinar_tamano_prueba(20, None, -0.1)

    def test_porcentaje_excesivo_lanza_error(self):
        with self.assertRaises(InvalidEvaluationConfigurationError):
            evaluation.determinar_tamano_prueba(20, None, 0.5)

    def test_configuracion_predeterminada_usa_20_por_ciento(self):
        # Sin cantidad ni porcentaje: 20% de 20 = 4.
        self.assertEqual(evaluation.determinar_tamano_prueba(20, None, None), 4)

    def test_configuracion_predeterminada_nunca_deja_cero_de_prueba(self):
        self.assertGreaterEqual(evaluation.determinar_tamano_prueba(3, None, None), 1)

    def test_serie_minima_compatible_deja_al_menos_una_observacion_de_entrenamiento(self):
        n_prueba = evaluation.determinar_tamano_prueba(2, None, None)
        self.assertLess(n_prueba, 2)

    def test_cantidad_no_entera_lanza_error(self):
        with self.assertRaises(InvalidEvaluationConfigurationError):
            evaluation.determinar_tamano_prueba(20, 2.5, None)

    def test_porcentaje_no_numerico_lanza_error(self):
        with self.assertRaises(InvalidEvaluationConfigurationError):
            evaluation.determinar_tamano_prueba(20, None, "0.2")


class EvaluarHoldoutTemporalTests(SimpleTestCase):
    def _funcion_pronostico_constante(self, valor):
        def _pronosticar(entrenamiento, pasos):
            return [valor] * pasos
        return _pronosticar

    def test_usa_las_ultimas_observaciones_como_prueba(self):
        serie = np.arange(20, dtype=float)
        capturado = {}

        def _pronosticar(entrenamiento, pasos):
            capturado["entrenamiento"] = entrenamiento.copy()
            return [0.0] * pasos

        evaluation.evaluar_holdout_temporal(
            serie=serie, minimo_observaciones_entrenamiento=3,
            funcion_pronostico=_pronosticar, cantidad_prueba=5,
        )
        # El entrenamiento debe ser el tramo inicial (las primeras 15 obs).
        np.testing.assert_array_equal(capturado["entrenamiento"], serie[:15])

    def test_orden_preservado_sin_mezcla_aleatoria(self):
        serie = np.arange(20, dtype=float)
        resultado = evaluation.evaluar_holdout_temporal(
            serie=serie, minimo_observaciones_entrenamiento=3,
            funcion_pronostico=self._funcion_pronostico_constante(0.0),
            cantidad_prueba=5,
        )
        # Los "valores_reales" de prueba deben ser exactamente el tramo final
        # en orden, no una muestra aleatoria de la serie.
        self.assertEqual(resultado["valores_reales"], [15.0, 16.0, 17.0, 18.0, 19.0])

    def test_entrenamiento_insuficiente_omite_evaluacion_sin_lanzar(self):
        serie = np.arange(5, dtype=float)
        resultado = evaluation.evaluar_holdout_temporal(
            serie=serie, minimo_observaciones_entrenamiento=10,
            funcion_pronostico=self._funcion_pronostico_constante(0.0),
            cantidad_prueba=1,
        )
        self.assertFalse(resultado["ejecutada"])
        self.assertIn("motivo", resultado)

    def test_configuracion_invalida_se_propaga_como_excepcion(self):
        serie = np.arange(20, dtype=float)
        with self.assertRaises(InvalidEvaluationConfigurationError):
            evaluation.evaluar_holdout_temporal(
                serie=serie, minimo_observaciones_entrenamiento=3,
                funcion_pronostico=self._funcion_pronostico_constante(0.0),
                porcentaje_prueba=0.9,
            )

    def test_metricas_prueba_incluyen_mae_rmse_mape(self):
        serie = np.array([10.0, 10.0, 10.0, 10.0, 12.0])
        resultado = evaluation.evaluar_holdout_temporal(
            serie=serie, minimo_observaciones_entrenamiento=1,
            funcion_pronostico=self._funcion_pronostico_constante(10.0),
            cantidad_prueba=1,
        )
        self.assertTrue(resultado["ejecutada"])
        for clave in ("mae", "rmse", "mape"):
            self.assertIn(clave, resultado["metricas_prueba"])
        self.assertAlmostEqual(resultado["metricas_prueba"]["mae"], 2.0, places=6)

    def test_resultado_es_json_serializable(self):
        import json

        serie = np.arange(20, dtype=float)
        resultado = evaluation.evaluar_holdout_temporal(
            serie=serie, minimo_observaciones_entrenamiento=3,
            funcion_pronostico=self._funcion_pronostico_constante(0.0),
            cantidad_prueba=5,
        )
        json.dumps(resultado)
