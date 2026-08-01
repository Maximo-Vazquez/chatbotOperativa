"""Pruebas de `apps/herramientas/forecasting/metrics.py`: MAE, RMSE y MAPE.

Funciones puras, sin Django ni statsmodels de por medio: se prueban con
valores conocidos y casos limite (vacio, longitudes distintas, NaN, Inf,
booleanos, ceros, negativos).
"""

import math

from django.test import SimpleTestCase

from apps.herramientas.forecasting import metrics
from apps.herramientas.forecasting.exceptions import (
    MetricCalculationError,
    MetricLengthMismatchError,
)


class CalcularMaeTests(SimpleTestCase):
    def test_valores_conocidos(self):
        mae = metrics.calcular_mae([10, 20, 30], [12, 18, 33])
        self.assertAlmostEqual(mae, (2 + 2 + 3) / 3, places=6)

    def test_prediccion_perfecta_es_cero(self):
        self.assertEqual(metrics.calcular_mae([1, 2, 3], [1, 2, 3]), 0.0)

    def test_valores_negativos(self):
        mae = metrics.calcular_mae([-5, -10, 0], [-3, -12, 1])
        self.assertAlmostEqual(mae, (2 + 2 + 1) / 3, places=6)

    def test_valores_cero(self):
        mae = metrics.calcular_mae([0, 0, 0], [1, -1, 2])
        self.assertAlmostEqual(mae, (1 + 1 + 2) / 3, places=6)

    def test_longitudes_distintas_lanza_error(self):
        with self.assertRaises(MetricLengthMismatchError):
            metrics.calcular_mae([1, 2, 3], [1, 2])

    def test_arreglo_vacio_lanza_error(self):
        with self.assertRaises(MetricCalculationError):
            metrics.calcular_mae([], [])

    def test_nan_lanza_error(self):
        with self.assertRaises(MetricCalculationError):
            metrics.calcular_mae([1, 2, math.nan], [1, 2, 3])

    def test_infinito_lanza_error(self):
        with self.assertRaises(MetricCalculationError):
            metrics.calcular_mae([1, 2, 3], [1, 2, math.inf])

    def test_booleano_lanza_error(self):
        with self.assertRaises(MetricCalculationError):
            metrics.calcular_mae([1, 2, True], [1, 2, 3])

    def test_resultado_es_float_nativo_de_python(self):
        resultado = metrics.calcular_mae([1, 2, 3], [1, 2, 4])
        self.assertIsInstance(resultado, float)
        self.assertNotIsInstance(resultado, bool)


class CalcularRmseTests(SimpleTestCase):
    def test_valores_conocidos(self):
        rmse = metrics.calcular_rmse([10, 20, 30], [12, 18, 33])
        esperado = math.sqrt((4 + 4 + 9) / 3)
        self.assertAlmostEqual(rmse, esperado, places=6)

    def test_prediccion_perfecta_es_cero(self):
        self.assertEqual(metrics.calcular_rmse([5, 6, 7], [5, 6, 7]), 0.0)

    def test_penaliza_mas_los_errores_grandes_que_mae(self):
        reales = [10, 10, 10, 10]
        pronosticados = [11, 11, 11, 20]  # un error grande, el resto chico
        mae = metrics.calcular_mae(reales, pronosticados)
        rmse = metrics.calcular_rmse(reales, pronosticados)
        self.assertGreater(rmse, mae)

    def test_valores_negativos(self):
        rmse = metrics.calcular_rmse([-1, -2, -3], [-2, -2, -1])
        self.assertGreater(rmse, 0)

    def test_longitudes_distintas_lanza_error(self):
        with self.assertRaises(MetricLengthMismatchError):
            metrics.calcular_rmse([1, 2, 3], [1, 2])

    def test_arreglo_vacio_lanza_error(self):
        with self.assertRaises(MetricCalculationError):
            metrics.calcular_rmse([], [])

    def test_nan_lanza_error(self):
        with self.assertRaises(MetricCalculationError):
            metrics.calcular_rmse([1, 2, math.nan], [1, 2, 3])

    def test_infinito_lanza_error(self):
        with self.assertRaises(MetricCalculationError):
            metrics.calcular_rmse([1, 2, 3], [1, 2, math.inf])

    def test_resultado_es_float_nativo_de_python(self):
        resultado = metrics.calcular_rmse([1, 2, 3], [1, 2, 4])
        self.assertIsInstance(resultado, float)


class CalcularMapeTests(SimpleTestCase):
    def test_valores_positivos_conocidos(self):
        resultado = metrics.calcular_mape([100, 200], [110, 190])
        # |10/100| + |10/200| = 0.10 + 0.05 -> promedio 0.075 -> 7.5%
        self.assertAlmostEqual(resultado["mape"], 7.5, places=4)
        self.assertTrue(resultado["mape_detalle"]["calculado"])

    def test_prediccion_perfecta_es_cero(self):
        resultado = metrics.calcular_mape([10, 20, 30], [10, 20, 30])
        self.assertEqual(resultado["mape"], 0.0)

    def test_un_valor_real_igual_a_cero_se_excluye(self):
        resultado = metrics.calcular_mape([0, 10, 20], [1, 11, 22])
        detalle = resultado["mape_detalle"]
        self.assertEqual(detalle["observaciones_totales"], 3)
        self.assertEqual(detalle["observaciones_excluidas_por_cero"], 1)
        self.assertEqual(detalle["observaciones_utilizadas"], 2)
        self.assertTrue(detalle["calculado"])

    def test_multiples_ceros_se_excluyen_todos(self):
        resultado = metrics.calcular_mape([0, 0, 10, 20], [1, -1, 11, 22])
        detalle = resultado["mape_detalle"]
        self.assertEqual(detalle["observaciones_excluidas_por_cero"], 2)
        self.assertEqual(detalle["observaciones_utilizadas"], 2)

    def test_todos_los_valores_reales_cero_devuelve_null(self):
        resultado = metrics.calcular_mape([0, 0, 0], [1, 2, 3])
        self.assertIsNone(resultado["mape"])
        self.assertFalse(resultado["mape_detalle"]["calculado"])
        self.assertIn("motivo", resultado["mape_detalle"])

    def test_valores_cercanos_a_cero_generan_advertencia_pero_se_calculan(self):
        resultado = metrics.calcular_mape([1e-8, 10, 20], [1e-8, 11, 22], tolerancia_cero=1e-6)
        detalle = resultado["mape_detalle"]
        self.assertTrue(detalle["calculado"])
        self.assertEqual(detalle["observaciones_excluidas_por_cero"], 0)
        codigos = [a["codigo"] for a in detalle["advertencias"]]
        self.assertIn("MAPE_VALORES_CERCANOS_A_CERO", codigos)

    def test_valores_negativos_generan_advertencia_pero_se_calculan(self):
        resultado = metrics.calcular_mape([-10, 20, 30], [-11, 19, 33])
        detalle = resultado["mape_detalle"]
        self.assertTrue(detalle["calculado"])
        codigos = [a["codigo"] for a in detalle["advertencias"]]
        self.assertIn("MAPE_VALORES_NEGATIVOS", codigos)

    def test_longitudes_distintas_lanza_error(self):
        with self.assertRaises(MetricLengthMismatchError):
            metrics.calcular_mape([1, 2, 3], [1, 2])

    def test_arreglo_vacio_lanza_error(self):
        with self.assertRaises(MetricCalculationError):
            metrics.calcular_mape([], [])

    def test_nan_lanza_error(self):
        with self.assertRaises(MetricCalculationError):
            metrics.calcular_mape([1, 2, math.nan], [1, 2, 3])

    def test_infinito_lanza_error(self):
        with self.assertRaises(MetricCalculationError):
            metrics.calcular_mape([1, 2, 3], [1, 2, math.inf])

    def test_advertencias_correctas_por_ceros_excluidos(self):
        resultado = metrics.calcular_mape([0, 10], [1, 11])
        codigos = [a["codigo"] for a in resultado["mape_detalle"]["advertencias"]]
        self.assertIn("MAPE_VALORES_CERO_EXCLUIDOS", codigos)

    def test_cantidad_de_observaciones_excluidas_es_exacta(self):
        resultado = metrics.calcular_mape([0, 0, 0, 10, 20], [1, 1, 1, 11, 22])
        self.assertEqual(resultado["mape_detalle"]["observaciones_excluidas_por_cero"], 3)

    def test_no_devuelve_infinito(self):
        resultado = metrics.calcular_mape([100, 200], [100, 200])
        self.assertNotEqual(resultado["mape"], math.inf)
        if resultado["mape"] is not None:
            self.assertTrue(math.isfinite(resultado["mape"]))
