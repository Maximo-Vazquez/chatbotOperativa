"""Pruebas de `apps/herramientas/forecasting/diagnostics.py`: seleccion de
rezago de Ljung-Box, ejecucion de la prueba con `model_df`, clasificacion de
parametros y de advertencias de statsmodels. No ajustan modelos: operan
sobre residuos y mensajes de advertencia sinteticos.
"""

import warnings

import numpy as np
from django.test import SimpleTestCase

from apps.herramientas.forecasting import diagnostics


class SeleccionarLagLjungBoxTests(SimpleTestCase):
    def test_devuelve_lag_mayor_que_model_df_y_menor_que_cantidad_residuos(self):
        lag = diagnostics.seleccionar_lag_ljung_box(cantidad_residuos=30, model_df=2)
        self.assertIsNotNone(lag)
        self.assertGreater(lag, 2)
        self.assertLess(lag, 30)

    def test_devuelve_none_si_no_existe_lag_valido_por_muestra_pequena(self):
        # con muy pocos residuos y model_df alto, ningun lag cumple
        # simultaneamente lag > model_df y lag < cantidad_residuos.
        lag = diagnostics.seleccionar_lag_ljung_box(cantidad_residuos=4, model_df=5)
        self.assertIsNone(lag)

    def test_devuelve_none_con_cantidad_residuos_minima(self):
        lag = diagnostics.seleccionar_lag_ljung_box(cantidad_residuos=1, model_df=0)
        self.assertIsNone(lag)

    def test_no_excede_el_lag_maximo(self):
        lag = diagnostics.seleccionar_lag_ljung_box(cantidad_residuos=1000, model_df=0, lag_maximo=10)
        self.assertLessEqual(lag, 10)


class EjecutarLjungBoxTests(SimpleTestCase):
    def test_estructura_completa_cuando_es_ejecutable(self):
        rng = np.random.default_rng(0)
        residuos = rng.normal(size=40)
        resultado = diagnostics.ejecutar_ljung_box(residuos, model_df=2)

        self.assertTrue(resultado["ejecutado"])
        self.assertIn("estadistico", resultado)
        self.assertIn("p_value", resultado)
        self.assertIn("autocorrelacion_significativa", resultado)
        self.assertIn("es_ruido_blanco", resultado)
        self.assertEqual(resultado["model_df"], 2)
        self.assertGreater(resultado["lags"], 2)

    def test_no_ejecutado_cuando_no_hay_lag_valido(self):
        residuos = np.array([0.1, -0.2, 0.05])
        resultado = diagnostics.ejecutar_ljung_box(residuos, model_df=5)

        self.assertFalse(resultado["ejecutado"])
        self.assertIn("motivo", resultado)
        self.assertIsNone(resultado["estadistico"])
        self.assertIsNone(resultado["p_value"])

    def test_no_afirma_modelo_valido_solo_por_ausencia_de_autocorrelacion(self):
        rng = np.random.default_rng(0)
        residuos = rng.normal(size=40)
        resultado = diagnostics.ejecutar_ljung_box(residuos, model_df=2)

        self.assertNotIn("modelo valido", resultado["interpretacion"].lower())
        self.assertNotIn("modelo válido", resultado["interpretacion"].lower())


class ConstruirDiagnosticoResiduosTests(SimpleTestCase):
    def test_descarta_los_primeros_d_residuos(self):
        residuos = np.array([100.0, 200.0, 1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.0, 1.02])
        diagnostico = diagnostics.construir_diagnostico_residuos(residuos, d=2, p=1, q=0)

        self.assertEqual(diagnostico["cantidad_residuos"], 8)
        self.assertLess(diagnostico["mse"], 10)  # los outliers descartados no deben pesar

    def test_sin_diferenciacion_usa_todos_los_residuos(self):
        residuos = np.array([1.0, -1.0, 0.5, -0.5, 0.2, -0.2, 0.1, -0.1])
        diagnostico = diagnostics.construir_diagnostico_residuos(residuos, d=0, p=0, q=0)

        self.assertEqual(diagnostico["cantidad_residuos"], len(residuos))

    def test_agrega_advertencia_de_muestra_pequena(self):
        residuos = np.array([1.0, -1.0, 0.5, -0.5, 0.2])
        diagnostico = diagnostics.construir_diagnostico_residuos(residuos, d=0, p=0, q=0)

        codigos = [a["codigo"] for a in diagnostico["advertencias"]]
        self.assertIn("MUESTRA_PEQUENA_DIAGNOSTICO", codigos)


class ClasificarParametroTests(SimpleTestCase):
    def test_autorregresivo(self):
        self.assertEqual(diagnostics.clasificar_parametro("ar.L1"), "autorregresivo")

    def test_media_movil(self):
        self.assertEqual(diagnostics.clasificar_parametro("ma.L2"), "media_movil")

    def test_constante(self):
        self.assertEqual(diagnostics.clasificar_parametro("const"), "constante")

    def test_tendencia_drift(self):
        self.assertEqual(diagnostics.clasificar_parametro("x1"), "tendencia")

    def test_varianza(self):
        self.assertEqual(diagnostics.clasificar_parametro("sigma2"), "varianza")

    def test_otro_para_nombres_desconocidos(self):
        self.assertEqual(diagnostics.clasificar_parametro("beta_temperatura"), "otro")


class ClasificarAdvertenciasTests(SimpleTestCase):
    def _capturar(self, mensaje, categoria):
        with warnings.catch_warnings(record=True) as capturadas:
            warnings.simplefilter("always")
            warnings.warn(mensaje, category=categoria)
        return capturadas

    def test_convergencia_no_alcanzada(self):
        capturadas = self._capturar("Maximum Likelihood optimization failed to converge.", UserWarning)
        resultado = diagnostics.clasificar_advertencias(capturadas)
        self.assertEqual(resultado[0]["codigo"], "CONVERGENCIA_NO_ALCANZADA")

    def test_parametros_no_estacionarios(self):
        capturadas = self._capturar("Non-stationary starting autoregressive parameters found.", UserWarning)
        resultado = diagnostics.clasificar_advertencias(capturadas)
        self.assertEqual(resultado[0]["codigo"], "PARAMETROS_INICIALES_NO_ESTACIONARIOS")

    def test_parametros_no_invertibles(self):
        capturadas = self._capturar("Non-invertible starting MA parameters found.", UserWarning)
        resultado = diagnostics.clasificar_advertencias(capturadas)
        self.assertEqual(resultado[0]["codigo"], "PARAMETROS_INICIALES_NO_INVERTIBLES")

    def test_deduplica_advertencias_con_el_mismo_codigo(self):
        capturadas = self._capturar("Non-invertible starting MA parameters found.", UserWarning)
        capturadas += self._capturar("Non-invertible starting MA parameters found (bis).", UserWarning)
        resultado = diagnostics.clasificar_advertencias(capturadas)
        self.assertEqual(len(resultado), 1)

    def test_lista_vacia_no_lanza(self):
        self.assertEqual(diagnostics.clasificar_advertencias([]), [])
