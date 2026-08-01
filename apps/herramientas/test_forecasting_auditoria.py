"""Pruebas finales de robustez, integración y compatibilidad del forecasting."""

import json

import numpy as np
from django.test import SimpleTestCase

from apps.herramientas.forecasting import diagnostics
from apps.herramientas.tools import (
    TOOL_DEFINITIONS,
    TOOL_META,
    TOOL_REGISTRY,
    _to_json_safe,
    ejecutar_herramienta,
)


def _serie_estacional(n=40):
    indice = np.arange(n, dtype=float)
    return (30.0 + 0.25 * indice + 4.0 * np.sin(2 * np.pi * indice / 4)).tolist()


class ModeloARCompartidoTests(SimpleTestCase):
    def test_ar_preserva_contrato_y_agrega_diagnostico_comun(self):
        resultado = ejecutar_herramienta(
            "modelo_ar",
            {"valores": _serie_estacional(), "p": 1, "pasos_pronostico": 3},
        )

        for clave in (
            "modelo", "orden_p", "n_observaciones", "coeficientes", "aic",
            "bic", "mse_residuos", "media_residuos", "varianza_residuos",
            "ljung_box", "pasos_pronostico", "pronostico",
        ):
            self.assertIn(clave, resultado)
        self.assertEqual(resultado["representacion_interna"], "ARIMA(1,0,0)")
        self.assertEqual(resultado["diagnostico_residuos"]["ljung_box"]["model_df"], 1)
        self.assertEqual(len(resultado["intervalos_pronostico"]), 3)

    def test_ar_y_sarimax_equivalente_comparten_resultados(self):
        valores = _serie_estacional()
        ar = ejecutar_herramienta("modelo_ar", {"valores": valores, "p": 1})
        general = ejecutar_herramienta(
            "modelo_sarimax",
            {"valores": valores, "p": 1, "d": 0, "q": 0},
        )
        self.assertNotIn("error", ar)
        self.assertNotIn("error", general)
        self.assertAlmostEqual(ar["aic"], general["aic"], places=6)
        self.assertEqual(ar["pronostico"], general["pronostico"])

    def test_ar_rechaza_argumentos_invalidos_en_backend(self):
        casos = [
            {"valores": [1.0, 2.0, 3.0, 4.0], "p": True},
            {"valores": [1.0, 2.0, np.nan, 4.0], "p": 1},
            {"valores": [1.0, 2.0, 3.0, 4.0], "p": -1},
            {"valores": [1.0, 2.0, 3.0, 4.0], "p": 1, "pasos_pronostico": 0},
        ]
        for argumentos in casos:
            with self.subTest(argumentos=argumentos):
                resultado = ejecutar_herramienta("modelo_ar", argumentos)
                self.assertIn("codigo_error", resultado)
                self.assertNotIn("Traceback", resultado.get("error", ""))


class DiagnosticoYSerializacionTests(SimpleTestCase):
    def test_residuos_no_finitos_se_excluyen_de_forma_explicita(self):
        resultado = diagnostics.construir_diagnostico_residuos(
            np.array([0.1, np.nan, 0.2, np.inf, -0.1, 0.0]),
            d=0,
            p=0,
            q=0,
        )
        self.assertFalse(resultado["residuos_finitos"])
        self.assertEqual(resultado["residuos_no_finitos_excluidos"], 2)
        self.assertEqual(resultado["cantidad_residuos"], 4)
        self.assertTrue(any(
            advertencia["codigo"] == "RESIDUOS_NO_FINITOS"
            for advertencia in resultado["advertencias"]
        ))
        json.dumps(resultado, allow_nan=False)

    def test_serializador_reemplaza_nan_e_inf_por_null(self):
        resultado = _to_json_safe({
            "nan": float("nan"),
            "positivo": float("inf"),
            "negativo": np.float64("-inf"),
        })
        self.assertEqual(resultado, {"nan": None, "positivo": None, "negativo": None})
        json.dumps(resultado, allow_nan=False)


class RobustezVolatilidadTests(SimpleTestCase):
    def test_series_volatiles_y_outliers_no_exponen_fallos_crudos(self):
        rng = np.random.default_rng(20260728)
        base = np.cumsum(rng.normal(0.0, 1.0, 48)) + 100.0
        series = {
            "amplitud_elevada": (base * 1e6).tolist(),
            "cambio_nivel": (base + np.where(np.arange(48) >= 24, 80.0, 0.0)).tolist(),
            "outlier": np.where(np.arange(48) == 24, base * 8.0, base).tolist(),
            "varianza_cambiante": (
                100.0 + np.cumsum(rng.normal(0.0, np.linspace(0.5, 8.0, 48)))
            ).tolist(),
            "agrupamiento": (
                100.0 + np.cumsum(rng.normal(0.0, np.repeat([0.5, 6.0, 1.0], 16)))
            ).tolist(),
        }
        for nombre, valores in series.items():
            with self.subTest(nombre=nombre):
                resultado = ejecutar_herramienta(
                    "modelo_arima",
                    {"valores": valores, "p": 1, "d": 1, "q": 0, "pasos_pronostico": 2},
                )
                self.assertTrue("pronostico" in resultado or "codigo_error" in resultado)
                self.assertNotIn("Traceback", resultado.get("error", ""))
                json.dumps(resultado, allow_nan=False)


class IntegracionGlobalTests(SimpleTestCase):
    def test_registro_sin_colisiones_y_metadatos_completos(self):
        nombres = [
            definicion["function"]["name"]
            for definicion in TOOL_DEFINITIONS
        ]
        self.assertEqual(len(nombres), len(set(nombres)))
        esperadas = {
            "modelo_ar", "modelo_ma", "modelo_arima", "modelo_sarima",
            "modelo_arimax", "modelo_sarimax", "acf", "pacf",
            "modelo_dickey_fuller", "descomposicion_visualizacion_serie",
        }
        self.assertTrue(esperadas.issubset(TOOL_REGISTRY))
        self.assertTrue(esperadas.issubset(TOOL_META))

    def test_todas_las_herramientas_generan_json_estricto(self):
        valores = _serie_estacional()
        exog = {"temperatura": np.linspace(10.0, 20.0, len(valores)).tolist()}
        argumentos = {
            "acf": {"valores": valores, "lags": 6},
            "pacf": {"valores": valores, "lags": 6},
            "modelo_dickey_fuller": {"valores": valores},
            "descomposicion_visualizacion_serie": {
                "valores": valores, "frecuencia": 4,
            },
            "estabilizacion_media": {"valores": valores},
            "estabilizacion_varianza": {"valores": valores},
            "modelo_ar": {"valores": valores, "p": 1},
            "modelo_ma": {"valores": valores, "q": 1},
            "modelo_arima": {"valores": valores, "p": 1, "d": 0, "q": 0},
            "modelo_sarima": {
                "valores": valores, "p": 0, "d": 0, "q": 0,
                "P": 1, "D": 0, "Q": 0, "s": 4,
            },
            "modelo_arimax": {
                "valores": valores, "p": 0, "d": 0, "q": 0,
                "variables_exogenas_historicas": exog,
                "variables_exogenas_futuras": {"temperatura": [20.5]},
            },
            "modelo_sarimax": {
                "valores": valores, "p": 0, "d": 0, "q": 0,
                "P": 1, "D": 0, "Q": 0, "s": 4,
                "variables_exogenas_historicas": exog,
                "variables_exogenas_futuras": {"temperatura": [20.5]},
            },
        }
        self.assertEqual(set(argumentos), set(TOOL_REGISTRY))
        for nombre, entrada in argumentos.items():
            with self.subTest(herramienta=nombre):
                resultado = ejecutar_herramienta(nombre, entrada)
                self.assertNotIn("error", resultado)
                json.dumps(resultado, allow_nan=False)

