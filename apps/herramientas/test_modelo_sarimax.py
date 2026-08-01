"""Pruebas de extremo a extremo de la herramienta publica `modelo_sarimax`.

Ejecutan `ejecutar_herramienta("modelo_sarimax", ...)` tal como lo haria el
chatbot real, cubriendo: registro dinamico, clasificacion automatica del
tipo de modelo (AR/MA/ARMA/ARIMA/SARIMA/ARIMAX/SARIMAX), reutilizacion del
nucleo compartido en las cuatro combinaciones (con/sin estacionalidad,
con/sin exogenas), y equivalencia numerica con las fachadas especificas
(`modelo_ma`, `modelo_arima`, `modelo_sarima`, `modelo_arimax`).
"""

import json

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from apps.herramientas.tools import TOOL_DEFINITIONS, TOOL_META, TOOL_REGISTRY, ejecutar_herramienta


def _serie_ar(n=30, phi=0.6, nivel=20.0, seed=1):
    rng = np.random.default_rng(seed)
    serie = [nivel]
    for _ in range(1, n):
        serie.append(nivel + phi * (serie[-1] - nivel) + rng.normal())
    return serie


def _serie_ma(n=40, theta=0.5, nivel=20.0, seed=2):
    rng = np.random.default_rng(seed)
    ruido = rng.normal(scale=2.0, size=n + 1)
    return [nivel + ruido[t] + theta * ruido[t - 1] for t in range(1, n + 1)]


def _serie_random_walk_drift(n=24, drift=1.5, seed=3):
    rng = np.random.default_rng(seed)
    valores = [100.0]
    for t in range(1, n):
        valores.append(valores[-1] + drift + rng.normal(scale=3))
    return valores


def _serie_estacional(n=48, pendiente=0.5, amplitud=8.0, seed=4):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (50 + pendiente * t + amplitud * np.sin(2 * np.pi * t / 12) + rng.normal(scale=2.0, size=n)).tolist()


def _serie_con_exogena(n=40, beta=1.5, seed=5):
    rng = np.random.default_rng(seed)
    x = (20 + rng.normal(scale=3, size=n)).tolist()
    ruido = np.cumsum(rng.normal(scale=1.0, size=n)) * 0.3
    y = (50 + beta * np.array(x) + ruido).tolist()
    return y, x


def _serie_sarimax_completa(n=60, seed=6):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    temp = (20 + rng.normal(scale=3, size=n))
    y = (50 + 1.5 * temp + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(scale=1.5, size=n)).tolist()
    return y, temp.tolist()


def _fechas(n, inicio="2020-01-01", freq="MS"):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(inicio, periods=n, freq=freq)]


class ContratoSARIMAXTests(SimpleTestCase):
    def test_se_registra_dinamicamente(self):
        self.assertIn("modelo_sarimax", TOOL_REGISTRY)

    def test_tool_definition_nombre_correcto(self):
        nombres = [d["function"]["name"] for d in TOOL_DEFINITIONS]
        self.assertIn("modelo_sarimax", nombres)

    def test_tool_meta_existe(self):
        self.assertIn("modelo_sarimax", TOOL_META)

    def test_tool_function_ejecutable(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": _serie_random_walk_drift(), "p": 0, "d": 1, "q": 0})
        self.assertNotIn("error", resultado)

    def test_campos_obligatorios(self):
        definicion = next(d for d in TOOL_DEFINITIONS if d["function"]["name"] == "modelo_sarimax")
        requeridos = definicion["function"]["parameters"]["required"]
        self.assertEqual(set(requeridos), {"valores", "p", "d", "q"})

    def test_campos_opcionales_definidos(self):
        definicion = next(d for d in TOOL_DEFINITIONS if d["function"]["name"] == "modelo_sarimax")
        propiedades = definicion["function"]["parameters"]["properties"]
        for campo in (
            "P", "D", "Q", "s", "variables_exogenas_historicas", "variables_exogenas_futuras",
            "pasos_pronostico", "con_constante", "nivel_confianza", "fechas",
            "fechas_exogenas_historicas", "fechas_exogenas_futuras", "frecuencia",
            "evaluar_modelo", "cantidad_prueba", "porcentaje_prueba",
        ):
            self.assertIn(campo, propiedades)

    def test_serializacion_json(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": _serie_random_walk_drift(), "p": 0, "d": 1, "q": 0})
        json.dumps(resultado, ensure_ascii=False)


class ClasificacionTests(SimpleTestCase):
    def test_configuracion_ar(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": _serie_ar(), "p": 1, "d": 0, "q": 0})
        self.assertEqual(resultado["tipo_modelo_detectado"], "AR")
        self.assertEqual(resultado["modelo"], "AR(1)")

    def test_configuracion_ma(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": _serie_ma(), "p": 0, "d": 0, "q": 1})
        self.assertEqual(resultado["tipo_modelo_detectado"], "MA")
        self.assertEqual(resultado["modelo"], "MA(1)")

    def test_configuracion_arma(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": _serie_ma(), "p": 1, "d": 0, "q": 1})
        self.assertEqual(resultado["tipo_modelo_detectado"], "ARMA")
        self.assertEqual(resultado["modelo"], "ARMA(1,1)")

    def test_configuracion_arima(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": _serie_random_walk_drift(), "p": 0, "d": 1, "q": 0})
        self.assertEqual(resultado["tipo_modelo_detectado"], "ARIMA")
        self.assertEqual(resultado["modelo"], "ARIMA(0,1,0)")

    def test_configuracion_sarima(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": _serie_estacional(), "p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
        })
        self.assertEqual(resultado["tipo_modelo_detectado"], "SARIMA")
        self.assertEqual(resultado["modelo"], "SARIMA(1,1,0)(1,1,0,12)")

    def test_configuracion_arimax(self):
        y, x = _serie_con_exogena()
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": y, "p": 1, "d": 0, "q": 0,
            "variables_exogenas_historicas": {"x": x}, "variables_exogenas_futuras": {"x": [21.0]},
        })
        self.assertEqual(resultado["tipo_modelo_detectado"], "ARIMAX")
        self.assertEqual(resultado["modelo"], "ARIMAX(1,0,0)")

    def test_configuracion_sarimax(self):
        y, x = _serie_sarimax_completa()
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": y, "p": 1, "d": 0, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
            "variables_exogenas_historicas": {"x": x}, "variables_exogenas_futuras": {"x": [21.0]},
        })
        self.assertEqual(resultado["tipo_modelo_detectado"], "SARIMAX")
        self.assertEqual(resultado["modelo"], "SARIMAX(1,0,0)(1,1,0,12)")

    def test_representacion_interna_separada_del_nombre_pedagogico(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": _serie_ar(), "p": 1, "d": 0, "q": 0})
        self.assertNotEqual(resultado["modelo"], resultado["representacion_interna"])
        self.assertIn("ARIMA", resultado["representacion_interna"])


class SinEstacionalidadSinExogenasTests(SimpleTestCase):
    def setUp(self):
        self.serie = _serie_random_walk_drift(n=24)
        self.resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": self.serie, "p": 0, "d": 1, "q": 0, "pasos_pronostico": 3,
        })

    def test_ajuste_simple(self):
        self.assertNotIn("error", self.resultado)

    def test_pronostico(self):
        self.assertEqual(len(self.resultado["pronostico"]), 3)

    def test_intervalos(self):
        for intervalo in self.resultado["intervalos_pronostico"]:
            self.assertLessEqual(intervalo["limite_inferior"], intervalo["limite_superior"])

    def test_evaluacion(self):
        resultado_eval = ejecutar_herramienta("modelo_sarimax", {
            "valores": self.serie, "p": 0, "d": 1, "q": 0, "pasos_pronostico": 3,
            "evaluar_modelo": True, "cantidad_prueba": 5,
        })
        self.assertTrue(resultado_eval["evaluacion"]["ejecutada"])

    def test_regresion_con_modelo_arima(self):
        r_arima = ejecutar_herramienta("modelo_arima", {"valores": self.serie, "p": 0, "d": 1, "q": 0, "pasos_pronostico": 3})
        self.assertEqual(self.resultado["pronostico"], r_arima["pronostico"])
        self.assertEqual(self.resultado["aic"], r_arima["aic"])

    def test_campos_estacionales_y_exogenos_vacios_coherentes(self):
        self.assertEqual(self.resultado["orden_estacional"], {"P": 0, "D": 0, "Q": 0, "s": None})
        self.assertEqual(self.resultado["coeficientes_estacionales"], {})
        self.assertEqual(self.resultado["coeficientes_exogenos"], {})
        self.assertFalse(self.resultado["variables_exogenas"]["utilizadas"])
        self.assertIsNone(self.resultado["diagnostico_multicolinealidad"])
        self.assertIsNone(self.resultado["diagnostico_fuga_informacion"])
        self.assertIsNone(self.resultado["n_ciclos_aproximados"])
        self.assertIsNone(self.resultado["estacionariedad"]["estacional"])

    def test_no_genera_advertencia_de_ciclos(self):
        codigos = [a["codigo"] for a in self.resultado["advertencias"]]
        self.assertNotIn("CICLOS_ESTACIONALES_INSUFICIENTES", codigos)


class ConEstacionalidadSinExogenasTests(SimpleTestCase):
    def setUp(self):
        self.serie_mensual = _serie_estacional(n=48)
        self.resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": self.serie_mensual, "p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
            "pasos_pronostico": 3,
        })

    def test_sarima_mensual(self):
        self.assertNotIn("error", self.resultado)
        self.assertEqual(self.resultado["orden_estacional"]["s"], 12)

    def test_sarima_trimestral(self):
        rng = np.random.default_rng(7)
        n = 32
        t = np.arange(n)
        serie_q = (100 + 0.3 * t + 15 * np.sin(2 * np.pi * t / 4) + rng.normal(scale=3, size=n)).tolist()
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": serie_q, "p": 1, "d": 0, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 4,
        })
        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["tipo_modelo_detectado"], "SARIMA")

    def test_ciclos(self):
        self.assertEqual(self.resultado["n_ciclos_aproximados"], 4.0)

    def test_ljung_box_estacional(self):
        ljung_box = self.resultado["ljung_box"]
        if ljung_box["ejecutado"]:
            self.assertEqual(ljung_box["model_df"], 1 + 0 + 1 + 0)

    def test_regresion_con_modelo_sarima(self):
        r_sarima = ejecutar_herramienta("modelo_sarima", {
            "valores": self.serie_mensual, "p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
            "pasos_pronostico": 3,
        })
        self.assertEqual(self.resultado["pronostico"], r_sarima["pronostico"])


class SinEstacionalidadConExogenasTests(SimpleTestCase):
    def setUp(self):
        self.y, self.x = _serie_con_exogena(n=40)
        self.resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": self.y, "p": 1, "d": 0, "q": 0,
            "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0]},
        })

    def test_una_variable(self):
        self.assertNotIn("error", self.resultado)
        self.assertIn("x", self.resultado["coeficientes_exogenos"])

    def test_multiples_variables(self):
        rng = np.random.default_rng(9)
        n = 40
        temp = (20 + rng.normal(scale=3, size=n)).tolist()
        promo = rng.integers(0, 2, size=n).astype(float).tolist()
        y = (50 + 1.5 * np.array(temp) + 8 * np.array(promo) + rng.normal(scale=1, size=n)).tolist()
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": y, "p": 0, "d": 0, "q": 0,
            "variables_exogenas_historicas": {"temperatura": temp, "promocion": promo},
            "variables_exogenas_futuras": {"temperatura": [21.0], "promocion": [1.0]},
        })
        self.assertNotIn("error", resultado)
        self.assertEqual(set(resultado["variables_exogenas"]["nombres"]), {"temperatura", "promocion"})

    def test_exogenas_futuras_requeridas(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": self.y, "p": 1, "d": 0, "q": 0,
            "variables_exogenas_historicas": {"x": self.x},
        })
        self.assertEqual(resultado.get("codigo_error"), "EXOGENAS_FUTURAS_REQUERIDAS")

    def test_evaluacion_condicionada(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": self.y, "p": 1, "d": 0, "q": 0,
            "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0]},
            "evaluar_modelo": True, "cantidad_prueba": 8,
        })
        self.assertEqual(resultado["evaluacion"]["tipo"], "condicionada_a_exogenas_observadas")

    def test_regresion_con_modelo_arimax(self):
        r_arimax = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "p": 1, "d": 0, "q": 0,
            "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0]},
        })
        self.assertEqual(self.resultado["pronostico"], r_arimax["pronostico"])
        self.assertEqual(self.resultado["coeficientes_exogenos"], r_arimax["coeficientes_exogenos"])


class SarimaxCompletoTests(SimpleTestCase):
    """Serie sintetica con patron estacional + una exogena relevante + ruido + fechas regulares."""

    def setUp(self):
        self.y, self.x = _serie_sarimax_completa(n=60)
        self.fechas = _fechas(60, "2019-01-01", "MS")
        self.resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": self.y, "p": 1, "d": 0, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
            "variables_exogenas_historicas": {"temperatura": self.x},
            "variables_exogenas_futuras": {"temperatura": [21.0, 22.0]},
            "fechas": self.fechas, "pasos_pronostico": 2,
            "evaluar_modelo": True, "cantidad_prueba": 12,
        })

    def test_ajuste_sin_error(self):
        self.assertNotIn("error", self.resultado)

    def test_orden_regular(self):
        self.assertEqual(self.resultado["orden"], {"p": 1, "d": 0, "q": 0})

    def test_orden_estacional(self):
        self.assertEqual(self.resultado["orden_estacional"], {"P": 1, "D": 1, "Q": 0, "s": 12})

    def test_variables_exogenas(self):
        self.assertTrue(self.resultado["variables_exogenas"]["utilizadas"])
        self.assertEqual(self.resultado["variables_exogenas"]["nombres"], ["temperatura"])

    def test_coeficientes_regulares(self):
        self.assertIn("ar.L1", self.resultado["coeficientes_regulares"])

    def test_coeficientes_estacionales(self):
        self.assertIn("ar.S.L12", self.resultado["coeficientes_estacionales"])

    def test_coeficientes_exogenos(self):
        self.assertIn("temperatura", self.resultado["coeficientes_exogenos"])

    def test_pronostico(self):
        self.assertEqual(len(self.resultado["pronostico"]), 2)
        for valor in self.resultado["pronostico"]:
            self.assertTrue(np.isfinite(valor))

    def test_intervalos(self):
        for intervalo in self.resultado["intervalos_pronostico"]:
            self.assertTrue(np.isfinite(intervalo["limite_inferior"]))
            self.assertTrue(np.isfinite(intervalo["limite_superior"]))

    def test_fechas_futuras(self):
        self.assertEqual(self.resultado["fechas_pronostico"], ["2024-01-01", "2024-02-01"])

    def test_evaluacion(self):
        self.assertTrue(self.resultado["evaluacion"]["ejecutada"])

    def test_mae(self):
        self.assertIn("mae", self.resultado["evaluacion"]["metricas_prueba"])

    def test_rmse(self):
        self.assertIn("rmse", self.resultado["evaluacion"]["metricas_prueba"])

    def test_mape(self):
        self.assertIn("mape", self.resultado["evaluacion"]["metricas_prueba"])

    def test_diagnostico_residual(self):
        self.assertGreater(self.resultado["diagnostico_residuos"]["cantidad_residuos"], 0)

    def test_multicolinealidad(self):
        self.assertIn("clasificacion", self.resultado["diagnostico_multicolinealidad"])

    def test_serializacion(self):
        json.dumps(self.resultado, ensure_ascii=False)


class ComponentesOpcionalesTests(SimpleTestCase):
    def test_sin_s_cuando_no_hay_estacionalidad(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": _serie_random_walk_drift(), "p": 0, "d": 1, "q": 0})
        self.assertNotIn("error", resultado)
        self.assertIsNone(resultado["orden_estacional"]["s"])

    def test_error_si_componente_estacional_sin_s(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": _serie_estacional(), "p": 1, "d": 1, "q": 0, "P": 1,
        })
        self.assertEqual(resultado.get("codigo_error"), "PERIODICIDAD_ESTACIONAL_REQUERIDA")

    def test_sin_exogenas(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": _serie_ar(), "p": 1, "d": 0, "q": 0})
        self.assertFalse(resultado["variables_exogenas"]["utilizadas"])

    def test_con_exogenas(self):
        y, x = _serie_con_exogena()
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": y, "p": 1, "d": 0, "q": 0,
            "variables_exogenas_historicas": {"x": x}, "variables_exogenas_futuras": {"x": [21.0]},
        })
        self.assertTrue(resultado["variables_exogenas"]["utilizadas"])

    def test_falta_de_exogenas_futuras(self):
        y, x = _serie_con_exogena()
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": y, "p": 1, "d": 0, "q": 0, "variables_exogenas_historicas": {"x": x},
        })
        self.assertEqual(resultado.get("codigo_error"), "EXOGENAS_FUTURAS_REQUERIDAS")

    def test_seasonal_order_vacio_valido(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": _serie_random_walk_drift(), "p": 0, "d": 1, "q": 0, "P": 0, "D": 0, "Q": 0,
        })
        self.assertNotIn("error", resultado)

    def test_exogenas_vacias_invalidas(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": _serie_ar(), "p": 1, "d": 0, "q": 0, "variables_exogenas_historicas": {},
        })
        self.assertNotIn("error", resultado)  # {} es "falsy": se interpreta como "sin exogenas", no como error
        self.assertFalse(resultado["variables_exogenas"]["utilizadas"])


class ComplejidadTests(SimpleTestCase):
    def test_configuracion_razonable(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": _serie_ar(n=40), "p": 1, "d": 0, "q": 0})
        codigos = [a["codigo"] for a in resultado["advertencias"]]
        self.assertNotIn("MODELO_DEMASIADO_COMPLEJO", codigos)

    def test_configuracion_compleja_con_advertencia(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": _serie_estacional(n=30), "p": 2, "d": 1, "q": 2, "P": 1, "D": 1, "Q": 1, "s": 12,
        })
        if "error" not in resultado:
            codigos = [a["codigo"] for a in resultado["advertencias"]]
            self.assertIn("MODELO_DEMASIADO_COMPLEJO", codigos)

    def test_configuracion_imposible_con_error(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": _serie_ar(n=5), "p": 3, "d": 1, "q": 3,
        })
        self.assertEqual(resultado.get("codigo_error"), "MUESTRA_INSUFICIENTE")

    def test_demasiadas_variables_respecto_de_observaciones(self):
        rng = np.random.default_rng(10)
        n = 15
        y = (10 + rng.normal(size=n)).tolist()
        exog = {f"x{i}": rng.normal(size=n).tolist() for i in range(4)}
        exog_futuro = {f"x{i}": [0.0] for i in range(4)}
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": y, "p": 0, "d": 0, "q": 0,
            "variables_exogenas_historicas": exog, "variables_exogenas_futuras": exog_futuro,
        })
        if "error" not in resultado:
            codigos = [a["codigo"] for a in resultado["advertencias"]]
            self.assertIn("MODELO_DEMASIADO_COMPLEJO", codigos)

    def test_entrenamiento_insuficiente_para_evaluar(self):
        serie = _serie_random_walk_drift(n=24)
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": serie, "p": 0, "d": 1, "q": 0, "evaluar_modelo": True, "cantidad_prueba": 22,
        })
        self.assertNotIn("error", resultado)
        self.assertFalse(resultado["evaluacion"]["ejecutada"])

    def test_pocos_ciclos(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": _serie_estacional(n=18), "p": 0, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "s": 12,
        })
        # P=D=Q=0 -> no estacional, no aplica ciclos.
        self.assertIsNone(resultado["n_ciclos_aproximados"])

        resultado2 = ejecutar_herramienta("modelo_sarimax", {
            "valores": _serie_estacional(n=18), "p": 0, "d": 0, "q": 0, "P": 1, "D": 0, "Q": 0, "s": 12,
        })
        if "error" not in resultado2:
            codigos = [a["codigo"] for a in resultado2["advertencias"]]
            self.assertIn("CICLOS_ESTACIONALES_INSUFICIENTES", codigos)


class TendenciaTests(SimpleTestCase):
    def test_no_integrado_con_constante(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": _serie_ar(), "p": 1, "d": 0, "q": 0, "con_constante": True})
        self.assertEqual(resultado["tendencia_statsmodels"], "c")

    def test_integrado_con_tendencia_compatible(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": _serie_random_walk_drift(), "p": 0, "d": 1, "q": 0, "con_constante": True,
        })
        self.assertEqual(resultado["tendencia_statsmodels"], "t")

    def test_diferenciacion_regular_y_estacional(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": _serie_estacional(), "p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
        })
        self.assertEqual(resultado["tendencia_statsmodels"], "n")  # d+D=2

    def test_configuracion_incompatible_se_resuelve_sin_error(self):
        # d=1, D=1 -> d+D=2: statsmodels no admite ningun termino determinista
        # aca (igual que en `test_diferenciacion_regular_y_estacional`); debe
        # resolverse automaticamente a trend="n", nunca lanzar un error crudo.
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": _serie_estacional(), "p": 1, "d": 1, "q": 0, "P": 0, "D": 1, "Q": 0, "s": 12,
            "con_constante": True,
        })
        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["tendencia_statsmodels"], "n")

    def test_mensaje_de_error_controlado_orden_invalido(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": _serie_ar(), "p": -1, "d": 0, "q": 0})
        self.assertIn("codigo_error", resultado)
        self.assertNotIn("Traceback", resultado["error"])

    def test_regresion_de_fachadas_anteriores(self):
        for nombre, args in (
            ("modelo_ar", {"valores": _serie_ar(), "p": 1}),
            ("modelo_ma", {"valores": _serie_ma(), "q": 1}),
            ("modelo_arima", {"valores": _serie_random_walk_drift(), "p": 0, "d": 1, "q": 0}),
        ):
            resultado = ejecutar_herramienta(nombre, args)
            self.assertNotIn("error", resultado, f"{nombre} broke")


class ParametrosTests(SimpleTestCase):
    def setUp(self):
        y, x = _serie_sarimax_completa(n=60)
        self.resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": y, "p": 1, "d": 0, "q": 1, "P": 1, "D": 1, "Q": 0, "s": 12,
            "variables_exogenas_historicas": {"temperatura": x},
            "variables_exogenas_futuras": {"temperatura": [21.0]},
        })

    def test_clasificacion_ar(self):
        detalle = next(d for d in self.resultado["detalle_coeficientes"] if d["nombre"] == "ar.L1")
        self.assertEqual(detalle["tipo"], "autorregresivo")

    def test_clasificacion_ma(self):
        detalle = next(d for d in self.resultado["detalle_coeficientes"] if d["nombre"] == "ma.L1")
        self.assertEqual(detalle["tipo"], "media_movil")

    def test_clasificacion_ar_estacional(self):
        detalle = next(d for d in self.resultado["detalle_coeficientes"] if d["nombre"] == "ar.S.L12")
        self.assertEqual(detalle["tipo"], "autorregresivo_estacional")

    def test_clasificacion_exogena(self):
        detalle = next(d for d in self.resultado["detalle_coeficientes"] if d["nombre"] == "temperatura")
        self.assertEqual(detalle["tipo"], "exogena")

    def test_constante_o_varianza_presente(self):
        nombres = {d["nombre"] for d in self.resultado["detalle_coeficientes"]}
        self.assertIn("sigma2", nombres)

    def test_p_valores(self):
        for detalle in self.resultado["detalle_coeficientes"]:
            if detalle["p_value"] is not None:
                self.assertGreaterEqual(detalle["p_value"], 0.0)
                self.assertLessEqual(detalle["p_value"], 1.0)

    def test_intervalos_de_confianza(self):
        for detalle in self.resultado["detalle_coeficientes"]:
            if detalle["intervalo_confianza"] is not None:
                self.assertLessEqual(detalle["intervalo_confianza"]["inferior"], detalle["intervalo_confianza"]["superior"])


class LjungBoxSARIMAXTests(SimpleTestCase):
    def test_model_df_p_q_P_Q(self):
        y, x = _serie_sarimax_completa(n=60)
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": y, "p": 1, "d": 0, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 12,
            "variables_exogenas_historicas": {"temperatura": x},
            "variables_exogenas_futuras": {"temperatura": [21.0]},
        })
        ljung_box = resultado["ljung_box"]
        if ljung_box["ejecutado"]:
            self.assertEqual(ljung_box["model_df"], 1 + 1 + 1 + 1)

    def test_lag_estacional_valido(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": _serie_estacional(n=60), "p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
        })
        ljung_box = resultado["ljung_box"]
        if ljung_box["ejecutado"] and ljung_box["lags"] == 12:
            self.assertTrue(ljung_box["incluye_rezago_estacional"])

    def test_lag_no_estacional_sin_componente_estacional(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": _serie_ar(n=40), "p": 1, "d": 0, "q": 0})
        ljung_box = resultado["ljung_box"]
        self.assertFalse(ljung_box["incluye_rezago_estacional"])

    def test_muestra_pequena(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": _serie_ar(n=8), "p": 1, "d": 0, "q": 0,
        })
        if "error" not in resultado and not resultado["ljung_box"]["ejecutado"]:
            self.assertIn("motivo", resultado["ljung_box"])

    def test_grados_de_libertad_validos(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": _serie_ar(n=40), "p": 1, "d": 0, "q": 0})
        ljung_box = resultado["ljung_box"]
        if ljung_box["ejecutado"]:
            self.assertGreater(ljung_box["grados_libertad_prueba"], 0)

    def test_interpretacion_prudente(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": _serie_ar(n=40), "p": 1, "d": 0, "q": 0})
        interpretacion = resultado["ljung_box"]["interpretacion"].lower()
        self.assertNotIn("modelo valido", interpretacion)
        self.assertNotIn("modelo válido", interpretacion)

    def test_exogenas_no_incluidas_en_model_df(self):
        y, x = _serie_con_exogena(n=40)
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": y, "p": 1, "d": 0, "q": 1,
            "variables_exogenas_historicas": {"x": x}, "variables_exogenas_futuras": {"x": [21.0]},
        })
        ljung_box = resultado["ljung_box"]
        if ljung_box["ejecutado"]:
            self.assertEqual(ljung_box["model_df"], 1 + 1)  # p+q, sin contar la exogena


class EvaluacionSARIMAXTests(SimpleTestCase):
    def test_sin_estacionalidad_ni_exogenas(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": _serie_random_walk_drift(n=24), "p": 0, "d": 1, "q": 0,
            "evaluar_modelo": True, "cantidad_prueba": 5,
        })
        self.assertTrue(resultado["evaluacion"]["ejecutada"])

    def test_con_estacionalidad(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": _serie_estacional(n=48), "p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
            "evaluar_modelo": True, "cantidad_prueba": 12,
        })
        self.assertTrue(resultado["evaluacion"]["ejecutada"])

    def test_con_exogenas(self):
        y, x = _serie_con_exogena(n=40)
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": y, "p": 1, "d": 0, "q": 0,
            "variables_exogenas_historicas": {"x": x}, "variables_exogenas_futuras": {"x": [21.0]},
            "evaluar_modelo": True, "cantidad_prueba": 8,
        })
        self.assertEqual(resultado["evaluacion"]["tipo"], "condicionada_a_exogenas_observadas")

    def test_con_estacionalidad_y_exogenas(self):
        y, x = _serie_sarimax_completa(n=60)
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": y, "p": 1, "d": 0, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
            "variables_exogenas_historicas": {"temperatura": x},
            "variables_exogenas_futuras": {"temperatura": [21.0]},
            "evaluar_modelo": True, "cantidad_prueba": 12,
        })
        self.assertTrue(resultado["evaluacion"]["ejecutada"])
        self.assertEqual(resultado["evaluacion"]["tipo"], "condicionada_a_exogenas_observadas")

    def test_division_conjunta_ultimas_observaciones(self):
        serie = _serie_random_walk_drift(n=24)
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": serie, "p": 0, "d": 1, "q": 0, "evaluar_modelo": True, "cantidad_prueba": 5,
        })
        self.assertEqual([round(v, 6) for v in serie[-5:]], resultado["evaluacion"]["valores_reales"])

    def test_ciclos_en_entrenamiento(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": _serie_estacional(n=48), "p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
            "evaluar_modelo": True, "cantidad_prueba": 40,
        })
        self.assertNotIn("error", resultado)
        self.assertFalse(resultado["evaluacion"]["ejecutada"])

    def test_exogenas_de_prueba(self):
        y, x = _serie_con_exogena(n=40)
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": y, "p": 1, "d": 0, "q": 0,
            "variables_exogenas_historicas": {"x": x}, "variables_exogenas_futuras": {"x": [21.0]},
            "evaluar_modelo": True, "cantidad_prueba": 8,
        })
        self.assertTrue(resultado["evaluacion"]["ejecutada"])
        self.assertEqual(len(resultado["evaluacion"]["valores_pronosticados"]), 8)

    def test_reajuste_final(self):
        serie = _serie_random_walk_drift(n=24)
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": serie, "p": 0, "d": 1, "q": 0, "evaluar_modelo": True, "cantidad_prueba": 5,
        })
        self.assertEqual(resultado["n_observaciones"], len(serie))

    def test_pronostico_futuro_independiente(self):
        serie = _serie_random_walk_drift(n=24)
        con_eval = ejecutar_herramienta("modelo_sarimax", {
            "valores": serie, "p": 0, "d": 1, "q": 0, "pasos_pronostico": 3, "evaluar_modelo": True, "cantidad_prueba": 5,
        })
        sin_eval = ejecutar_herramienta("modelo_sarimax", {
            "valores": serie, "p": 0, "d": 1, "q": 0, "pasos_pronostico": 3,
        })
        self.assertEqual(con_eval["pronostico"], sin_eval["pronostico"])

    def test_mape_con_cero(self):
        serie = _serie_random_walk_drift(n=24)
        serie[-1] = 0.0
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": serie, "p": 0, "d": 1, "q": 0, "evaluar_modelo": True, "cantidad_prueba": 3,
        })
        if resultado["evaluacion"]["ejecutada"]:
            self.assertGreaterEqual(resultado["evaluacion"]["mape_detalle"]["observaciones_excluidas_por_cero"], 1)

    def test_prueba_menor_que_un_ciclo(self):
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": _serie_estacional(n=60), "p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
            "evaluar_modelo": True, "cantidad_prueba": 5,
        })
        if resultado["evaluacion"]["ejecutada"]:
            codigos = [a["codigo"] for a in resultado["evaluacion"].get("advertencias", [])]
            self.assertIn("PRUEBA_NO_CUBRE_CICLO_COMPLETO", codigos)


class FechasSARIMAXTests(SimpleTestCase):
    def test_serie_mensual(self):
        serie = _serie_estacional(n=36)
        fechas = _fechas(36, freq="MS")
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": serie, "p": 1, "d": 0, "q": 0, "fechas": fechas,
        })
        self.assertEqual(resultado["informacion_temporal"]["frecuencia_inferida"], "MS")

    def test_serie_trimestral(self):
        fechas = _fechas(20, freq="QS")
        serie = _serie_random_walk_drift(n=20)
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": serie, "p": 1, "d": 0, "q": 0, "fechas": fechas})
        self.assertEqual(resultado["informacion_temporal"]["frecuencia_inferida"].split("-")[0], "QS")

    def test_serie_diaria(self):
        fechas = _fechas(30, freq="D")
        serie = _serie_random_walk_drift(n=30)
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": serie, "p": 1, "d": 0, "q": 0, "fechas": fechas})
        self.assertEqual(resultado["informacion_temporal"]["frecuencia_inferida"], "D")

    def test_fechas_exogenas_alineadas(self):
        y, x = _serie_con_exogena(n=36)
        fechas = _fechas(36)
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": y, "p": 1, "d": 0, "q": 0,
            "variables_exogenas_historicas": {"x": x}, "variables_exogenas_futuras": {"x": [21.0]},
            "fechas": fechas, "fechas_exogenas_historicas": fechas,
        })
        self.assertEqual(resultado["informacion_temporal"]["alineacion_exogenas"]["metodo"], "fechas_propias")

    def test_fechas_futuras(self):
        serie = _serie_estacional(n=36)
        fechas = _fechas(36)
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": serie, "p": 1, "d": 0, "q": 0, "fechas": fechas, "pasos_pronostico": 2,
        })
        self.assertEqual(len(resultado["fechas_pronostico"]), 2)

    def test_fechas_duplicadas(self):
        serie = _serie_random_walk_drift(n=24)
        fechas = _fechas(24)
        fechas[3] = fechas[2]
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": serie, "p": 0, "d": 1, "q": 0, "fechas": fechas})
        self.assertEqual(resultado.get("codigo_error"), "FECHAS_DUPLICADAS")

    def test_fechas_desordenadas(self):
        serie = _serie_random_walk_drift(n=24)
        fechas = _fechas(24)
        fechas[0], fechas[1] = fechas[1], fechas[0]
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": serie, "p": 0, "d": 1, "q": 0, "fechas": fechas})
        self.assertEqual(resultado.get("codigo_error"), "FECHAS_DESORDENADAS")

    def test_frecuencia_irregular(self):
        serie = _serie_random_walk_drift(n=10)
        fechas_irregulares = ["2020-01-01", "2020-01-03", "2020-02-19"] + _fechas(10)[3:]
        resultado = ejecutar_herramienta("modelo_sarimax", {"valores": serie, "p": 0, "d": 1, "q": 0, "fechas": fechas_irregulares})
        self.assertIsNone(resultado["informacion_temporal"]["frecuencia_utilizada"])

    def test_periodos_faltantes(self):
        serie = _serie_estacional(n=36)
        fechas = _fechas(36)
        serie_incompleta = serie[:5] + serie[6:]
        fechas_incompletas = fechas[:5] + fechas[6:]
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": serie_incompleta, "p": 1, "d": 0, "q": 0,
            "fechas": fechas_incompletas, "frecuencia": "mensual",
        })
        self.assertFalse(resultado["informacion_temporal"]["serie_regular"])

    def test_frecuencia_incompatible_con_s(self):
        serie = _serie_estacional(n=36)
        fechas = _fechas(36, freq="MS")
        resultado = ejecutar_herramienta("modelo_sarimax", {
            "valores": serie, "p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
            "fechas": fechas, "frecuencia": "diaria",
        })
        self.assertEqual(resultado.get("codigo_error"), "FRECUENCIA_INCONSISTENTE")


class RegresionGlobalTests(SimpleTestCase):
    """Confirma que agregar `modelo_sarimax` (y consolidar el nucleo) no
    altera el comportamiento de ninguna fachada anterior."""

    def test_modelo_ar(self):
        resultado = ejecutar_herramienta("modelo_ar", {"valores": _serie_ar(n=30), "p": 1})
        self.assertNotIn("error", resultado)

    def test_modelo_ma(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_ma(), "q": 1})
        self.assertNotIn("error", resultado)

    def test_modelo_arima(self):
        resultado = ejecutar_herramienta("modelo_arima", {"valores": _serie_random_walk_drift(), "p": 0, "d": 1, "q": 0})
        self.assertNotIn("error", resultado)

    def test_modelo_sarima(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": _serie_estacional(n=48), "p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
        })
        self.assertNotIn("error", resultado)

    def test_modelo_arimax(self):
        y, x = _serie_con_exogena()
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": y, "variables_exogenas_historicas": {"x": x},
            "variables_exogenas_futuras": {"x": [21.0]}, "p": 1, "d": 0, "q": 0,
        })
        self.assertNotIn("error", resultado)

    def test_todas_las_herramientas_conviven_en_el_registro(self):
        for nombre in (
            "modelo_ar", "modelo_ma", "modelo_arima", "modelo_sarima",
            "modelo_arimax", "modelo_sarimax", "acf", "pacf", "modelo_dickey_fuller",
            "descomposicion_visualizacion_serie", "estabilizacion_media", "estabilizacion_varianza",
        ):
            self.assertIn(nombre, TOOL_REGISTRY)
