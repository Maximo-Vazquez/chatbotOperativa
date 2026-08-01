"""Pruebas de extremo a extremo de la fase 3 de `modelo_arima`: evaluacion
temporal fuera de muestra, fechas/frecuencia opcionales, fechas futuras del
pronostico y compatibilidad con las llamadas de las fases anteriores.

Complementa (no reemplaza) `test_modelo_arima.py` de la fase 2, que sigue
cubriendo el contrato base de ajuste/diagnostico/intervalos.
"""

import json

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from apps.herramientas.tools import ejecutar_herramienta


def _serie_random_walk_con_drift(n=24, drift=1.5, escala_ruido=4.0, seed=4):
    rng = np.random.default_rng(seed)
    ruido = rng.normal(scale=escala_ruido, size=n)
    valores = [100.0]
    for t in range(1, n):
        valores.append(valores[-1] + drift + ruido[t])
    return valores


def _fechas_mensuales(n, inicio="2024-01-01"):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(inicio, periods=n, freq="MS")]


class ArimaEvaluacionTemporalTests(SimpleTestCase):
    def setUp(self):
        self.serie = _serie_random_walk_con_drift(n=24)
        self.args_base = {
            "valores": self.serie, "p": 0, "d": 1, "q": 0,
            "pasos_pronostico": 3, "con_constante": True,
        }

    def test_pronostico_de_evaluacion_tiene_longitud_igual_a_prueba(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            **self.args_base, "evaluar_modelo": True, "cantidad_prueba": 4,
        })
        evaluacion = resultado["evaluacion"]
        self.assertTrue(evaluacion["ejecutada"])
        self.assertEqual(len(evaluacion["valores_pronosticados"]), 4)
        self.assertEqual(evaluacion["n_prueba"], 4)

    def test_valores_reales_de_prueba_alineados_con_el_final_de_la_serie(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            **self.args_base, "evaluar_modelo": True, "cantidad_prueba": 4,
        })
        evaluacion = resultado["evaluacion"]
        self.assertEqual(
            [round(v, 6) for v in self.serie[-4:]],
            evaluacion["valores_reales"],
        )

    def test_metricas_prueba_mae_rmse_mape_presentes(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            **self.args_base, "evaluar_modelo": True, "cantidad_prueba": 4,
        })
        metricas = resultado["evaluacion"]["metricas_prueba"]
        for clave in ("mae", "rmse", "mape"):
            self.assertIn(clave, metricas)
            if metricas[clave] is not None:
                self.assertTrue(np.isfinite(metricas[clave]))

    def test_reajuste_final_usa_todas_las_observaciones(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            **self.args_base, "evaluar_modelo": True, "cantidad_prueba": 4,
        })
        self.assertEqual(resultado["n_observaciones"], len(self.serie))

    def test_pronostico_futuro_es_independiente_de_la_evaluacion(self):
        con_evaluacion = ejecutar_herramienta("modelo_arima", {
            **self.args_base, "evaluar_modelo": True, "cantidad_prueba": 4,
        })
        sin_evaluacion = ejecutar_herramienta("modelo_arima", {
            **self.args_base, "evaluar_modelo": False,
        })
        # El pronostico final se genera reajustando con TODA la serie en
        # ambos casos: activar la evaluacion no debe alterarlo.
        self.assertEqual(con_evaluacion["pronostico"], sin_evaluacion["pronostico"])

    def test_error_residual_separado_de_metricas_de_prueba(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            **self.args_base, "evaluar_modelo": True, "cantidad_prueba": 4,
        })
        self.assertIn("mse_residuos_entrenamiento", resultado)
        self.assertIn("metricas_prueba", resultado["evaluacion"])
        # Son magnitudes de naturaleza distinta (MSE vs MAE/RMSE/MAPE) sobre
        # datos distintos (entrenamiento completo vs holdout): no deben
        # mezclarse en la misma clave.
        self.assertNotIn("mae", resultado)
        self.assertNotIn("rmse", resultado)

    def test_evaluacion_imposible_permite_ajuste_final(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            **self.args_base, "evaluar_modelo": True, "cantidad_prueba": 22,
        })
        self.assertNotIn("error", resultado)
        self.assertFalse(resultado["evaluacion"]["ejecutada"])
        self.assertIn("motivo", resultado["evaluacion"])
        self.assertEqual(len(resultado["pronostico"]), 3)

    def test_evaluacion_desactivada_por_defecto(self):
        resultado = ejecutar_herramienta("modelo_arima", self.args_base)
        self.assertEqual(resultado["evaluacion"], {"ejecutada": False})

    def test_porcentaje_prueba_valido(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            **self.args_base, "evaluar_modelo": True, "porcentaje_prueba": 0.25,
        })
        self.assertTrue(resultado["evaluacion"]["ejecutada"])
        self.assertEqual(resultado["evaluacion"]["n_prueba"], round(24 * 0.25))

    def test_resultado_con_evaluacion_es_json_serializable(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            **self.args_base, "evaluar_modelo": True, "cantidad_prueba": 4,
        })
        json.dumps(resultado, ensure_ascii=False)

    def test_configuracion_de_prueba_invalida_devuelve_error_controlado(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            **self.args_base, "evaluar_modelo": True, "porcentaje_prueba": 0.9,
        })
        self.assertEqual(resultado.get("codigo_error"), "CONFIGURACION_PRUEBA_INVALIDA")


class ArimaFechasYFrecuenciaTests(SimpleTestCase):
    def setUp(self):
        self.serie = _serie_random_walk_con_drift(n=24)

    def test_fechas_mensuales_con_frecuencia_inferida(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": self.serie, "p": 0, "d": 1, "q": 0,
            "pasos_pronostico": 2, "fechas": _fechas_mensuales(24),
        })
        info = resultado["informacion_temporal"]
        self.assertTrue(info["fechas_proporcionadas"])
        self.assertEqual(info["frecuencia_inferida"], "MS")
        self.assertEqual(info["frecuencia_utilizada"], "MS")
        self.assertTrue(info["serie_regular"])

    def test_alias_pedagogico_de_frecuencia_explicita(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": self.serie, "p": 0, "d": 1, "q": 0,
            "pasos_pronostico": 2, "fechas": _fechas_mensuales(24),
            "frecuencia": "mensual",
        })
        self.assertEqual(resultado["informacion_temporal"]["frecuencia_solicitada"], "MS")

    def test_frecuencia_explicita_incompatible_es_error_controlado(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": self.serie, "p": 0, "d": 1, "q": 0,
            "fechas": _fechas_mensuales(24), "frecuencia": "diaria",
        })
        self.assertEqual(resultado.get("codigo_error"), "FRECUENCIA_INCONSISTENTE")

    def test_fechas_duplicadas_es_error_controlado(self):
        fechas = _fechas_mensuales(24)
        fechas[5] = fechas[4]
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": self.serie, "p": 0, "d": 1, "q": 0, "fechas": fechas,
        })
        self.assertEqual(resultado.get("codigo_error"), "FECHAS_DUPLICADAS")

    def test_fechas_desordenadas_es_error_controlado(self):
        fechas = _fechas_mensuales(24)
        fechas[0], fechas[1] = fechas[1], fechas[0]
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": self.serie, "p": 0, "d": 1, "q": 0, "fechas": fechas,
        })
        self.assertEqual(resultado.get("codigo_error"), "FECHAS_DESORDENADAS")

    def test_periodo_faltante_genera_advertencia_sin_completar_datos(self):
        fechas = _fechas_mensuales(24)
        serie_incompleta = self.serie[:5] + self.serie[6:]
        fechas_incompletas = fechas[:5] + fechas[6:]
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie_incompleta, "p": 0, "d": 1, "q": 0,
            "fechas": fechas_incompletas, "frecuencia": "mensual",
        })
        self.assertFalse(resultado["informacion_temporal"]["serie_regular"])
        self.assertEqual(resultado["n_observaciones"], len(serie_incompleta))
        codigos = [a["codigo"] for a in resultado["advertencias"]]
        self.assertIn("PERIODOS_FALTANTES", codigos)

    def test_serie_sin_fechas_no_genera_ruido_de_advertencias(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": self.serie, "p": 0, "d": 1, "q": 0,
        })
        self.assertFalse(resultado["informacion_temporal"]["fechas_proporcionadas"])
        codigos = [a["codigo"] for a in resultado["advertencias"]]
        self.assertNotIn("SERIE_SIN_FECHAS", codigos)


class ArimaFechasFuturasTests(SimpleTestCase):
    def test_fechas_pronostico_mensuales(self):
        serie = _serie_random_walk_con_drift(n=24)
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie, "p": 0, "d": 1, "q": 0,
            "pasos_pronostico": 3, "fechas": _fechas_mensuales(24),
        })
        self.assertEqual(resultado["fechas_pronostico"], ["2026-01-01", "2026-02-01", "2026-03-01"])

    def test_primera_fecha_pronosticada_es_posterior_a_la_ultima_observada(self):
        serie = _serie_random_walk_con_drift(n=24)
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie, "p": 0, "d": 1, "q": 0,
            "pasos_pronostico": 1, "fechas": _fechas_mensuales(24),
        })
        self.assertGreater(pd.Timestamp(resultado["fechas_pronostico"][0]), pd.Timestamp(_fechas_mensuales(24)[-1]))

    def test_intervalos_pronostico_incluyen_fecha_correspondiente(self):
        serie = _serie_random_walk_con_drift(n=24)
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie, "p": 0, "d": 1, "q": 0,
            "pasos_pronostico": 2, "fechas": _fechas_mensuales(24),
        })
        for intervalo, fecha in zip(resultado["intervalos_pronostico"], resultado["fechas_pronostico"]):
            self.assertEqual(intervalo["fecha"], fecha)

    def test_sin_frecuencia_conocida_fechas_pronostico_es_none(self):
        serie = _serie_random_walk_con_drift(n=24)
        fechas_irregulares = ["2024-01-01", "2024-01-03", "2024-02-19"] + _fechas_mensuales(24)[3:]
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie, "p": 0, "d": 1, "q": 0,
            "pasos_pronostico": 2, "fechas": fechas_irregulares,
        })
        self.assertIsNone(resultado["fechas_pronostico"])
        self.assertIsNone(resultado["intervalos_pronostico"][0]["fecha"])

    def test_sin_fechas_fechas_pronostico_es_none(self):
        serie = _serie_random_walk_con_drift(n=24)
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie, "p": 0, "d": 1, "q": 0, "pasos_pronostico": 2,
        })
        self.assertIsNone(resultado["fechas_pronostico"])


class ArimaCompatibilidadFase3Tests(SimpleTestCase):
    def test_llamada_de_fase_2_sigue_funcionando_sin_argumentos_nuevos(self):
        serie = _serie_random_walk_con_drift(n=24)
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie, "p": 0, "d": 1, "q": 0,
            "pasos_pronostico": 3, "con_constante": True,
        })
        self.assertNotIn("error", resultado)
        claves_anteriores = (
            "modelo", "orden", "n_observaciones", "coeficientes", "aic", "bic",
            "mse_residuos", "mse_residuos_entrenamiento", "media_residuos",
            "varianza_residuos", "ljung_box", "pasos_pronostico", "pronostico",
            "detalle_coeficientes", "diagnostico_residuos", "intervalos_pronostico",
            "nivel_confianza", "tendencia_statsmodels", "descripcion_tendencia",
            "informacion_ajuste", "advertencias",
        )
        for clave in claves_anteriores:
            self.assertIn(clave, resultado)

    def test_claves_nuevas_presentes_con_valores_por_defecto(self):
        serie = _serie_random_walk_con_drift(n=24)
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie, "p": 0, "d": 1, "q": 0, "pasos_pronostico": 3,
        })
        self.assertIn("evaluacion", resultado)
        self.assertIn("informacion_temporal", resultado)
        self.assertIn("fechas_pronostico", resultado)
        self.assertEqual(resultado["evaluacion"], {"ejecutada": False})
        self.assertIsNone(resultado["fechas_pronostico"])

    def test_argumentos_nuevos_son_opcionales_en_tool_definition(self):
        from apps.herramientas.tools import TOOL_DEFINITIONS

        definicion = next(d for d in TOOL_DEFINITIONS if d["function"]["name"] == "modelo_arima")
        requeridos = definicion["function"]["parameters"]["required"]
        propiedades = definicion["function"]["parameters"]["properties"]
        for nuevo in ("fechas", "frecuencia", "evaluar_modelo", "cantidad_prueba", "porcentaje_prueba"):
            self.assertIn(nuevo, propiedades)
            self.assertNotIn(nuevo, requeridos)
        self.assertEqual(set(requeridos), {"valores", "p", "d", "q"})


class ArimaCasoAcademicoFase3Tests(SimpleTestCase):
    """Igual que en la fase 2: no existe en el repositorio una serie fija del
    caso academico de 24 meses, asi que se usa una serie sintetica
    documentada como tal, cubriendo los mismos chequeos que pediria esa
    prueba: ajuste con drift, evaluacion temporal pequena, MAE/RMSE/MAPE,
    tres pronosticos futuros finitos, intervalos finitos y serializacion.
    """

    def test_caso_academico_sintetico_con_evaluacion(self):
        serie = _serie_random_walk_con_drift(n=24)
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie, "p": 0, "d": 1, "q": 0,
            "pasos_pronostico": 3, "con_constante": True,
            "evaluar_modelo": True, "cantidad_prueba": 3,
        })

        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["tendencia_statsmodels"], "t")

        evaluacion = resultado["evaluacion"]
        self.assertTrue(evaluacion["ejecutada"])
        for clave in ("mae", "rmse", "mape"):
            self.assertIn(clave, evaluacion["metricas_prueba"])

        self.assertEqual(len(resultado["pronostico"]), 3)
        for valor in resultado["pronostico"]:
            self.assertTrue(np.isfinite(valor))
        for intervalo in resultado["intervalos_pronostico"]:
            self.assertTrue(np.isfinite(intervalo["limite_inferior"]))
            self.assertTrue(np.isfinite(intervalo["limite_superior"]))

        json.dumps(resultado, ensure_ascii=False)
