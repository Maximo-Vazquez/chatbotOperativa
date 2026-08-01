"""Pruebas de extremo a extremo de la herramienta publica `modelo_ma`.

Ejecutan `ejecutar_herramienta("modelo_ma", ...)` tal como lo haria el
chatbot real (mismo loader dinamico de `apps/herramientas/tools.py`), para
cubrir: registro dinamico, validacion propia de MA, reutilizacion del nucleo
compartido (motor, metricas, evaluacion, fechas, diagnostico), estacionariedad
(via `modelo_dickey_fuller`), invertibilidad e identificacion (via `acf`), y
contenido pedagogico. Las series sinteticas usan semillas fijas.
"""

import json

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from apps.herramientas.tools import TOOL_DEFINITIONS, TOOL_META, TOOL_REGISTRY, ejecutar_herramienta


def _serie_ma1(n=60, theta=0.6, mu=20.0, escala=2.0, seed=7):
    rng = np.random.default_rng(seed)
    ruido = rng.normal(scale=escala, size=n + 1)
    return [mu + ruido[t] + theta * ruido[t - 1] for t in range(1, n + 1)]


def _serie_ma2(n=60, theta1=0.5, theta2=-0.2, mu=15.0, escala=2.0, seed=8):
    rng = np.random.default_rng(seed)
    ruido = rng.normal(scale=escala, size=n + 2)
    return [
        mu + ruido[t] + theta1 * ruido[t - 1] + theta2 * ruido[t - 2]
        for t in range(2, n + 2)
    ]


def _serie_con_tendencia(n=40, pendiente=2.0, seed=10):
    rng = np.random.default_rng(seed)
    ruido = rng.normal(scale=1.5, size=n)
    return [100.0 + pendiente * t + ruido[t] for t in range(n)]


def _fechas_mensuales(n, inicio="2022-01-01"):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(inicio, periods=n, freq="MS")]


class ContratoMATests(SimpleTestCase):
    def test_modelo_ma_se_registra_dinamicamente(self):
        self.assertIn("modelo_ma", TOOL_REGISTRY)

    def test_tool_definition_tiene_nombre_correcto(self):
        nombres = [d["function"]["name"] for d in TOOL_DEFINITIONS]
        self.assertIn("modelo_ma", nombres)

    def test_tool_meta_existe(self):
        self.assertIn("modelo_ma", TOOL_META)
        self.assertIn("label", TOOL_META["modelo_ma"])

    def test_tool_function_es_ejecutable(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_ma1(), "q": 1})
        self.assertNotIn("error", resultado)

    def test_valores_y_q_son_obligatorios(self):
        definicion = next(d for d in TOOL_DEFINITIONS if d["function"]["name"] == "modelo_ma")
        requeridos = definicion["function"]["parameters"]["required"]
        self.assertEqual(set(requeridos), {"valores", "q"})

    def test_campos_opcionales_definidos(self):
        definicion = next(d for d in TOOL_DEFINITIONS if d["function"]["name"] == "modelo_ma")
        propiedades = definicion["function"]["parameters"]["properties"]
        for campo in (
            "pasos_pronostico", "con_constante", "nivel_confianza", "fechas",
            "frecuencia", "evaluar_modelo", "cantidad_prueba", "porcentaje_prueba",
        ):
            self.assertIn(campo, propiedades)

    def test_respuesta_es_json_serializable(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_ma1(), "q": 1})
        json.dumps(resultado, ensure_ascii=False)


class ValidacionMATests(SimpleTestCase):
    def setUp(self):
        self.serie = _serie_ma1()

    def test_q_uno_es_valido(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": self.serie, "q": 1})
        self.assertNotIn("error", resultado)

    def test_q_dos_es_valido(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_ma2(), "q": 2})
        self.assertNotIn("error", resultado)

    def test_q_cero_invalido(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": self.serie, "q": 0})
        self.assertEqual(resultado.get("codigo_error"), "ORDEN_MA_INVALIDO")

    def test_q_negativo_invalido(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": self.serie, "q": -1})
        self.assertIn("error", resultado)

    def test_q_decimal_invalido(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": self.serie, "q": 1.5})
        self.assertIn("error", resultado)

    def test_q_booleano_invalido(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": self.serie, "q": True})
        self.assertIn("error", resultado)

    def test_q_excesivo_invalido(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": self.serie, "q": 25})
        self.assertEqual(resultado.get("codigo_error"), "ORDEN_MA_INVALIDO")

    def test_serie_vacia(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": [], "q": 1})
        self.assertIn("error", resultado)

    def test_serie_corta(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": self.serie[:3], "q": 2})
        self.assertEqual(resultado.get("codigo_error"), "MUESTRA_INSUFICIENTE")

    def test_serie_constante(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": [5.0] * 20, "q": 1})
        self.assertEqual(resultado.get("codigo_error"), "SERIE_INVALIDA")

    def test_serie_con_texto(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": self.serie[:-1] + ["x"], "q": 1})
        self.assertEqual(resultado.get("codigo_error"), "SERIE_INVALIDA")

    def test_serie_con_booleanos(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": self.serie[:-1] + [True], "q": 1})
        self.assertEqual(resultado.get("codigo_error"), "SERIE_INVALIDA")

    def test_serie_con_nan(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": self.serie[:-1] + [float("nan")], "q": 1})
        self.assertEqual(resultado.get("codigo_error"), "SERIE_INVALIDA")

    def test_serie_con_infinito(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": self.serie[:-1] + [float("inf")], "q": 1})
        self.assertEqual(resultado.get("codigo_error"), "SERIE_INVALIDA")

    def test_horizonte_invalido(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": self.serie, "q": 1, "pasos_pronostico": 0})
        self.assertEqual(resultado.get("codigo_error"), "HORIZONTE_INVALIDO")

    def test_nivel_confianza_invalido(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": self.serie, "q": 1, "nivel_confianza": 1.5})
        self.assertEqual(resultado.get("codigo_error"), "NIVEL_CONFIANZA_INVALIDO")


class MA1Tests(SimpleTestCase):
    def setUp(self):
        self.resultado = ejecutar_herramienta("modelo_ma", {
            "valores": _serie_ma1(), "q": 1, "pasos_pronostico": 3,
        })

    def test_ajuste_sin_error(self):
        self.assertNotIn("error", self.resultado)

    def test_nombre_ma1(self):
        self.assertEqual(self.resultado["modelo"], "MA(1)")

    def test_representacion_interna_arima(self):
        self.assertEqual(self.resultado["representacion_interna"], "ARIMA(0,0,1)")

    def test_presencia_de_ma_l1(self):
        self.assertIn("ma.L1", self.resultado["coeficientes_ma"])
        self.assertIn("ma.L1", self.resultado["coeficientes"])

    def test_pronostico_finito(self):
        for valor in self.resultado["pronostico"]:
            self.assertTrue(np.isfinite(valor))

    def test_intervalos_finitos(self):
        for intervalo in self.resultado["intervalos_pronostico"]:
            self.assertTrue(np.isfinite(intervalo["limite_inferior"]))
            self.assertTrue(np.isfinite(intervalo["limite_superior"]))

    def test_aic_y_bic_presentes(self):
        self.assertTrue(np.isfinite(self.resultado["aic"]))
        self.assertTrue(np.isfinite(self.resultado["bic"]))

    def test_diagnostico_residual_presente(self):
        self.assertIn("diagnostico_residuos", self.resultado)
        self.assertGreater(self.resultado["diagnostico_residuos"]["cantidad_residuos"], 0)

    def test_serializable(self):
        json.dumps(self.resultado, ensure_ascii=False)


class MA2Tests(SimpleTestCase):
    def setUp(self):
        self.resultado = ejecutar_herramienta("modelo_ma", {
            "valores": _serie_ma2(), "q": 2, "pasos_pronostico": 2,
        })

    def test_presencia_ma_l1_y_ma_l2(self):
        self.assertIn("ma.L1", self.resultado["coeficientes_ma"])
        self.assertIn("ma.L2", self.resultado["coeficientes_ma"])

    def test_orden_q_es_2(self):
        self.assertEqual(self.resultado["orden_q"], 2)
        self.assertEqual(self.resultado["orden"], {"p": 0, "d": 0, "q": 2})

    def test_detalle_coeficientes_incluye_ambos_ma(self):
        nombres = {d["nombre"] for d in self.resultado["detalle_coeficientes"]}
        self.assertIn("ma.L1", nombres)
        self.assertIn("ma.L2", nombres)

    def test_p_valores_presentes(self):
        for detalle in self.resultado["detalle_coeficientes"]:
            if detalle["nombre"] in ("ma.L1", "ma.L2"):
                self.assertIn("p_value", detalle)

    def test_intervalos_de_confianza_presentes(self):
        for detalle in self.resultado["detalle_coeficientes"]:
            if detalle["nombre"] in ("ma.L1", "ma.L2") and detalle["intervalo_confianza"] is not None:
                ic = detalle["intervalo_confianza"]
                self.assertLessEqual(ic["inferior"], ic["superior"])

    def test_coeficientes_ma_clasificados_como_media_movil(self):
        for detalle in self.resultado["detalle_coeficientes"]:
            if detalle["nombre"] in ("ma.L1", "ma.L2"):
                self.assertEqual(detalle["tipo"], "media_movil")

    def test_diagnostico_presente(self):
        self.assertIn("ljung_box", self.resultado)

    def test_pronostico_presente(self):
        self.assertEqual(len(self.resultado["pronostico"]), 2)


class ConstanteMATests(SimpleTestCase):
    def test_con_constante_true(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_ma1(), "q": 1, "con_constante": True})
        self.assertEqual(resultado["tendencia_statsmodels"], "c")
        self.assertIn("const", resultado["coeficientes"])

    def test_con_constante_false(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_ma1(), "q": 1, "con_constante": False})
        self.assertEqual(resultado["tendencia_statsmodels"], "n")
        self.assertNotIn("const", resultado["coeficientes"])

    def test_ausencia_de_drift_en_descripcion(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_ma1(), "q": 1, "con_constante": True})
        self.assertNotIn("drift", resultado["descripcion_tendencia"].lower())
        self.assertNotIn("drift", resultado["modelo"].lower())

    def test_ambos_ajustes_son_compatibles_sin_error(self):
        for con_constante in (True, False):
            resultado = ejecutar_herramienta("modelo_ma", {
                "valores": _serie_ma1(), "q": 1, "con_constante": con_constante,
            })
            self.assertNotIn("error", resultado)


class EstacionariedadMATests(SimpleTestCase):
    def test_serie_estacionaria_ejecuta_adf(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_ma1(n=80), "q": 1})
        self.assertEqual(resultado["estacionariedad"]["prueba"], "ADF")
        self.assertTrue(resultado["estacionariedad"]["ejecutada"])

    def test_serie_con_tendencia_genera_advertencia(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_con_tendencia(), "q": 1})
        codigos = [a["codigo"] for a in resultado["advertencias"]]
        self.assertIn("SERIE_NO_ESTACIONARIA_PARA_MA", codigos)

    def test_recomienda_arima_con_diferenciacion(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_con_tendencia(), "q": 1})
        advertencia = next(a for a in resultado["advertencias"] if a["codigo"] == "SERIE_NO_ESTACIONARIA_PARA_MA")
        self.assertIn("ARIMA", advertencia["mensaje"])
        self.assertEqual(advertencia["severidad"], "advertencia_alta")

    def test_no_diferencia_automaticamente(self):
        # El ajuste sigue siendo MA(q) puro (d=0), nunca se convierte en ARIMA.
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_con_tendencia(), "q": 1})
        self.assertEqual(resultado["orden"]["d"], 0)
        self.assertNotIn("error", resultado)

    def test_test_no_ejecutable_por_muestra_pequena_no_bloquea_el_ajuste(self):
        # Serie muy corta: el ADF interno puede recurrir a su diagnostico
        # operativo, pero el ajuste MA en si debe completarse igual.
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_ma1(n=6, seed=11), "q": 1})
        self.assertNotIn("error", resultado)
        self.assertIn("estacionariedad", resultado)


class InvertibilidadMATests(SimpleTestCase):
    def test_configuracion_informada(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_ma1(), "q": 1})
        invertibilidad = resultado["invertibilidad"]
        self.assertTrue(invertibilidad["forzada_por_statsmodels"])
        self.assertFalse(invertibilidad["verificacion_manual"])

    def test_advertencias_se_capturan_con_parametros_iniciales_problematicos(self):
        # Orden alto sobre ruido corto: statsmodels suele emitir advertencias
        # de parametros iniciales no invertibles/convergencia.
        rng = np.random.default_rng(3)
        serie_ruidosa = list(rng.normal(size=20))
        resultado = ejecutar_herramienta("modelo_ma", {"valores": serie_ruidosa, "q": 5})
        self.assertNotIn("error", resultado)
        self.assertIsInstance(resultado["advertencias"], list)

    def test_resultado_serializable(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_ma1(), "q": 1})
        json.dumps(resultado["invertibilidad"], ensure_ascii=False)


class LjungBoxMATests(SimpleTestCase):
    def test_model_df_igual_a_q(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_ma2(), "q": 2})
        ljung_box = resultado["ljung_box"]
        if ljung_box["ejecutado"]:
            self.assertEqual(ljung_box["model_df"], 2)

    def test_lag_mayor_que_q(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_ma1(), "q": 1})
        ljung_box = resultado["ljung_box"]
        if ljung_box["ejecutado"]:
            self.assertGreater(ljung_box["lags"], 1)

    def test_lag_valido_da_estructura_completa(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_ma1(n=80), "q": 1})
        ljung_box = resultado["ljung_box"]
        self.assertTrue(ljung_box["ejecutado"])
        for campo in ("estadistico", "p_value", "autocorrelacion_significativa"):
            self.assertIn(campo, ljung_box)

    def test_muestra_pequena_sin_lag_valido_no_produce_nan(self):
        rng = np.random.default_rng(12)
        serie_corta = list(20 + rng.normal(size=10))
        resultado = ejecutar_herramienta("modelo_ma", {"valores": serie_corta, "q": 4})
        if "error" not in resultado:
            ljung_box = resultado["ljung_box"]
            if not ljung_box["ejecutado"]:
                self.assertIn("motivo", ljung_box)
                self.assertIsNone(ljung_box["estadistico"])

    def test_interpretacion_prudente_sin_frase_modelo_valido(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_ma1(n=80), "q": 1})
        interpretacion = resultado["ljung_box"]["interpretacion"].lower()
        self.assertNotIn("modelo valido", interpretacion)
        self.assertNotIn("modelo válido", interpretacion)


class EvaluacionTemporalMATests(SimpleTestCase):
    def setUp(self):
        self.serie = _serie_ma1(n=48, seed=13)

    def test_evaluacion_desactivada_por_defecto(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": self.serie, "q": 1})
        self.assertEqual(resultado["evaluacion"], {"ejecutada": False})

    def test_evaluacion_activada(self):
        resultado = ejecutar_herramienta("modelo_ma", {
            "valores": self.serie, "q": 1, "evaluar_modelo": True, "cantidad_prueba": 6,
        })
        self.assertTrue(resultado["evaluacion"]["ejecutada"])

    def test_ultimas_observaciones_como_prueba(self):
        resultado = ejecutar_herramienta("modelo_ma", {
            "valores": self.serie, "q": 1, "evaluar_modelo": True, "cantidad_prueba": 6,
        })
        self.assertEqual(
            [round(v, 6) for v in self.serie[-6:]],
            resultado["evaluacion"]["valores_reales"],
        )

    def test_mae_rmse_mape_presentes(self):
        resultado = ejecutar_herramienta("modelo_ma", {
            "valores": self.serie, "q": 1, "evaluar_modelo": True, "cantidad_prueba": 6,
        })
        metricas = resultado["evaluacion"]["metricas_prueba"]
        for clave in ("mae", "rmse", "mape"):
            self.assertIn(clave, metricas)

    def test_reajuste_con_toda_la_serie(self):
        resultado = ejecutar_herramienta("modelo_ma", {
            "valores": self.serie, "q": 1, "evaluar_modelo": True, "cantidad_prueba": 6,
        })
        self.assertEqual(resultado["n_observaciones"], len(self.serie))

    def test_pronostico_futuro_independiente_de_la_evaluacion(self):
        con_eval = ejecutar_herramienta("modelo_ma", {
            "valores": self.serie, "q": 1, "evaluar_modelo": True, "cantidad_prueba": 6, "pasos_pronostico": 3,
        })
        sin_eval = ejecutar_herramienta("modelo_ma", {
            "valores": self.serie, "q": 1, "evaluar_modelo": False, "pasos_pronostico": 3,
        })
        self.assertEqual(con_eval["pronostico"], sin_eval["pronostico"])

    def test_cantidad_de_prueba_respetada(self):
        resultado = ejecutar_herramienta("modelo_ma", {
            "valores": self.serie, "q": 1, "evaluar_modelo": True, "cantidad_prueba": 5,
        })
        self.assertEqual(resultado["evaluacion"]["n_prueba"], 5)

    def test_porcentaje_de_prueba_respetado(self):
        resultado = ejecutar_herramienta("modelo_ma", {
            "valores": self.serie, "q": 1, "evaluar_modelo": True, "porcentaje_prueba": 0.25,
        })
        self.assertEqual(resultado["evaluacion"]["n_prueba"], round(48 * 0.25))

    def test_entrenamiento_insuficiente_no_bloquea_ajuste_final(self):
        resultado = ejecutar_herramienta("modelo_ma", {
            "valores": self.serie, "q": 1, "evaluar_modelo": True, "cantidad_prueba": 45,
        })
        self.assertNotIn("error", resultado)
        self.assertFalse(resultado["evaluacion"]["ejecutada"])
        self.assertIn("motivo", resultado["evaluacion"])
        self.assertTrue(len(resultado["pronostico"]) >= 1)

    def test_mape_con_cero_en_la_prueba(self):
        serie = list(self.serie)
        serie[-1] = 0.0
        resultado = ejecutar_herramienta("modelo_ma", {
            "valores": serie, "q": 1, "evaluar_modelo": True, "cantidad_prueba": 3,
        })
        detalle = resultado["evaluacion"]["mape_detalle"]
        self.assertGreaterEqual(detalle["observaciones_excluidas_por_cero"], 1)


class FechasMATests(SimpleTestCase):
    def setUp(self):
        self.serie = _serie_ma1(n=36, seed=14)
        self.fechas = _fechas_mensuales(36)

    def test_fechas_mensuales(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": self.serie, "q": 1, "fechas": self.fechas})
        self.assertTrue(resultado["informacion_temporal"]["fechas_proporcionadas"])

    def test_frecuencia_inferida(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": self.serie, "q": 1, "fechas": self.fechas})
        self.assertEqual(resultado["informacion_temporal"]["frecuencia_inferida"], "MS")

    def test_frecuencia_explicita(self):
        resultado = ejecutar_herramienta("modelo_ma", {
            "valores": self.serie, "q": 1, "fechas": self.fechas, "frecuencia": "mensual",
        })
        self.assertEqual(resultado["informacion_temporal"]["frecuencia_solicitada"], "MS")

    def test_fechas_futuras(self):
        resultado = ejecutar_herramienta("modelo_ma", {
            "valores": self.serie, "q": 1, "fechas": self.fechas, "pasos_pronostico": 2,
        })
        self.assertEqual(resultado["fechas_pronostico"], ["2025-01-01", "2025-02-01"])

    def test_intervalos_con_fecha(self):
        resultado = ejecutar_herramienta("modelo_ma", {
            "valores": self.serie, "q": 1, "fechas": self.fechas, "pasos_pronostico": 2,
        })
        for intervalo, fecha in zip(resultado["intervalos_pronostico"], resultado["fechas_pronostico"]):
            self.assertEqual(intervalo["fecha"], fecha)

    def test_fechas_duplicadas(self):
        fechas = list(self.fechas)
        fechas[5] = fechas[4]
        resultado = ejecutar_herramienta("modelo_ma", {"valores": self.serie, "q": 1, "fechas": fechas})
        self.assertEqual(resultado.get("codigo_error"), "FECHAS_DUPLICADAS")

    def test_fechas_desordenadas(self):
        fechas = list(self.fechas)
        fechas[0], fechas[1] = fechas[1], fechas[0]
        resultado = ejecutar_herramienta("modelo_ma", {"valores": self.serie, "q": 1, "fechas": fechas})
        self.assertEqual(resultado.get("codigo_error"), "FECHAS_DESORDENADAS")

    def test_periodo_faltante(self):
        serie_incompleta = self.serie[:5] + self.serie[6:]
        fechas_incompletas = self.fechas[:5] + self.fechas[6:]
        resultado = ejecutar_herramienta("modelo_ma", {
            "valores": serie_incompleta, "q": 1, "fechas": fechas_incompletas, "frecuencia": "mensual",
        })
        self.assertFalse(resultado["informacion_temporal"]["serie_regular"])
        codigos = [a["codigo"] for a in resultado["advertencias"]]
        self.assertIn("PERIODOS_FALTANTES", codigos)

    def test_serie_sin_fechas(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": self.serie, "q": 1})
        self.assertFalse(resultado["informacion_temporal"]["fechas_proporcionadas"])
        self.assertIsNone(resultado["fechas_pronostico"])


class PedagogicaMATests(SimpleTestCase):
    def test_explicacion_de_errores_anteriores(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_ma1(), "q": 1})
        explicacion = resultado["explicacion_modelo"]
        self.assertIn("descripcion", explicacion)
        texto = explicacion["descripcion"].lower()
        self.assertTrue("error" in texto or "innovacion" in texto)

    def test_diferencia_con_promedio_movil(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_ma1(), "q": 1})
        explicacion = resultado["explicacion_modelo"]
        self.assertIn("diferencia_promedio_movil", explicacion)
        self.assertIn("promedio", explicacion["diferencia_promedio_movil"].lower())

    def test_advertencia_de_estacionariedad_cuando_corresponde(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_con_tendencia(), "q": 1})
        codigos = [a["codigo"] for a in resultado["advertencias"]]
        self.assertIn("SERIE_NO_ESTACIONARIA_PARA_MA", codigos)

    def test_identificacion_compara_con_acf_sin_forzar_seleccion(self):
        resultado = ejecutar_herramienta("modelo_ma", {"valores": _serie_ma1(), "q": 3})
        identificacion = resultado["identificacion"]
        self.assertEqual(identificacion["q_solicitado"], 3)
        self.assertIn("q_sugerido_acf", identificacion)
        # El q solicitado por el usuario se respeta aunque no coincida con la ACF.
        self.assertEqual(resultado["orden_q"], 3)


class RegresionARIMATests(SimpleTestCase):
    """Confirma que agregar `modelo_ma` no altera el comportamiento de `modelo_arima`."""

    def test_modelo_arima_sigue_funcionando(self):
        rng = np.random.default_rng(4)
        n = 24
        drift = 1.5
        valores = [100.0]
        for t in range(1, n):
            valores.append(valores[-1] + drift + rng.normal(scale=4))

        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": valores, "p": 0, "d": 1, "q": 0, "pasos_pronostico": 3, "con_constante": True,
        })
        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["orden"], {"p": 0, "d": 1, "q": 0})
        self.assertEqual(len(resultado["pronostico"]), 3)

    def test_ambas_herramientas_conviven_en_el_registro(self):
        self.assertIn("modelo_arima", TOOL_REGISTRY)
        self.assertIn("modelo_ma", TOOL_REGISTRY)
        self.assertIn("modelo_ar", TOOL_REGISTRY)
        self.assertIn("acf", TOOL_REGISTRY)
        self.assertIn("modelo_dickey_fuller", TOOL_REGISTRY)
