"""Pruebas de extremo a extremo de la herramienta publica `modelo_sarima`.

Ejecutan `ejecutar_herramienta("modelo_sarima", ...)` tal como lo haria el
chatbot real (mismo loader dinamico de `apps/herramientas/tools.py`), para
cubrir: registro dinamico, validacion de ordenes/periodicidad estacional,
reutilizacion del nucleo compartido (motor con `seasonal_order`, metricas,
evaluacion, fechas, diagnostico), estacionariedad, ciclos, coherencia
frecuencia-periodicidad, identificacion estacional y compatibilidad con
ARIMA/MA. Series sinteticas con semillas fijas.
"""

import json

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from apps.herramientas.tools import TOOL_DEFINITIONS, TOOL_META, TOOL_REGISTRY, ejecutar_herramienta


def _serie_mensual_s12(n=48, seed=21, pendiente=0.5, amplitud=8.0, escala_ruido=2.0, nivel=50.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    estacional = amplitud * np.sin(2 * np.pi * t / 12)
    return (nivel + pendiente * t + estacional + rng.normal(scale=escala_ruido, size=n)).tolist()


def _serie_trimestral_s4(n=32, seed=22, pendiente=0.3, amplitud=15.0, escala_ruido=3.0, nivel=100.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    estacional = amplitud * np.sin(2 * np.pi * t / 4)
    return (nivel + pendiente * t + estacional + rng.normal(scale=escala_ruido, size=n)).tolist()


def _serie_diaria_s7(n=60, seed=23, amplitud=5.0, escala_ruido=1.0, nivel=20.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    estacional = amplitud * np.sin(2 * np.pi * t / 7)
    return (nivel + estacional + rng.normal(scale=escala_ruido, size=n)).tolist()


def _fechas(n, inicio, freq):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(inicio, periods=n, freq=freq)]


class ContratoSARIMATests(SimpleTestCase):
    def test_se_registra_dinamicamente(self):
        self.assertIn("modelo_sarima", TOOL_REGISTRY)

    def test_tool_definition_nombre_correcto(self):
        nombres = [d["function"]["name"] for d in TOOL_DEFINITIONS]
        self.assertIn("modelo_sarima", nombres)

    def test_tool_meta_existe(self):
        self.assertIn("modelo_sarima", TOOL_META)
        self.assertIn("label", TOOL_META["modelo_sarima"])

    def test_tool_function_ejecutable(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": _serie_mensual_s12(), "p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 12,
        })
        self.assertNotIn("error", resultado)

    def test_campos_obligatorios(self):
        definicion = next(d for d in TOOL_DEFINITIONS if d["function"]["name"] == "modelo_sarima")
        requeridos = definicion["function"]["parameters"]["required"]
        self.assertEqual(set(requeridos), {"valores", "p", "d", "q", "P", "D", "Q", "s"})

    def test_campos_opcionales_definidos(self):
        definicion = next(d for d in TOOL_DEFINITIONS if d["function"]["name"] == "modelo_sarima")
        propiedades = definicion["function"]["parameters"]["properties"]
        for campo in (
            "pasos_pronostico", "con_constante", "nivel_confianza", "fechas",
            "frecuencia", "evaluar_modelo", "cantidad_prueba", "porcentaje_prueba",
        ):
            self.assertIn(campo, propiedades)

    def test_respuesta_serializable(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": _serie_mensual_s12(), "p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 12,
        })
        json.dumps(resultado, ensure_ascii=False)


class OrdenesSARIMATests(SimpleTestCase):
    def setUp(self):
        self.serie = _serie_mensual_s12()
        self.base = {"valores": self.serie, "p": 0, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "s": 12}

    def test_todos_cero_salvo_estacional(self):
        resultado = ejecutar_herramienta("modelo_sarima", {**self.base, "P": 1, "D": 1, "Q": 0})
        self.assertNotIn("error", resultado)

    def test_P_uno(self):
        resultado = ejecutar_herramienta("modelo_sarima", {**self.base, "P": 1})
        self.assertNotIn("error", resultado)

    def test_Q_uno(self):
        resultado = ejecutar_herramienta("modelo_sarima", {**self.base, "Q": 1})
        self.assertNotIn("error", resultado)

    def test_D_uno(self):
        resultado = ejecutar_herramienta("modelo_sarima", {**self.base, "D": 1})
        self.assertNotIn("error", resultado)

    def test_orden_negativo(self):
        resultado = ejecutar_herramienta("modelo_sarima", {**self.base, "P": -1})
        self.assertEqual(resultado.get("codigo_error"), "ORDEN_ESTACIONAL_INVALIDO")

    def test_orden_decimal(self):
        resultado = ejecutar_herramienta("modelo_sarima", {**self.base, "Q": 1.5})
        self.assertEqual(resultado.get("codigo_error"), "ORDEN_ESTACIONAL_INVALIDO")

    def test_orden_booleano(self):
        resultado = ejecutar_herramienta("modelo_sarima", {**self.base, "D": True})
        self.assertEqual(resultado.get("codigo_error"), "ORDEN_ESTACIONAL_INVALIDO")

    def test_orden_excesivo(self):
        resultado = ejecutar_herramienta("modelo_sarima", {**self.base, "P": 10})
        self.assertEqual(resultado.get("codigo_error"), "ORDEN_ESTACIONAL_INVALIDO")

    def test_s_uno(self):
        resultado = ejecutar_herramienta("modelo_sarima", {**self.base, "s": 1})
        self.assertEqual(resultado.get("codigo_error"), "PERIODICIDAD_INVALIDA")

    def test_s_cero(self):
        resultado = ejecutar_herramienta("modelo_sarima", {**self.base, "s": 0})
        self.assertEqual(resultado.get("codigo_error"), "PERIODICIDAD_INVALIDA")

    def test_s_negativo(self):
        resultado = ejecutar_herramienta("modelo_sarima", {**self.base, "s": -12})
        self.assertEqual(resultado.get("codigo_error"), "PERIODICIDAD_INVALIDA")

    def test_s_decimal(self):
        resultado = ejecutar_herramienta("modelo_sarima", {**self.base, "s": 12.5})
        self.assertEqual(resultado.get("codigo_error"), "PERIODICIDAD_INVALIDA")

    def test_s_booleano(self):
        resultado = ejecutar_herramienta("modelo_sarima", {**self.base, "s": True})
        self.assertEqual(resultado.get("codigo_error"), "PERIODICIDAD_INVALIDA")

    def test_configuracion_demasiado_compleja_para_la_muestra(self):
        # p+q+P+Q+constante muy alto respecto de una serie corta: debe
        # rechazarse tecnicamente (MUESTRA_INSUFICIENTE) o, si el motor logra
        # ajustar, al menos advertir sobre la complejidad.
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": self.serie[:20], "p": 5, "d": 1, "q": 5, "P": 3, "D": 1, "Q": 3, "s": 12,
        })
        if "error" in resultado:
            self.assertEqual(resultado["codigo_error"], "MUESTRA_INSUFICIENTE")
        else:
            codigos = [a["codigo"] for a in resultado["advertencias"]]
            self.assertIn("CONFIGURACION_ESTACIONAL_COMPLEJA", codigos)


class SerieMensualTests(SimpleTestCase):
    def setUp(self):
        self.resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": _serie_mensual_s12(), "p": 1, "d": 1, "q": 1,
            "P": 1, "D": 1, "Q": 1, "s": 12, "pasos_pronostico": 6,
        })

    def test_ajuste_sin_error(self):
        self.assertNotIn("error", self.resultado)

    def test_nombre_sarima(self):
        self.assertEqual(self.resultado["modelo"], "SARIMA(1,1,1)(1,1,1,12)")

    def test_orden_estacional_correcto(self):
        self.assertEqual(self.resultado["orden_estacional"], {"P": 1, "D": 1, "Q": 1, "s": 12})

    def test_periodicidad_12(self):
        self.assertEqual(self.resultado["diferenciacion"]["periodicidad"], 12)

    def test_ciclos_calculados(self):
        self.assertEqual(self.resultado["n_ciclos_aproximados"], 4.0)

    def test_pronostico(self):
        self.assertEqual(len(self.resultado["pronostico"]), 6)
        for valor in self.resultado["pronostico"]:
            self.assertTrue(np.isfinite(valor))

    def test_intervalos(self):
        for intervalo in self.resultado["intervalos_pronostico"]:
            self.assertLessEqual(intervalo["limite_inferior"], intervalo["limite_superior"])

    def test_aic(self):
        self.assertTrue(np.isfinite(self.resultado["aic"]))

    def test_bic(self):
        self.assertTrue(np.isfinite(self.resultado["bic"]))

    def test_diagnostico(self):
        self.assertIn("diagnostico_residuos", self.resultado)
        self.assertGreater(self.resultado["diagnostico_residuos"]["cantidad_residuos"], 0)

    def test_serializacion(self):
        json.dumps(self.resultado, ensure_ascii=False)


class SerieTrimestralTests(SimpleTestCase):
    def setUp(self):
        self.serie = _serie_trimestral_s4()
        self.fechas = _fechas(32, "2016-01-01", "QS")
        self.resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": self.serie, "p": 1, "d": 0, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 4,
            "fechas": self.fechas, "pasos_pronostico": 4,
        })

    def test_fechas_trimestrales(self):
        self.assertTrue(self.resultado["informacion_temporal"]["fechas_proporcionadas"])

    def test_frecuencia_inferida(self):
        frecuencia = self.resultado["informacion_temporal"]["frecuencia_inferida"]
        self.assertEqual(frecuencia.split("-")[0], "QS")

    def test_coherencia_estacional_habitual(self):
        self.assertEqual(self.resultado["coherencia_estacional"]["clasificacion"], "habitual")

    def test_pronostico(self):
        self.assertEqual(len(self.resultado["pronostico"]), 4)

    def test_fechas_futuras(self):
        self.assertEqual(
            self.resultado["fechas_pronostico"],
            ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"],
        )

    def test_intervalos(self):
        self.assertEqual(len(self.resultado["intervalos_pronostico"]), 4)
        for intervalo, fecha in zip(self.resultado["intervalos_pronostico"], self.resultado["fechas_pronostico"]):
            self.assertEqual(intervalo["fecha"], fecha)


class SerieDiariaTests(SimpleTestCase):
    def setUp(self):
        self.serie = _serie_diaria_s7()
        self.fechas = _fechas(60, "2023-01-01", "D")
        self.resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": self.serie, "p": 1, "d": 0, "q": 0, "P": 0, "D": 1, "Q": 1, "s": 7,
            "fechas": self.fechas, "pasos_pronostico": 7,
        })

    def test_ciclo_semanal(self):
        self.assertEqual(self.resultado["diferenciacion"]["periodicidad"], 7)

    def test_fechas_diarias(self):
        self.assertTrue(self.resultado["informacion_temporal"]["fechas_proporcionadas"])

    def test_pronostico_futuro(self):
        self.assertEqual(len(self.resultado["fechas_pronostico"]), 7)
        self.assertGreater(pd.Timestamp(self.resultado["fechas_pronostico"][0]), pd.Timestamp(self.fechas[-1]))

    def test_diagnostico(self):
        self.assertIn("ljung_box", self.resultado)


class DiferenciacionEstacionalTests(SimpleTestCase):
    def setUp(self):
        self.serie = _serie_mensual_s12()

    def test_D_cero(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": self.serie, "p": 1, "d": 1, "q": 0, "P": 1, "D": 0, "Q": 0, "s": 12,
        })
        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["diferenciacion"]["estacional"], 0)

    def test_D_uno(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": self.serie, "p": 1, "d": 0, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
        })
        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["diferenciacion"]["estacional"], 1)

    def test_combinacion_d1_D1(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": self.serie, "p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 12,
        })
        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["diferenciacion"], {"regular": 1, "estacional": 1, "periodicidad": 12})

    def test_configuracion_no_compatible_con_constante_se_resuelve_sin_error(self):
        # d+D=2: statsmodels no admite ningun termino determinista; el motor
        # debe resolverlo a trend='n' automaticamente, sin lanzar un error crudo.
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": self.serie, "p": 1, "d": 1, "q": 0, "P": 0, "D": 1, "Q": 0, "s": 12,
            "con_constante": True,
        })
        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["tendencia_statsmodels"], "n")

    def test_informacion_de_diferenciacion_devuelta(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": self.serie, "p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 12,
        })
        self.assertIn("diferenciacion", resultado)
        for clave in ("regular", "estacional", "periodicidad"):
            self.assertIn(clave, resultado["diferenciacion"])

    def test_ausencia_de_doble_diferenciacion_manual(self):
        # n_observaciones debe reflejar la serie ORIGINAL (sin diferenciar
        # manualmente antes de pasarla al modelo).
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": self.serie, "p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 12,
        })
        self.assertEqual(resultado["n_observaciones"], len(self.serie))


class CiclosTests(SimpleTestCase):
    def test_menos_de_un_ciclo(self):
        serie = _serie_mensual_s12(n=8)
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": serie, "p": 0, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "s": 12,
        })
        self.assertNotIn("error", resultado)
        codigos = [a["codigo"] for a in resultado["advertencias"]]
        self.assertIn("CICLOS_ESTACIONALES_INSUFICIENTES", codigos)

    def test_entre_uno_y_dos_ciclos(self):
        serie = _serie_mensual_s12(n=18)
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": serie, "p": 0, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "s": 12,
        })
        self.assertNotIn("error", resultado)
        codigos = [a["codigo"] for a in resultado["advertencias"]]
        self.assertIn("CICLOS_ESTACIONALES_INSUFICIENTES", codigos)

    def test_exactamente_dos_ciclos(self):
        serie = _serie_mensual_s12(n=24)
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": serie, "p": 0, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "s": 12,
        })
        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["n_ciclos_aproximados"], 2.0)
        # Dos ciclos no deben presentarse como garantia estadistica: sigue
        # habiendo alguna advertencia de ciclos limitados/insuficientes.
        codigos = [a["codigo"] for a in resultado["advertencias"]]
        self.assertTrue(any("CICLOS" in c for c in codigos))

    def test_mas_de_tres_ciclos_sin_advertencia_de_ciclos(self):
        serie = _serie_mensual_s12(n=48)
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": serie, "p": 0, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "s": 12,
        })
        self.assertNotIn("error", resultado)
        codigos = [a["codigo"] for a in resultado["advertencias"]]
        self.assertNotIn("CICLOS_ESTACIONALES_INSUFICIENTES", codigos)
        self.assertNotIn("CICLOS_ESTACIONALES_LIMITADOS", codigos)

    def test_rechazo_tecnico_cuando_no_alcanza_ni_para_ajustar(self):
        serie = _serie_mensual_s12(n=6)
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": serie, "p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 12,
        })
        self.assertEqual(resultado.get("codigo_error"), "MUESTRA_INSUFICIENTE")

    def test_n_ciclos_aproximados_correcto(self):
        serie = _serie_mensual_s12(n=36)
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": serie, "p": 0, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "s": 12,
        })
        self.assertEqual(resultado["n_ciclos_aproximados"], 3.0)


class CoeficientesSARIMATests(SimpleTestCase):
    def setUp(self):
        self.resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": _serie_mensual_s12(), "p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 12,
        })

    def test_coeficiente_ar_regular(self):
        self.assertIn("ar.L1", self.resultado["coeficientes_regulares"])

    def test_coeficiente_ma_regular(self):
        self.assertIn("ma.L1", self.resultado["coeficientes_regulares"])

    def test_coeficiente_ar_estacional(self):
        self.assertIn("ar.S.L12", self.resultado["coeficientes_estacionales"])

    def test_coeficiente_ma_estacional(self):
        self.assertIn("ma.S.L12", self.resultado["coeficientes_estacionales"])

    def test_clasificacion_correcta(self):
        tipos = {d["nombre"]: d["tipo"] for d in self.resultado["detalle_coeficientes"]}
        self.assertEqual(tipos["ar.L1"], "autorregresivo")
        self.assertEqual(tipos["ma.L1"], "media_movil")
        self.assertEqual(tipos["ar.S.L12"], "autorregresivo_estacional")
        self.assertEqual(tipos["ma.S.L12"], "media_movil_estacional")

    def test_p_valores(self):
        for detalle in self.resultado["detalle_coeficientes"]:
            self.assertIn("p_value", detalle)

    def test_intervalos_de_confianza(self):
        for detalle in self.resultado["detalle_coeficientes"]:
            ic = detalle["intervalo_confianza"]
            if ic is not None:
                self.assertLessEqual(ic["inferior"], ic["superior"])

    def test_significancia(self):
        for detalle in self.resultado["detalle_coeficientes"]:
            self.assertIn("significativo_005", detalle)


class LjungBoxSARIMATests(SimpleTestCase):
    def test_model_df_es_p_q_P_Q(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": _serie_mensual_s12(), "p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 12,
        })
        ljung_box = resultado["ljung_box"]
        if ljung_box["ejecutado"]:
            self.assertEqual(ljung_box["model_df"], 1 + 1 + 1 + 1)

    def test_lag_mayor_que_model_df(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": _serie_mensual_s12(), "p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 12,
        })
        ljung_box = resultado["ljung_box"]
        if ljung_box["ejecutado"]:
            self.assertGreater(ljung_box["lags"], ljung_box["model_df"])

    def test_usa_lag_estacional_cuando_es_viable(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": _serie_mensual_s12(n=60), "p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
        })
        ljung_box = resultado["ljung_box"]
        if ljung_box["ejecutado"] and ljung_box["lags"] == 12:
            self.assertTrue(ljung_box["incluye_rezago_estacional"])

    def test_lag_alternativo_cuando_s_es_demasiado_grande(self):
        # Serie corta con s grande: el rezago estacional (=s) no deberia ser
        # viable (consumiria mas de la mitad de los residuos: con n=20 quedan
        # ~20 residuos, mitad=10 < s=12), asi que se recurre al criterio no
        # estacional o no se ejecuta la prueba.
        serie = _serie_mensual_s12(n=20, seed=30)
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": serie, "p": 1, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "s": 12,
        })
        if "error" not in resultado:
            ljung_box = resultado["ljung_box"]
            if ljung_box["ejecutado"]:
                self.assertNotEqual(ljung_box["lags"], 12)

    def test_muestra_pequena_sin_lag_valido_no_produce_nan(self):
        serie = _serie_mensual_s12(n=30, seed=31)
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": serie, "p": 1, "d": 0, "q": 1, "P": 1, "D": 0, "Q": 1, "s": 12,
        })
        if "error" not in resultado and not resultado["ljung_box"]["ejecutado"]:
            self.assertIn("motivo", resultado["ljung_box"])
            self.assertIsNone(resultado["ljung_box"]["estadistico"])

    def test_interpretacion_prudente(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": _serie_mensual_s12(), "p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 12,
        })
        interpretacion = resultado["ljung_box"]["interpretacion"].lower()
        self.assertNotIn("modelo valido", interpretacion)
        self.assertNotIn("modelo válido", interpretacion)

    def test_resultado_serializable(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": _serie_mensual_s12(), "p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 12,
        })
        json.dumps(resultado["ljung_box"], ensure_ascii=False)


class EvaluacionTemporalSARIMATests(SimpleTestCase):
    def setUp(self):
        self.serie = _serie_mensual_s12(n=60, seed=40)
        self.base = {
            "valores": self.serie, "p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
            "pasos_pronostico": 3,
        }

    def test_evaluacion_desactivada(self):
        resultado = ejecutar_herramienta("modelo_sarima", self.base)
        self.assertEqual(resultado["evaluacion"], {"ejecutada": False})

    def test_evaluacion_activada(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            **self.base, "evaluar_modelo": True, "cantidad_prueba": 12,
        })
        self.assertTrue(resultado["evaluacion"]["ejecutada"])

    def test_holdout_cronologico_usa_las_ultimas_observaciones(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            **self.base, "evaluar_modelo": True, "cantidad_prueba": 12,
        })
        self.assertEqual(
            [round(v, 6) for v in self.serie[-12:]],
            resultado["evaluacion"]["valores_reales"],
        )

    def test_entrenamiento_con_ciclos_suficientes(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            **self.base, "evaluar_modelo": True, "cantidad_prueba": 12,
        })
        # 60-12=48 observaciones de entrenamiento = 4 ciclos, mas que suficiente.
        self.assertTrue(resultado["evaluacion"]["ejecutada"])

    def test_entrenamiento_con_ciclos_insuficientes_omite_evaluacion(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            **self.base, "evaluar_modelo": True, "cantidad_prueba": 50,
        })
        self.assertNotIn("error", resultado)
        self.assertFalse(resultado["evaluacion"]["ejecutada"])
        self.assertIn("motivo", resultado["evaluacion"])
        # El ajuste final sigue produciendo pronostico igual.
        self.assertEqual(len(resultado["pronostico"]), 3)

    def test_prueba_menor_a_un_ciclo_genera_advertencia_informativa(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            **self.base, "evaluar_modelo": True, "cantidad_prueba": 5,
        })
        self.assertTrue(resultado["evaluacion"]["ejecutada"])
        codigos = [a["codigo"] for a in resultado["evaluacion"].get("advertencias", [])]
        self.assertIn("PRUEBA_NO_CUBRE_CICLO_COMPLETO", codigos)

    def test_mae_rmse_mape_presentes(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            **self.base, "evaluar_modelo": True, "cantidad_prueba": 12,
        })
        for clave in ("mae", "rmse", "mape"):
            self.assertIn(clave, resultado["evaluacion"]["metricas_prueba"])

    def test_reajuste_final_con_toda_la_serie(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            **self.base, "evaluar_modelo": True, "cantidad_prueba": 12,
        })
        self.assertEqual(resultado["n_observaciones"], len(self.serie))

    def test_pronostico_futuro_separado_de_la_evaluacion(self):
        con_eval = ejecutar_herramienta("modelo_sarima", {
            **self.base, "evaluar_modelo": True, "cantidad_prueba": 12,
        })
        sin_eval = ejecutar_herramienta("modelo_sarima", {**self.base, "evaluar_modelo": False})
        self.assertEqual(con_eval["pronostico"], sin_eval["pronostico"])

    def test_fechas_de_prueba_alineadas(self):
        fechas = _fechas(60, "2019-01-01", "MS")
        resultado = ejecutar_herramienta("modelo_sarima", {
            **self.base, "fechas": fechas, "evaluar_modelo": True, "cantidad_prueba": 12,
        })
        self.assertEqual(len(resultado["evaluacion"]["fechas_prueba"]), 12)
        self.assertEqual(resultado["evaluacion"]["fechas_prueba"], fechas[-12:])


class FechasSARIMATests(SimpleTestCase):
    def setUp(self):
        self.serie = _serie_mensual_s12(n=48)
        self.fechas = _fechas(48, "2020-01-01", "MS")

    def test_fechas_mensuales_regulares(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": self.serie, "p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
            "fechas": self.fechas,
        })
        self.assertTrue(resultado["informacion_temporal"]["serie_regular"])

    def test_fechas_duplicadas(self):
        fechas = list(self.fechas)
        fechas[5] = fechas[4]
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": self.serie, "p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12, "fechas": fechas,
        })
        self.assertEqual(resultado.get("codigo_error"), "FECHAS_DUPLICADAS")

    def test_fechas_desordenadas(self):
        fechas = list(self.fechas)
        fechas[0], fechas[1] = fechas[1], fechas[0]
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": self.serie, "p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12, "fechas": fechas,
        })
        self.assertEqual(resultado.get("codigo_error"), "FECHAS_DESORDENADAS")

    def test_periodos_faltantes(self):
        serie_incompleta = self.serie[:5] + self.serie[6:]
        fechas_incompletas = self.fechas[:5] + self.fechas[6:]
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": serie_incompleta, "p": 1, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "s": 12,
            "fechas": fechas_incompletas, "frecuencia": "mensual",
        })
        self.assertFalse(resultado["informacion_temporal"]["serie_regular"])

    def test_frecuencia_explicita_compatible(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": self.serie, "p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
            "fechas": self.fechas, "frecuencia": "mensual",
        })
        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["informacion_temporal"]["frecuencia_solicitada"], "MS")

    def test_frecuencia_explicita_incompatible(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": self.serie, "p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
            "fechas": self.fechas, "frecuencia": "diaria",
        })
        self.assertEqual(resultado.get("codigo_error"), "FRECUENCIA_INCONSISTENTE")

    def test_frecuencia_no_inferible(self):
        # Irregulares pero en orden cronologico estricto (self.fechas[3] es
        # 2020-04-01, posterior a 2020-02-19): no deben rechazarse por
        # desorden, solo no permitir inferir una frecuencia regular.
        fechas_irregulares = ["2020-01-01", "2020-01-03", "2020-02-19"] + self.fechas[3:]
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": self.serie, "p": 1, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "s": 12,
            "fechas": fechas_irregulares,
        })
        self.assertNotIn("error", resultado)
        self.assertIsNone(resultado["informacion_temporal"]["frecuencia_utilizada"])

    def test_serie_sin_fechas(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": self.serie, "p": 1, "d": 1, "q": 0, "P": 1, "D": 1, "Q": 0, "s": 12,
        })
        self.assertFalse(resultado["informacion_temporal"]["fechas_proporcionadas"])
        self.assertIsNone(resultado["fechas_pronostico"])


class AdvertenciasSARIMATests(SimpleTestCase):
    def test_menos_de_dos_ciclos(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": _serie_mensual_s12(n=18), "p": 0, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "s": 12,
        })
        codigos = [a["codigo"] for a in resultado["advertencias"]]
        self.assertIn("CICLOS_ESTACIONALES_INSUFICIENTES", codigos)

    def test_configuracion_compleja(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": _serie_mensual_s12(n=30), "p": 3, "d": 1, "q": 3,
            "P": 2, "D": 1, "Q": 2, "s": 12,
        })
        if "error" not in resultado:
            codigos = [a["codigo"] for a in resultado["advertencias"]]
            self.assertIn("CONFIGURACION_ESTACIONAL_COMPLEJA", codigos)

    def test_frecuencia_y_s_inusuales(self):
        fechas = _fechas(24, "2020-01-01", "MS")
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": _serie_mensual_s12(n=24), "p": 0, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "s": 6,
            "fechas": fechas,
        })
        codigos = [a["codigo"] for a in resultado["advertencias"]]
        self.assertIn("FRECUENCIA_Y_PERIODICIDAD_INUSUALES", codigos)

    def test_adf_no_determina_diferenciacion_estacional(self):
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": _serie_mensual_s12(), "p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 12,
        })
        codigos = [a["codigo"] for a in resultado["advertencias"]]
        self.assertIn("ADF_NO_DETERMINA_DIFERENCIACION_ESTACIONAL", codigos)

    def test_advertencias_no_silenciadas_con_parametros_iniciales_problematicos(self):
        rng = np.random.default_rng(50)
        serie_ruidosa = list(rng.normal(size=25))
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": serie_ruidosa, "p": 1, "d": 0, "q": 1, "P": 1, "D": 0, "Q": 1, "s": 4,
        })
        if "error" not in resultado:
            self.assertIsInstance(resultado["advertencias"], list)


class RegresionARIMAMATests(SimpleTestCase):
    """Confirma que agregar `modelo_sarima` (y ampliar el motor con
    `seasonal_order`) no altera el comportamiento de `modelo_arima` ni `modelo_ma`."""

    def test_modelo_arima_sigue_funcionando(self):
        rng = np.random.default_rng(4)
        n = 24
        valores = [100.0]
        for t in range(1, n):
            valores.append(valores[-1] + 1.5 + rng.normal(scale=4))
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": valores, "p": 0, "d": 1, "q": 0, "pasos_pronostico": 3, "con_constante": True,
        })
        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["orden"], {"p": 0, "d": 1, "q": 0})
        self.assertEqual(len(resultado["pronostico"]), 3)

    def test_modelo_ma_sigue_funcionando(self):
        rng = np.random.default_rng(7)
        n = 40
        ruido = rng.normal(scale=2.0, size=n + 1)
        serie = [20.0 + ruido[t] + 0.6 * ruido[t - 1] for t in range(1, n + 1)]
        resultado = ejecutar_herramienta("modelo_ma", {"valores": serie, "q": 1, "pasos_pronostico": 2})
        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["modelo"], "MA(1)")

    def test_las_tres_herramientas_conviven_en_el_registro(self):
        for nombre in ("modelo_arima", "modelo_ma", "modelo_sarima", "modelo_ar", "acf", "modelo_dickey_fuller"):
            self.assertIn(nombre, TOOL_REGISTRY)
