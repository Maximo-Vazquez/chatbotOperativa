"""Pruebas de extremo a extremo de la herramienta publica `modelo_arima`.

Ejecutan `ejecutar_herramienta("modelo_arima", ...)` tal como lo hace el
chatbot real (mismo loader dinamico de `apps/herramientas/tools.py`), para
cubrir el contrato completo: validacion, ajuste sobre el nucleo compartido
(`forecasting/engine.py`) y serializacion JSON de la respuesta.

Las series sinteticas usan semillas fijas (`numpy.random.default_rng`) para
que los resultados sean deterministas. Las comparaciones numericas usan
`assertAlmostEqual`/tolerancias en vez de igualdad exacta, dado que
statsmodels puede variar levemente segun la version instalada.
"""

import json

import numpy as np
from django.test import SimpleTestCase

from apps.herramientas.tools import TOOL_DEFINITIONS, TOOL_META, TOOL_REGISTRY, ejecutar_herramienta


def _serie_random_walk_con_drift(n=24, drift=1.5, escala_ruido=4.0, seed=4):
    """DGP de un ARIMA(0,1,0) con drift: paseo aleatorio con incremento medio constante."""
    rng = np.random.default_rng(seed)
    ruido = rng.normal(scale=escala_ruido, size=n)
    valores = [100.0]
    for t in range(1, n):
        valores.append(valores[-1] + drift + ruido[t])
    return valores


def _serie_ar1(n=40, phi=0.6, nivel=5.0, seed=1):
    rng = np.random.default_rng(seed)
    ruido = rng.normal(size=n)
    serie = np.zeros(n)
    for t in range(1, n):
        serie[t] = nivel + phi * (serie[t - 1] - nivel) + ruido[t]
    serie[0] = nivel
    return serie.tolist()


def _serie_ma1(n=40, theta=0.5, nivel=10.0, seed=2):
    rng = np.random.default_rng(seed)
    ruido = rng.normal(size=n + 1)
    return [nivel + ruido[t] + theta * ruido[t - 1] for t in range(1, n + 1)]


def _serie_arma11(n=40, phi=0.5, theta=0.3, nivel=20.0, seed=3):
    rng = np.random.default_rng(seed)
    ruido = rng.normal(size=n + 1)
    serie = [nivel]
    for t in range(1, n + 1):
        valor = nivel + phi * (serie[-1] - nivel) + ruido[t] + theta * ruido[t - 1]
        serie.append(valor)
    return serie[1:]


def _serie_con_curvatura(n=20, seed=5):
    """Serie con curvatura suficiente para que d=2 sea informativo."""
    rng = np.random.default_rng(seed)
    incremento = np.cumsum(rng.normal(scale=1.0, size=n)) + np.arange(n) * 0.5
    return (np.cumsum(incremento) + 50).tolist()


class ArimaCasosBasicosTests(SimpleTestCase):
    """Casos 1-6 del plan: distintas configuraciones de orden y drift."""

    def test_arima_0_1_0_con_drift(self):
        serie = _serie_random_walk_con_drift()
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie, "p": 0, "d": 1, "q": 0,
            "pasos_pronostico": 3, "con_constante": True,
        })
        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["tendencia_statsmodels"], "t")
        self.assertIn("drift", resultado["modelo"])

    def test_arima_0_1_0_sin_drift(self):
        serie = _serie_random_walk_con_drift()
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie, "p": 0, "d": 1, "q": 0,
            "pasos_pronostico": 3, "con_constante": False,
        })
        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["tendencia_statsmodels"], "n")
        self.assertNotIn("drift", resultado["modelo"])

    def test_arima_1_0_0(self):
        serie = _serie_ar1()
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie, "p": 1, "d": 0, "q": 0, "pasos_pronostico": 2,
        })
        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["orden"], {"p": 1, "d": 0, "q": 0})
        self.assertTrue(any(c["tipo"] == "autorregresivo" for c in resultado["detalle_coeficientes"]))

    def test_arima_0_0_1(self):
        serie = _serie_ma1()
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie, "p": 0, "d": 0, "q": 1, "pasos_pronostico": 2,
        })
        self.assertNotIn("error", resultado)
        self.assertTrue(any(c["tipo"] == "media_movil" for c in resultado["detalle_coeficientes"]))

    def test_arima_1_0_1(self):
        serie = _serie_arma11()
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie, "p": 1, "d": 0, "q": 1, "pasos_pronostico": 2,
        })
        self.assertNotIn("error", resultado)
        tipos = {c["tipo"] for c in resultado["detalle_coeficientes"]}
        self.assertIn("autorregresivo", tipos)
        self.assertIn("media_movil", tipos)

    def test_diferenciacion_orden_2_admitida_por_el_contrato(self):
        serie = _serie_con_curvatura()
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie, "p": 0, "d": 2, "q": 0, "pasos_pronostico": 2,
        })
        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["orden"]["d"], 2)
        self.assertEqual(resultado["tendencia_statsmodels"], "n")


class ArimaPronosticoTests(SimpleTestCase):
    """Casos 7-10: horizonte, intervalos y nivel de confianza."""

    def test_pronostico_de_un_paso(self):
        serie = _serie_ar1()
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie, "p": 1, "d": 0, "q": 0, "pasos_pronostico": 1,
        })
        self.assertEqual(resultado["pasos_pronostico"], 1)
        self.assertEqual(len(resultado["pronostico"]), 1)

    def test_pronostico_de_varios_pasos(self):
        serie = _serie_ar1()
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie, "p": 1, "d": 0, "q": 0, "pasos_pronostico": 6,
        })
        self.assertEqual(resultado["pasos_pronostico"], 6)
        self.assertEqual(len(resultado["pronostico"]), 6)
        self.assertEqual(len(resultado["intervalos_pronostico"]), 6)

    def test_intervalos_de_prediccion_son_finitos_y_consistentes(self):
        serie = _serie_random_walk_con_drift()
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie, "p": 0, "d": 1, "q": 0, "pasos_pronostico": 4,
        })
        for paso in resultado["intervalos_pronostico"]:
            self.assertTrue(np.isfinite(paso["limite_inferior"]))
            self.assertTrue(np.isfinite(paso["limite_superior"]))
            self.assertLessEqual(paso["limite_inferior"], paso["limite_superior"])
            self.assertLessEqual(paso["limite_inferior"], paso["pronostico"])
            self.assertLessEqual(paso["pronostico"], paso["limite_superior"])

    def test_nivel_de_confianza_configurable_amplia_el_intervalo(self):
        serie = _serie_random_walk_con_drift()
        args_base = {"valores": serie, "p": 0, "d": 1, "q": 0, "pasos_pronostico": 3}

        resultado_95 = ejecutar_herramienta("modelo_arima", {**args_base, "nivel_confianza": 0.95})
        resultado_99 = ejecutar_herramienta("modelo_arima", {**args_base, "nivel_confianza": 0.99})

        ancho_95 = (
            resultado_95["intervalos_pronostico"][0]["limite_superior"]
            - resultado_95["intervalos_pronostico"][0]["limite_inferior"]
        )
        ancho_99 = (
            resultado_99["intervalos_pronostico"][0]["limite_superior"]
            - resultado_99["intervalos_pronostico"][0]["limite_inferior"]
        )
        self.assertEqual(resultado_99["nivel_confianza"], 0.99)
        self.assertGreater(ancho_99, ancho_95)


class ArimaCoeficientesTests(SimpleTestCase):
    """Casos 11-16: coeficientes simples y detallados, p-valores, IC, AIC, BIC."""

    def setUp(self):
        self.resultado = ejecutar_herramienta("modelo_arima", {
            "valores": _serie_arma11(), "p": 1, "d": 0, "q": 1, "pasos_pronostico": 1,
        })

    def test_coeficientes_simples_presentes(self):
        self.assertIn("coeficientes", self.resultado)
        self.assertGreaterEqual(len(self.resultado["coeficientes"]), 2)

    def test_detalle_coeficientes_tiene_los_campos_esperados(self):
        for detalle in self.resultado["detalle_coeficientes"]:
            for campo in ("nombre", "tipo", "coeficiente", "error_estandar", "estadistico",
                          "p_value", "intervalo_confianza", "significativo_005"):
                self.assertIn(campo, detalle)

    def test_p_values_estan_entre_0_y_1_cuando_disponibles(self):
        for detalle in self.resultado["detalle_coeficientes"]:
            if detalle["p_value"] is not None:
                self.assertGreaterEqual(detalle["p_value"], 0.0)
                self.assertLessEqual(detalle["p_value"], 1.0)

    def test_intervalo_de_confianza_de_parametros_es_consistente(self):
        for detalle in self.resultado["detalle_coeficientes"]:
            ic = detalle["intervalo_confianza"]
            if ic is not None:
                self.assertLessEqual(ic["inferior"], ic["superior"])

    def test_aic_presente_y_finito(self):
        self.assertIsNotNone(self.resultado["aic"])
        self.assertTrue(np.isfinite(self.resultado["aic"]))

    def test_bic_presente_y_finito(self):
        self.assertIsNotNone(self.resultado["bic"])
        self.assertTrue(np.isfinite(self.resultado["bic"]))


class ArimaDiagnosticoResidualTests(SimpleTestCase):
    """Casos 17-20: diagnostico de residuos, Ljung-Box con model_df y advertencias."""

    def test_diagnostico_residual_completo(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": _serie_ar1(), "p": 1, "d": 0, "q": 0, "pasos_pronostico": 1,
        })
        diagnostico = resultado["diagnostico_residuos"]
        for campo in ("cantidad_residuos", "media", "varianza", "mse", "ljung_box", "advertencias"):
            self.assertIn(campo, diagnostico)

    def test_ljung_box_usa_model_df_igual_a_p_mas_q(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": _serie_arma11(), "p": 1, "d": 0, "q": 1, "pasos_pronostico": 1,
        })
        ljung_box = resultado["ljung_box"]
        if ljung_box["ejecutado"]:
            self.assertEqual(ljung_box["model_df"], 1 + 1)

    def test_ljung_box_no_ejecutable_con_muestra_muy_pequena_y_orden_alto(self):
        # p+q=7 deja muy pocos residuos disponibles frente al minimo de muestra.
        serie = _serie_arma11(n=12, seed=6)
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie, "p": 4, "d": 0, "q": 3, "pasos_pronostico": 1,
        })
        # El ajuste puede fallar (muestra insuficiente/no convergencia) o
        # producir un Ljung-Box no ejecutable; ambos son resultados validos,
        # lo que no debe ocurrir es una excepcion no controlada.
        if "error" not in resultado:
            self.assertIn("ejecutado", resultado["ljung_box"])

    def test_advertencias_es_una_lista_estructurada(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": _serie_ar1(), "p": 1, "d": 0, "q": 0, "pasos_pronostico": 1,
        })
        self.assertIsInstance(resultado["advertencias"], list)
        for advertencia in resultado["advertencias"]:
            for campo in ("codigo", "mensaje", "categoria", "severidad"):
                self.assertIn(campo, advertencia)


class ArimaErroresControladosTests(SimpleTestCase):
    """Caso 21: errores de ajuste/validacion siempre controlados, nunca una excepcion cruda."""

    def test_muestra_insuficiente_devuelve_error_controlado(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": [1, 2, 3], "p": 2, "d": 1, "q": 2,
        })
        self.assertIn("error", resultado)
        self.assertEqual(resultado["codigo_error"], "MUESTRA_INSUFICIENTE")

    def test_serie_constante_devuelve_error_controlado(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": [3, 3, 3, 3, 3, 3, 3, 3], "p": 1, "d": 0, "q": 0,
        })
        self.assertIn("error", resultado)
        self.assertEqual(resultado["codigo_error"], "SERIE_INVALIDA")

    def test_error_no_expone_traceback(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": [1, 2, 3], "p": 2, "d": 1, "q": 2,
        })
        self.assertNotIn("Traceback", resultado["error"])
        self.assertNotIn(".py", resultado["error"])


class ArimaCompatibilidadClavesTests(SimpleTestCase):
    """Caso 22: las claves del contrato anterior siguen presentes con el mismo significado."""

    CLAVES_ANTERIORES = (
        "modelo", "orden", "n_observaciones", "coeficientes", "aic", "bic",
        "mse_residuos", "media_residuos", "varianza_residuos", "ljung_box",
        "pasos_pronostico", "pronostico",
    )

    def test_claves_anteriores_siguen_presentes(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": _serie_random_walk_con_drift(), "p": 0, "d": 1, "q": 0,
            "pasos_pronostico": 3, "con_constante": True,
        })
        for clave in self.CLAVES_ANTERIORES:
            self.assertIn(clave, resultado)

    def test_orden_conserva_forma_p_d_q(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": _serie_ar1(), "p": 1, "d": 0, "q": 0,
        })
        self.assertEqual(set(resultado["orden"].keys()), {"p", "d", "q"})

    def test_mse_residuos_es_alias_de_mse_residuos_entrenamiento(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": _serie_ar1(), "p": 1, "d": 0, "q": 0,
        })
        self.assertEqual(resultado["mse_residuos"], resultado["mse_residuos_entrenamiento"])

    def test_llamada_sin_nivel_confianza_usa_el_default_095(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": _serie_ar1(), "p": 1, "d": 0, "q": 0,
        })
        self.assertEqual(resultado["nivel_confianza"], 0.95)


class RegistroDinamicoTests(SimpleTestCase):
    """Caso 23: la herramienta sigue registrandose por el mecanismo dinamico existente."""

    def test_modelo_arima_esta_en_tool_registry(self):
        self.assertIn("modelo_arima", TOOL_REGISTRY)

    def test_modelo_arima_tiene_tool_definition_valida(self):
        nombres = [definicion["function"]["name"] for definicion in TOOL_DEFINITIONS]
        self.assertIn("modelo_arima", nombres)

    def test_modelo_arima_tiene_tool_meta(self):
        self.assertIn("modelo_arima", TOOL_META)

    def test_argumentos_obligatorios_se_conservan(self):
        definicion = next(d for d in TOOL_DEFINITIONS if d["function"]["name"] == "modelo_arima")
        requeridos = definicion["function"]["parameters"]["required"]
        self.assertEqual(set(requeridos), {"valores", "p", "d", "q"})

    def test_nivel_confianza_es_argumento_opcional_nuevo(self):
        definicion = next(d for d in TOOL_DEFINITIONS if d["function"]["name"] == "modelo_arima")
        propiedades = definicion["function"]["parameters"]["properties"]
        requeridos = definicion["function"]["parameters"]["required"]
        self.assertIn("nivel_confianza", propiedades)
        self.assertNotIn("nivel_confianza", requeridos)


class ResultadoSerializableTests(SimpleTestCase):
    """Caso 24: el resultado completo es JSON-serializable de punta a punta."""

    def test_resultado_exitoso_es_json_serializable(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": _serie_random_walk_con_drift(), "p": 0, "d": 1, "q": 0,
            "pasos_pronostico": 3,
        })
        volcado = json.dumps(resultado, ensure_ascii=False)
        self.assertIsInstance(volcado, str)

    def test_resultado_de_error_es_json_serializable(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": [1, 2, 3], "p": 2, "d": 1, "q": 2,
        })
        json.dumps(resultado, ensure_ascii=False)


class ArimaRegresionSinteticaTests(SimpleTestCase):
    """Prueba de regresion de comportamiento con una serie sintetica de 24
    observaciones (no se encontro en el repositorio una serie fija del caso
    academico del chatbot 7A: no hay fixtures ni datos de ejemplo versionados
    en `apps/herramientas/` ni en `docs/chatbot/`). Cubre los mismos chequeos
    que pediria esa prueba: ajuste ARIMA(0,1,0) con drift, tres pronosticos
    finitos, intervalos finitos y diagnostico residual ejecutado.
    """

    def test_serie_24_observaciones_arima_0_1_0_con_drift(self):
        serie = _serie_random_walk_con_drift(n=24)
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": serie, "p": 0, "d": 1, "q": 0,
            "pasos_pronostico": 3, "con_constante": True,
        })

        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["n_observaciones"], 24)
        self.assertEqual(len(resultado["pronostico"]), 3)
        for valor in resultado["pronostico"]:
            self.assertTrue(np.isfinite(valor))
        for paso in resultado["intervalos_pronostico"]:
            self.assertTrue(np.isfinite(paso["limite_inferior"]))
            self.assertTrue(np.isfinite(paso["limite_superior"]))
        self.assertIn("diagnostico_residuos", resultado)
        self.assertGreater(resultado["diagnostico_residuos"]["cantidad_residuos"], 0)
        json.dumps(resultado, ensure_ascii=False)


class ArimaRegresionCompatibleConVersionAnteriorTests(SimpleTestCase):
    """Llamada con la firma exacta de la version anterior (sin nivel_confianza):
    debe seguir funcionando sin exigir el argumento nuevo."""

    def test_llamada_compatible_con_version_anterior(self):
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": _serie_random_walk_con_drift(),
            "p": 0,
            "d": 1,
            "q": 0,
            "pasos_pronostico": 3,
            "con_constante": True,
        })

        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["modelo"].startswith("ARIMA(0,1,0)"), True)
        self.assertEqual(resultado["pasos_pronostico"], 3)
        for valor in resultado["pronostico"]:
            self.assertTrue(np.isfinite(valor))
        json.dumps(resultado, ensure_ascii=False)
