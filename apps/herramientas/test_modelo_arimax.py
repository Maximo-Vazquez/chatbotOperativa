"""Pruebas de extremo a extremo de la herramienta publica `modelo_arimax`.

Ejecutan `ejecutar_herramienta("modelo_arimax", ...)` tal como lo haria el
chatbot real (mismo loader dinamico de `apps/herramientas/tools.py`), para
cubrir: registro dinamico, validacion estructural de exogenas historicas y
futuras, reutilizacion del nucleo compartido (motor con `exog`, metricas,
evaluacion, fechas, diagnostico), multicolinealidad, controles basicos de
fuga de informacion, y compatibilidad con ARIMA/MA/SARIMA. Series sinteticas
con semillas fijas.
"""

import json

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from apps.herramientas.tools import TOOL_DEFINITIONS, TOOL_META, TOOL_REGISTRY, ejecutar_herramienta


def _serie_con_exogena_relevante(n=40, beta=1.5, seed=60, escala_ruido=1.0):
    rng = np.random.default_rng(seed)
    x = (20 + rng.normal(scale=3, size=n)).tolist()
    ruido = np.cumsum(rng.normal(scale=escala_ruido, size=n)) * 0.3
    y = (50 + beta * np.array(x) + ruido).tolist()
    return y, x


def _serie_con_binaria(n=40, beta=8.0, seed=61):
    rng = np.random.default_rng(seed)
    promo = rng.integers(0, 2, size=n).astype(float).tolist()
    ruido = np.cumsum(rng.normal(scale=1.0, size=n)) * 0.3
    y = (40 + beta * np.array(promo) + ruido).tolist()
    return y, promo


def _serie_multivariable(n=48, seed=62):
    rng = np.random.default_rng(seed)
    temp = (20 + rng.normal(scale=3, size=n)).tolist()
    promo = rng.integers(0, 2, size=n).astype(float).tolist()
    menor_relevancia = rng.normal(size=n).tolist()
    ruido = np.cumsum(rng.normal(scale=1.0, size=n)) * 0.3
    y = (50 + 1.5 * np.array(temp) + 8 * np.array(promo) + 0.05 * np.array(menor_relevancia) + ruido).tolist()
    return y, temp, promo, menor_relevancia


def _fechas(n, inicio="2020-01-01", freq="MS"):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(inicio, periods=n, freq=freq)]


class ContratoARIMAXTests(SimpleTestCase):
    def setUp(self):
        self.y, self.x = _serie_con_exogena_relevante()

    def test_se_registra_dinamicamente(self):
        self.assertIn("modelo_arimax", TOOL_REGISTRY)

    def test_tool_definition_nombre_correcto(self):
        nombres = [d["function"]["name"] for d in TOOL_DEFINITIONS]
        self.assertIn("modelo_arimax", nombres)

    def test_tool_meta_existe(self):
        self.assertIn("modelo_arimax", TOOL_META)

    def test_tool_function_ejecutable(self):
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0]}, "p": 1, "d": 0, "q": 0,
        })
        self.assertNotIn("error", resultado)

    def test_campos_obligatorios(self):
        definicion = next(d for d in TOOL_DEFINITIONS if d["function"]["name"] == "modelo_arimax")
        requeridos = definicion["function"]["parameters"]["required"]
        self.assertEqual(set(requeridos), {"valores", "variables_exogenas_historicas", "p", "d", "q"})

    def test_campos_opcionales_definidos(self):
        definicion = next(d for d in TOOL_DEFINITIONS if d["function"]["name"] == "modelo_arimax")
        propiedades = definicion["function"]["parameters"]["properties"]
        for campo in (
            "variables_exogenas_futuras", "pasos_pronostico", "con_constante",
            "nivel_confianza", "fechas", "fechas_exogenas_historicas",
            "fechas_exogenas_futuras", "frecuencia", "evaluar_modelo",
            "cantidad_prueba", "porcentaje_prueba",
        ):
            self.assertIn(campo, propiedades)

    def test_serializacion_json(self):
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0]}, "p": 1, "d": 0, "q": 0,
        })
        json.dumps(resultado, ensure_ascii=False)


class EstructuralesHistoricasTests(SimpleTestCase):
    def setUp(self):
        self.y, self.x = _serie_con_exogena_relevante()
        self.base = {"valores": self.y, "p": 1, "d": 0, "q": 0, "variables_exogenas_futuras": {"x": [21.0]}}

    def test_una_variable_valida(self):
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "variables_exogenas_historicas": {"x": self.x}})
        self.assertNotIn("error", resultado)

    def test_dos_variables_validas(self):
        _y, _x, promo, _ = _serie_multivariable()
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": _y, "p": 1, "d": 0, "q": 0,
            "variables_exogenas_historicas": {"x": _x, "promo": promo},
            "variables_exogenas_futuras": {"x": [21.0], "promo": [1.0]},
        })
        self.assertNotIn("error", resultado)

    def test_diccionario_vacio(self):
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "variables_exogenas_historicas": {}})
        self.assertEqual(resultado.get("codigo_error"), "EXOGENAS_HISTORICAS_REQUERIDAS")

    def test_columna_vacia(self):
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "variables_exogenas_historicas": {"x": []}})
        self.assertEqual(resultado.get("codigo_error"), "EXOGENA_NO_NUMERICA")

    def test_longitudes_historicas_diferentes(self):
        resultado = ejecutar_herramienta("modelo_arimax", {
            **self.base,
            "variables_exogenas_historicas": {"x": self.x, "z": self.x[:-5]},
            "variables_exogenas_futuras": {"x": [21.0], "z": [1.0]},
        })
        self.assertEqual(resultado.get("codigo_error"), "EXOGENA_LONGITUD_INCOMPATIBLE")

    def test_longitud_incompatible_con_objetivo(self):
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "variables_exogenas_historicas": {"x": self.x[:20]}})
        self.assertEqual(resultado.get("codigo_error"), "EXOGENA_LONGITUD_INCOMPATIBLE")

    def test_nombre_vacio(self):
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "variables_exogenas_historicas": {"": self.x}})
        self.assertEqual(resultado.get("codigo_error"), "EXOGENA_NO_NUMERICA")

    def test_valor_no_numerico(self):
        x_malo = list(self.x); x_malo[-1] = "no numero"
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "variables_exogenas_historicas": {"x": x_malo}})
        self.assertEqual(resultado.get("codigo_error"), "EXOGENA_NO_NUMERICA")

    def test_booleano(self):
        x_malo = list(self.x); x_malo[-1] = True
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "variables_exogenas_historicas": {"x": x_malo}})
        self.assertEqual(resultado.get("codigo_error"), "EXOGENA_NO_NUMERICA")

    def test_nan(self):
        x_malo = list(self.x); x_malo[-1] = float("nan")
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "variables_exogenas_historicas": {"x": x_malo}})
        self.assertEqual(resultado.get("codigo_error"), "EXOGENA_NO_FINITA")

    def test_infinito(self):
        x_malo = list(self.x); x_malo[-1] = float("inf")
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "variables_exogenas_historicas": {"x": x_malo}})
        self.assertEqual(resultado.get("codigo_error"), "EXOGENA_NO_FINITA")

    def test_estructura_anidada_invalida(self):
        x_malo = list(self.x); x_malo[-1] = [1, 2]
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "variables_exogenas_historicas": {"x": x_malo}})
        self.assertEqual(resultado.get("codigo_error"), "EXOGENA_NO_NUMERICA")


class ExogenasFuturasTests(SimpleTestCase):
    def setUp(self):
        self.y, self.x = _serie_con_exogena_relevante(n=40)
        self.base = {
            "valores": self.y, "p": 1, "d": 0, "q": 0,
            "variables_exogenas_historicas": {"x": self.x}, "pasos_pronostico": 2,
        }

    def test_estructura_valida(self):
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "variables_exogenas_futuras": {"x": [21.0, 22.0]}})
        self.assertNotIn("error", resultado)

    def test_falta_completa(self):
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "variables_exogenas_futuras": None})
        self.assertEqual(resultado.get("codigo_error"), "EXOGENAS_FUTURAS_REQUERIDAS")

    def test_falta_una_columna(self):
        _y, _x, promo, _ = _serie_multivariable(n=40)
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": _y, "p": 1, "d": 0, "q": 0, "pasos_pronostico": 2,
            "variables_exogenas_historicas": {"x": self.x, "promo": promo},
            "variables_exogenas_futuras": {"x": [21.0, 22.0]},
        })
        self.assertEqual(resultado.get("codigo_error"), "EXOGENAS_COLUMNAS_INCOMPATIBLES")

    def test_columna_adicional(self):
        resultado = ejecutar_herramienta("modelo_arimax", {
            **self.base, "variables_exogenas_futuras": {"x": [21.0, 22.0], "extra": [1.0, 2.0]},
        })
        self.assertEqual(resultado.get("codigo_error"), "EXOGENAS_COLUMNAS_INCOMPATIBLES")

    def test_longitud_menor_al_horizonte(self):
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "variables_exogenas_futuras": {"x": [21.0]}})
        self.assertEqual(resultado.get("codigo_error"), "HORIZONTE_EXOGENAS_INCOMPATIBLE")

    def test_longitud_mayor_al_horizonte(self):
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "variables_exogenas_futuras": {"x": [21.0, 22.0, 23.0]}})
        self.assertEqual(resultado.get("codigo_error"), "HORIZONTE_EXOGENAS_INCOMPATIBLE")

    def test_nan(self):
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "variables_exogenas_futuras": {"x": [21.0, float("nan")]}})
        self.assertEqual(resultado.get("codigo_error"), "EXOGENA_NO_FINITA")

    def test_infinito(self):
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "variables_exogenas_futuras": {"x": [21.0, float("inf")]}})
        self.assertEqual(resultado.get("codigo_error"), "EXOGENA_NO_FINITA")

    def test_orden_interno_corregido_de_forma_segura(self):
        # El diccionario futuro llega en orden distinto al historico
        # (promo antes que x): el motor debe reordenar para que coincida.
        _y, _x, promo, _ = _serie_multivariable(n=40)
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": _y, "p": 1, "d": 0, "q": 0, "pasos_pronostico": 2,
            "variables_exogenas_historicas": {"x": self.x, "promo": promo},
            "variables_exogenas_futuras": {"promo": [1.0, 0.0], "x": [21.0, 22.0]},
        })
        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["variables_exogenas"]["nombres"], ["x", "promo"])

    def test_pronostico_sin_inventar_valores(self):
        # Cambiar el valor futuro cambia el pronostico: confirma que se usa
        # el valor real recibido, no un valor inventado/derivado.
        resultado_a = ejecutar_herramienta("modelo_arimax", {**self.base, "variables_exogenas_futuras": {"x": [21.0, 22.0]}})
        resultado_b = ejecutar_herramienta("modelo_arimax", {**self.base, "variables_exogenas_futuras": {"x": [40.0, 41.0]}})
        self.assertNotEqual(resultado_a["pronostico"], resultado_b["pronostico"])


class VariableRelevanteTests(SimpleTestCase):
    def setUp(self):
        self.y, self.x = _serie_con_exogena_relevante(beta=1.5, n=50)
        self.resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0, 22.0]},
            "p": 1, "d": 0, "q": 0, "pasos_pronostico": 2,
        })

    def test_ajuste_sin_error(self):
        self.assertNotIn("error", self.resultado)

    def test_coeficiente_exogeno_presente(self):
        self.assertIn("x", self.resultado["coeficientes_exogenos"])

    def test_signo_razonable(self):
        # beta generador es positivo (1.5): el coeficiente estimado deberia serlo tambien.
        self.assertGreater(self.resultado["coeficientes_exogenos"]["x"], 0)

    def test_pronostico_finito(self):
        for valor in self.resultado["pronostico"]:
            self.assertTrue(np.isfinite(valor))

    def test_intervalos(self):
        for intervalo in self.resultado["intervalos_pronostico"]:
            self.assertLessEqual(intervalo["limite_inferior"], intervalo["limite_superior"])

    def test_evaluacion_disponible(self):
        resultado_eval = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0, 22.0]},
            "p": 1, "d": 0, "q": 0, "pasos_pronostico": 2,
            "evaluar_modelo": True, "cantidad_prueba": 8,
        })
        self.assertTrue(resultado_eval["evaluacion"]["ejecutada"])

    def test_serializable(self):
        json.dumps(self.resultado, ensure_ascii=False)


class VariableBinariaTests(SimpleTestCase):
    def setUp(self):
        self.y, self.promo = _serie_con_binaria()
        self.resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "variables_exogenas_historicas": {"promocion": self.promo},
            "variables_exogenas_futuras": {"promocion": [1.0, 0.0]},
            "p": 1, "d": 0, "q": 0, "pasos_pronostico": 2,
        })

    def test_ajuste(self):
        self.assertNotIn("error", self.resultado)

    def test_coeficiente(self):
        self.assertIn("promocion", self.resultado["coeficientes_exogenos"])

    def test_p_valor(self):
        detalle = next(d for d in self.resultado["detalle_coeficientes"] if d["nombre"] == "promocion")
        self.assertIn("p_value", detalle)

    def test_pronostico_con_escenarios_binarios_distintos(self):
        con_promo = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "variables_exogenas_historicas": {"promocion": self.promo},
            "variables_exogenas_futuras": {"promocion": [1.0, 1.0]},
            "p": 1, "d": 0, "q": 0, "pasos_pronostico": 2,
        })
        sin_promo = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "variables_exogenas_historicas": {"promocion": self.promo},
            "variables_exogenas_futuras": {"promocion": [0.0, 0.0]},
            "p": 1, "d": 0, "q": 0, "pasos_pronostico": 2,
        })
        self.assertNotEqual(con_promo["pronostico"], sin_promo["pronostico"])


class MultiplesVariablesTests(SimpleTestCase):
    def setUp(self):
        self.y, self.temp, self.promo, self.menor = _serie_multivariable()
        self.resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y,
            "variables_exogenas_historicas": {"temperatura": self.temp, "promocion": self.promo, "menor": self.menor},
            "variables_exogenas_futuras": {"temperatura": [21.0], "promocion": [1.0], "menor": [0.1]},
            "p": 1, "d": 0, "q": 0, "pasos_pronostico": 1,
        })

    def test_nombres_preservados(self):
        self.assertEqual(
            set(self.resultado["variables_exogenas"]["nombres"]),
            {"temperatura", "promocion", "menor"},
        )

    def test_orden_estable(self):
        self.assertEqual(self.resultado["variables_exogenas"]["nombres"], ["temperatura", "promocion", "menor"])

    def test_coeficientes(self):
        for nombre in ("temperatura", "promocion", "menor"):
            self.assertIn(nombre, self.resultado["coeficientes_exogenos"])

    def test_detalle_estadistico(self):
        for detalle in self.resultado["detalle_coeficientes"]:
            if detalle["nombre"] in ("temperatura", "promocion", "menor"):
                self.assertEqual(detalle["tipo"], "exogena")

    def test_diagnostico_multicolinealidad_presente(self):
        self.assertIn("diagnostico_multicolinealidad", self.resultado)
        self.assertIn("clasificacion", self.resultado["diagnostico_multicolinealidad"])


class VariableConstanteTests(SimpleTestCase):
    def setUp(self):
        self.y, self.x = _serie_con_exogena_relevante(n=30)

    def test_variable_completamente_constante(self):
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y,
            "variables_exogenas_historicas": {"x": self.x, "constante": [5.0] * 30},
            "variables_exogenas_futuras": {"x": [21.0], "constante": [5.0]},
            "p": 1, "d": 0, "q": 0, "con_constante": False,
        })
        self.assertNotIn("error", resultado)
        codigos = [a["codigo"] for a in resultado["advertencias"]]
        self.assertIn("EXOGENA_CONSTANTE", codigos)

    def test_constante_junto_con_con_constante_true_es_degenerada(self):
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y,
            "variables_exogenas_historicas": {"x": self.x, "constante": [5.0] * 30},
            "variables_exogenas_futuras": {"x": [21.0], "constante": [5.0]},
            "p": 1, "d": 0, "q": 0, "con_constante": True,
        })
        self.assertEqual(resultado.get("codigo_error"), "EXOGENAS_MATRIZ_DEGENERADA")

    def test_advertencia_presente(self):
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y,
            "variables_exogenas_historicas": {"x": self.x, "constante": [5.0] * 30},
            "variables_exogenas_futuras": {"x": [21.0], "constante": [5.0]},
            "p": 1, "d": 0, "q": 0, "con_constante": False,
        })
        advertencia = next(a for a in resultado["advertencias"] if a["codigo"] == "EXOGENA_CONSTANTE")
        self.assertEqual(advertencia["severidad"], "advertencia")

    def test_resultado_controlado_no_excepcion_cruda(self):
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y,
            "variables_exogenas_historicas": {"x": self.x, "constante": [5.0] * 30},
            "variables_exogenas_futuras": {"x": [21.0], "constante": [5.0]},
            "p": 1, "d": 0, "q": 0, "con_constante": True,
        })
        self.assertIn("error", resultado)
        self.assertIn("codigo_error", resultado)


class DuplicacionTests(SimpleTestCase):
    def test_dos_columnas_identicas(self):
        y, x = _serie_con_exogena_relevante(n=30)
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": y,
            "variables_exogenas_historicas": {"x": x, "x_copia": list(x)},
            "variables_exogenas_futuras": {"x": [21.0], "x_copia": [21.0]},
            "p": 1, "d": 0, "q": 0,
        })
        self.assertEqual(resultado.get("codigo_error"), "EXOGENAS_DUPLICADAS")

    def test_no_elimina_silenciosamente(self):
        y, x = _serie_con_exogena_relevante(n=30)
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": y,
            "variables_exogenas_historicas": {"x": x, "x_copia": list(x)},
            "variables_exogenas_futuras": {"x": [21.0], "x_copia": [21.0]},
            "p": 1, "d": 0, "q": 0,
        })
        self.assertIn("error", resultado)


class MulticolinealidadTests(SimpleTestCase):
    def test_variables_independientes(self):
        rng = np.random.default_rng(70)
        n = 60
        a = rng.normal(size=n).tolist()
        b = rng.normal(size=n).tolist()
        y = (10 + np.array(a) + np.array(b) + rng.normal(scale=0.5, size=n)).tolist()
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": y, "variables_exogenas_historicas": {"a": a, "b": b},
            "variables_exogenas_futuras": {"a": [0.0], "b": [0.0]},
            "p": 0, "d": 0, "q": 0,
        })
        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["diagnostico_multicolinealidad"]["clasificacion"], "sin_senales_relevantes")

    def test_correlacion_superior_a_095(self):
        rng = np.random.default_rng(71)
        n = 60
        a = rng.normal(size=n)
        b = a + rng.normal(scale=0.01, size=n)  # casi identica a `a`
        y = (10 + a + rng.normal(scale=0.5, size=n)).tolist()
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": y, "variables_exogenas_historicas": {"a": a.tolist(), "b": b.tolist()},
            "variables_exogenas_futuras": {"a": [0.0], "b": [0.0]},
            "p": 0, "d": 0, "q": 0,
        })
        self.assertNotIn("error", resultado)
        self.assertGreaterEqual(len(resultado["diagnostico_multicolinealidad"]["correlaciones_altas"]), 1)
        codigos = [adv["codigo"] for adv in resultado["advertencias"]]
        self.assertIn("EXOGENAS_ALTAMENTE_CORRELACIONADAS", codigos)

    def test_numero_condicion_alto(self):
        y, x = _serie_con_exogena_relevante(n=40)
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": y,
            "variables_exogenas_historicas": {"x_grande": (np.array(x) * 1000).tolist()},
            "variables_exogenas_futuras": {"x_grande": [21000.0]},
            "p": 1, "d": 0, "q": 0, "con_constante": True,
        })
        if "error" not in resultado and resultado["diagnostico_multicolinealidad"]["numero_condicion"] is not None:
            self.assertTrue(
                resultado["diagnostico_multicolinealidad"]["numero_condicion"] > 30
                or resultado["diagnostico_multicolinealidad"]["clasificacion"] != "sin_senales_relevantes"
            )

    def test_matriz_sin_rango_completo(self):
        y, x = _serie_con_exogena_relevante(n=30)
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": y,
            "variables_exogenas_historicas": {"x": x, "doble_x": (np.array(x) * 2).tolist()},
            "variables_exogenas_futuras": {"x": [21.0], "doble_x": [42.0]},
            "p": 1, "d": 0, "q": 0,
        })
        self.assertEqual(resultado.get("codigo_error"), "EXOGENAS_MATRIZ_DEGENERADA")

    def test_escalas_muy_diferentes_genera_advertencia(self):
        rng = np.random.default_rng(72)
        n = 40
        chico = rng.normal(scale=0.01, size=n).tolist()
        grande = rng.normal(scale=1000, size=n).tolist()
        y = (10 + rng.normal(size=n)).tolist()
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": y, "variables_exogenas_historicas": {"chico": chico, "grande": grande},
            "variables_exogenas_futuras": {"chico": [0.0], "grande": [0.0]},
            "p": 0, "d": 0, "q": 0,
        })
        if "error" not in resultado:
            codigos = [a["codigo"] for a in resultado["advertencias"]]
            self.assertIn("ESCALAS_EXOGENAS_MUY_DIFERENTES", codigos)


class FugaInformacionTests(SimpleTestCase):
    def test_exogena_identica_al_objetivo(self):
        y, x = _serie_con_exogena_relevante(n=30)
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": y, "variables_exogenas_historicas": {"copia_objetivo": list(y)},
            "variables_exogenas_futuras": {"copia_objetivo": [90.0]},
            "p": 1, "d": 0, "q": 0,
        })
        self.assertNotIn("error", resultado)
        self.assertIn("copia_objetivo", resultado["diagnostico_fuga_informacion"]["variables_identicas_al_objetivo"])
        codigos = [a["codigo"] for a in resultado["advertencias"]]
        self.assertIn("POSIBLE_FUGA_INFORMACION", codigos)

    def test_exogena_casi_identica(self):
        rng = np.random.default_rng(73)
        n = 40
        y = (50 + rng.normal(size=n) * 5).tolist()
        casi_igual = (np.array(y) + rng.normal(scale=0.01, size=n)).tolist()
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": y, "variables_exogenas_historicas": {"casi_igual": casi_igual},
            "variables_exogenas_futuras": {"casi_igual": [50.0]},
            "p": 0, "d": 0, "q": 0,
        })
        if "error" not in resultado:
            correlaciones = resultado["diagnostico_fuga_informacion"]["variables_correlacion_casi_perfecta_objetivo"]
            self.assertGreaterEqual(len(correlaciones), 1)

    def test_nombre_sospechoso(self):
        y, x = _serie_con_exogena_relevante(n=30)
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": y, "variables_exogenas_historicas": {"venta_futura": x},
            "variables_exogenas_futuras": {"venta_futura": [21.0]},
            "p": 1, "d": 0, "q": 0,
        })
        self.assertNotIn("error", resultado)
        self.assertIn("venta_futura", resultado["diagnostico_fuga_informacion"]["nombres_sospechosos"])

    def test_variable_legitimamente_correlacionada_no_es_error(self):
        y, x = _serie_con_exogena_relevante(n=40, beta=1.5)
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": y, "variables_exogenas_historicas": {"x": x},
            "variables_exogenas_futuras": {"x": [21.0]},
            "p": 1, "d": 0, "q": 0,
        })
        self.assertNotIn("error", resultado)

    def test_advertencia_sin_afirmacion_absoluta(self):
        rng = np.random.default_rng(73)
        n = 40
        y = (50 + rng.normal(size=n) * 5).tolist()
        casi_igual = (np.array(y) + rng.normal(scale=0.01, size=n)).tolist()
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": y, "variables_exogenas_historicas": {"casi_igual": casi_igual},
            "variables_exogenas_futuras": {"casi_igual": [50.0]},
            "p": 0, "d": 0, "q": 0,
        })
        if "error" not in resultado:
            for adv in resultado["advertencias"]:
                if adv["codigo"] == "EXOGENA_CORRELACION_CASI_PERFECTA_OBJETIVO":
                    self.assertIn("no confirma fuga", adv["mensaje"])

    def test_explicacion_de_disponibilidad_temporal(self):
        y, x = _serie_con_exogena_relevante(n=30)
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": y, "variables_exogenas_historicas": {"x": x},
            "variables_exogenas_futuras": {"x": [21.0]}, "p": 1, "d": 0, "q": 0,
        })
        self.assertIn("nota_disponibilidad_temporal", resultado["diagnostico_fuga_informacion"])


class EvaluacionTemporalARIMAXTests(SimpleTestCase):
    def setUp(self):
        self.y, self.x = _serie_con_exogena_relevante(n=48, seed=80)
        self.base = {
            "valores": self.y, "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0, 22.0]},
            "p": 1, "d": 0, "q": 0, "pasos_pronostico": 2,
        }

    def test_holdout_cronologico(self):
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "evaluar_modelo": True, "cantidad_prueba": 8})
        self.assertEqual(
            [round(v, 6) for v in self.y[-8:]],
            resultado["evaluacion"]["valores_reales"],
        )

    def test_entrenamiento_sin_datos_futuros_del_objetivo(self):
        # Verificado indirectamente: el pronostico final (reajuste con TODA
        # la serie) es independiente de si se evaluo o no.
        con_eval = ejecutar_herramienta("modelo_arimax", {**self.base, "evaluar_modelo": True, "cantidad_prueba": 8})
        sin_eval = ejecutar_herramienta("modelo_arimax", {**self.base, "evaluar_modelo": False})
        self.assertEqual(con_eval["pronostico"], sin_eval["pronostico"])

    def test_mae_rmse_mape(self):
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "evaluar_modelo": True, "cantidad_prueba": 8})
        for clave in ("mae", "rmse", "mape"):
            self.assertIn(clave, resultado["evaluacion"]["metricas_prueba"])

    def test_reajuste_completo(self):
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "evaluar_modelo": True, "cantidad_prueba": 8})
        self.assertEqual(resultado["n_observaciones"], len(self.y))

    def test_pronostico_futuro_con_exogenas_futuras(self):
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "evaluar_modelo": True, "cantidad_prueba": 8})
        self.assertEqual(len(resultado["pronostico"]), 2)

    def test_evaluacion_condicionada_documentada(self):
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "evaluar_modelo": True, "cantidad_prueba": 8})
        self.assertEqual(resultado["evaluacion"]["tipo"], "condicionada_a_exogenas_observadas")
        codigos = [a["codigo"] for a in resultado["evaluacion"].get("advertencias", [])]
        self.assertIn("EVALUACION_CON_EXOGENAS_OBSERVADAS", codigos)

    def test_muestra_insuficiente_para_evaluar_no_bloquea_ajuste(self):
        resultado = ejecutar_herramienta("modelo_arimax", {**self.base, "evaluar_modelo": True, "cantidad_prueba": 44})
        self.assertNotIn("error", resultado)
        self.assertFalse(resultado["evaluacion"]["ejecutada"])
        self.assertEqual(len(resultado["pronostico"]), 2)

    def test_cero_en_objetivo_de_prueba_mape(self):
        y = list(self.y)
        y[-1] = 0.0
        resultado = ejecutar_herramienta("modelo_arimax", {
            **self.base, "valores": y, "evaluar_modelo": True, "cantidad_prueba": 3,
        })
        if resultado["evaluacion"]["ejecutada"]:
            detalle = resultado["evaluacion"]["mape_detalle"]
            self.assertGreaterEqual(detalle["observaciones_excluidas_por_cero"], 1)

    def test_evaluacion_desactivada_por_defecto(self):
        resultado = ejecutar_herramienta("modelo_arimax", self.base)
        self.assertEqual(resultado["evaluacion"], {"ejecutada": False})


class FechasARIMAXTests(SimpleTestCase):
    def setUp(self):
        self.y, self.x = _serie_con_exogena_relevante(n=36, seed=90)
        self.fechas = _fechas(36)

    def test_fechas_historicas_alineadas_posicionalmente(self):
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0]},
            "p": 1, "d": 0, "q": 0, "fechas": self.fechas,
        })
        self.assertEqual(
            resultado["informacion_temporal"]["alineacion_exogenas"],
            {"metodo": "posicional", "fechas_exogenas_proporcionadas": False},
        )

    def test_fechas_exogenas_historicas_iguales(self):
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0]},
            "p": 1, "d": 0, "q": 0, "fechas": self.fechas,
            "fechas_exogenas_historicas": self.fechas,
        })
        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["informacion_temporal"]["alineacion_exogenas"]["metodo"], "fechas_propias")

    def test_fechas_desalineadas(self):
        fechas_otras = _fechas(36, inicio="2021-06-01")
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0]},
            "p": 1, "d": 0, "q": 0, "fechas": self.fechas,
            "fechas_exogenas_historicas": fechas_otras,
        })
        self.assertEqual(resultado.get("codigo_error"), "FECHAS_EXOGENAS_DESALINEADAS")

    def test_fechas_futuras_validas(self):
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0]},
            "p": 1, "d": 0, "q": 0, "fechas": self.fechas, "frecuencia": "mensual",
            "fechas_exogenas_futuras": ["2023-01-01"],
        })
        self.assertNotIn("error", resultado)

    def test_fechas_futuras_incompatibles(self):
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0]},
            "p": 1, "d": 0, "q": 0, "fechas": self.fechas, "frecuencia": "mensual",
            "fechas_exogenas_futuras": ["2099-01-01"],
        })
        self.assertEqual(resultado.get("codigo_error"), "FECHAS_EXOGENAS_DESALINEADAS")

    def test_alineacion_por_posicion_sin_fechas_exogenas(self):
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0]},
            "p": 1, "d": 0, "q": 0,
        })
        self.assertFalse(resultado["informacion_temporal"]["alineacion_exogenas"]["fechas_exogenas_proporcionadas"])

    def test_frecuencia_mensual(self):
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0]},
            "p": 1, "d": 0, "q": 0, "fechas": self.fechas,
        })
        self.assertEqual(resultado["informacion_temporal"]["frecuencia_inferida"], "MS")

    def test_frecuencia_diaria(self):
        fechas_diarias = _fechas(36, freq="D")
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0]},
            "p": 1, "d": 0, "q": 0, "fechas": fechas_diarias,
        })
        self.assertEqual(resultado["informacion_temporal"]["frecuencia_inferida"], "D")

    def test_periodos_faltantes(self):
        serie_incompleta = self.y[:5] + self.y[6:]
        x_incompleta = self.x[:5] + self.x[6:]
        fechas_incompletas = self.fechas[:5] + self.fechas[6:]
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": serie_incompleta, "variables_exogenas_historicas": {"x": x_incompleta},
            "variables_exogenas_futuras": {"x": [21.0]},
            "p": 1, "d": 0, "q": 0, "fechas": fechas_incompletas, "frecuencia": "mensual",
        })
        self.assertFalse(resultado["informacion_temporal"]["serie_regular"])

    def test_fechas_duplicadas(self):
        fechas_dup = list(self.fechas); fechas_dup[3] = fechas_dup[2]
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0]},
            "p": 1, "d": 0, "q": 0, "fechas": fechas_dup,
        })
        self.assertEqual(resultado.get("codigo_error"), "FECHAS_DUPLICADAS")

    def test_fechas_desordenadas(self):
        fechas_des = list(self.fechas); fechas_des[0], fechas_des[1] = fechas_des[1], fechas_des[0]
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0]},
            "p": 1, "d": 0, "q": 0, "fechas": fechas_des,
        })
        self.assertEqual(resultado.get("codigo_error"), "FECHAS_DESORDENADAS")


class CoeficientesARIMAXTests(SimpleTestCase):
    def setUp(self):
        self.y, self.x = _serie_con_exogena_relevante(n=40)
        self.resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0]},
            "p": 1, "d": 0, "q": 0,
        })

    def test_identificacion_de_parametros_exogenos(self):
        self.assertIn("x", self.resultado["coeficientes_exogenos"])

    def test_clasificacion_exogena(self):
        detalle = next(d for d in self.resultado["detalle_coeficientes"] if d["nombre"] == "x")
        self.assertEqual(detalle["tipo"], "exogena")

    def test_error_estandar(self):
        detalle = next(d for d in self.resultado["detalle_coeficientes"] if d["nombre"] == "x")
        self.assertIsNotNone(detalle["error_estandar"])

    def test_estadistico(self):
        detalle = next(d for d in self.resultado["detalle_coeficientes"] if d["nombre"] == "x")
        self.assertIsNotNone(detalle["estadistico"])

    def test_p_valor(self):
        detalle = next(d for d in self.resultado["detalle_coeficientes"] if d["nombre"] == "x")
        self.assertIsNotNone(detalle["p_value"])

    def test_intervalo_de_confianza(self):
        detalle = next(d for d in self.resultado["detalle_coeficientes"] if d["nombre"] == "x")
        if detalle["intervalo_confianza"] is not None:
            self.assertLessEqual(detalle["intervalo_confianza"]["inferior"], detalle["intervalo_confianza"]["superior"])

    def test_significancia(self):
        detalle = next(d for d in self.resultado["detalle_coeficientes"] if d["nombre"] == "x")
        self.assertIn("significativo_005", detalle)


class DiagnosticoARIMAXTests(SimpleTestCase):
    def setUp(self):
        self.y, self.x = _serie_con_exogena_relevante(n=40)
        self.resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": self.y, "variables_exogenas_historicas": {"x": self.x},
            "variables_exogenas_futuras": {"x": [21.0]},
            "p": 1, "d": 0, "q": 1,
        })

    def test_model_df_es_p_mas_q(self):
        ljung_box = self.resultado["ljung_box"]
        if ljung_box["ejecutado"]:
            self.assertEqual(ljung_box["model_df"], 1 + 1)

    def test_ljung_box_presente(self):
        self.assertIn("ljung_box", self.resultado)

    def test_residuos_finitos(self):
        self.assertGreater(self.resultado["diagnostico_residuos"]["cantidad_residuos"], 0)

    def test_pronostico_finito(self):
        for valor in self.resultado["pronostico"]:
            self.assertTrue(np.isfinite(valor))

    def test_intervalos_finitos(self):
        for intervalo in self.resultado["intervalos_pronostico"]:
            self.assertTrue(np.isfinite(intervalo["limite_inferior"]))
            self.assertTrue(np.isfinite(intervalo["limite_superior"]))

    def test_advertencias_es_lista(self):
        self.assertIsInstance(self.resultado["advertencias"], list)


class ExplicacionYCausalidadTests(SimpleTestCase):
    def test_explicacion_modelo_presente(self):
        y, x = _serie_con_exogena_relevante(n=30)
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": y, "variables_exogenas_historicas": {"x": x},
            "variables_exogenas_futuras": {"x": [21.0]}, "p": 1, "d": 0, "q": 0,
        })
        self.assertIn("descripcion", resultado["explicacion_modelo"])
        self.assertIn("causalidad", resultado["explicacion_modelo"])

    def test_advertencia_causalidad_presente(self):
        y, x = _serie_con_exogena_relevante(n=30)
        resultado = ejecutar_herramienta("modelo_arimax", {
            "valores": y, "variables_exogenas_historicas": {"x": x},
            "variables_exogenas_futuras": {"x": [21.0]}, "p": 1, "d": 0, "q": 0,
        })
        codigos = [a["codigo"] for a in resultado["advertencias"]]
        self.assertIn("ASOCIACION_NO_IMPLICA_CAUSALIDAD", codigos)


class RegresionARIMAMASARIMATests(SimpleTestCase):
    """Confirma que agregar `modelo_arimax` (y ampliar el motor con `exog`)
    no altera el comportamiento de ARIMA, MA ni SARIMA."""

    def test_modelo_arima_sigue_funcionando(self):
        rng = np.random.default_rng(4)
        valores = [100.0]
        for t in range(1, 24):
            valores.append(valores[-1] + 1.5 + rng.normal(scale=4))
        resultado = ejecutar_herramienta("modelo_arima", {
            "valores": valores, "p": 0, "d": 1, "q": 0, "pasos_pronostico": 3,
        })
        self.assertNotIn("error", resultado)

    def test_modelo_ma_sigue_funcionando(self):
        rng = np.random.default_rng(7)
        ruido = rng.normal(scale=2.0, size=41)
        serie = [20.0 + ruido[t] + 0.6 * ruido[t - 1] for t in range(1, 41)]
        resultado = ejecutar_herramienta("modelo_ma", {"valores": serie, "q": 1})
        self.assertNotIn("error", resultado)

    def test_modelo_sarima_sigue_funcionando(self):
        rng = np.random.default_rng(21)
        n = 48
        t = np.arange(n)
        serie = (50 + 0.5 * t + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(scale=2.0, size=n)).tolist()
        resultado = ejecutar_herramienta("modelo_sarima", {
            "valores": serie, "p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 12,
        })
        self.assertNotIn("error", resultado)

    def test_las_cuatro_herramientas_conviven_en_el_registro(self):
        for nombre in ("modelo_arima", "modelo_ma", "modelo_sarima", "modelo_arimax", "acf", "modelo_dickey_fuller"):
            self.assertIn(nombre, TOOL_REGISTRY)
