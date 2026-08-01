"""Pruebas de `apps/herramientas/forecasting/temporal.py`: parseo de fechas,
normalizacion/inferencia de frecuencia, deteccion de periodos faltantes y
generacion de fechas futuras de pronostico.
"""

import pandas as pd
from django.test import SimpleTestCase

from apps.herramientas.forecasting import temporal
from apps.herramientas.forecasting.exceptions import (
    DateLengthMismatchError,
    DateValidationError,
    DuplicateDatesError,
    FrequencyValidationError,
    InconsistentFrequencyError,
    UnsortedDatesError,
)


def _fechas_mensuales(n=12, inicio="2024-01-01"):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(inicio, periods=n, freq="MS")]


def _fechas_trimestrales(n=8, inicio="2022-01-01"):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(inicio, periods=n, freq="QS")]


def _fechas_diarias(n=10, inicio="2024-01-01"):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(inicio, periods=n, freq="D")]


class NormalizarFrecuenciaTests(SimpleTestCase):
    def test_alias_pedagogico_mensual(self):
        self.assertEqual(temporal.normalizar_frecuencia("mensual"), "MS")

    def test_alias_pedagogico_es_insensible_a_mayusculas(self):
        self.assertEqual(temporal.normalizar_frecuencia("Mensual"), "MS")

    def test_codigo_pandas_directo(self):
        self.assertEqual(temporal.normalizar_frecuencia("QS"), "QS")

    def test_none_devuelve_none(self):
        self.assertIsNone(temporal.normalizar_frecuencia(None))

    def test_frecuencia_no_reconocida_lanza_error(self):
        with self.assertRaises(FrequencyValidationError):
            temporal.normalizar_frecuencia("bimestral")

    def test_frecuencia_no_string_lanza_error(self):
        with self.assertRaises(FrequencyValidationError):
            temporal.normalizar_frecuencia(12)


class ValidarFechasTests(SimpleTestCase):
    def test_lista_iso_valida(self):
        fechas = _fechas_mensuales(6)
        indice = temporal.validar_fechas(fechas, 6)
        self.assertEqual(len(indice), 6)

    def test_formato_con_fecha_y_hora(self):
        fechas = [f"2024-01-{d:02d}T08:00:00" for d in range(1, 6)]
        indice = temporal.validar_fechas(fechas, 5)
        self.assertEqual(len(indice), 5)

    def test_longitud_distinta_de_valores_lanza_error(self):
        with self.assertRaises(DateLengthMismatchError):
            temporal.validar_fechas(_fechas_mensuales(6), 5)

    def test_fecha_invalida_lanza_error(self):
        fechas = _fechas_mensuales(5)
        fechas[2] = "no-es-una-fecha"
        with self.assertRaises(DateValidationError):
            temporal.validar_fechas(fechas, 5)

    def test_fecha_nula_lanza_error(self):
        fechas = _fechas_mensuales(5)
        fechas[2] = None
        with self.assertRaises(DateValidationError):
            temporal.validar_fechas(fechas, 5)

    def test_fechas_desordenadas_lanzan_error(self):
        fechas = _fechas_mensuales(5)
        fechas[0], fechas[1] = fechas[1], fechas[0]
        with self.assertRaises(UnsortedDatesError):
            temporal.validar_fechas(fechas, 5)

    def test_fechas_duplicadas_lanzan_error(self):
        fechas = _fechas_mensuales(5)
        fechas[3] = fechas[2]
        with self.assertRaises(DuplicateDatesError):
            temporal.validar_fechas(fechas, 5)


class InformacionTemporalTests(SimpleTestCase):
    def test_fechas_mensuales_regulares(self):
        fechas = _fechas_mensuales(12)
        info, advertencias, indice, frecuencia = temporal.construir_informacion_temporal(fechas, None, 12)
        self.assertTrue(info["fechas_proporcionadas"])
        self.assertEqual(info["frecuencia_utilizada"], "MS")
        self.assertTrue(info["serie_regular"])
        self.assertEqual(info["periodos_faltantes"], [])
        self.assertEqual(advertencias, [])

    def test_fechas_trimestrales_regulares(self):
        fechas = _fechas_trimestrales(8)
        info, _adv, _indice, frecuencia = temporal.construir_informacion_temporal(fechas, None, 8)
        # pandas puede inferir un codigo anclado (p. ej. "QS-OCT"): la base
        # de frecuencia debe seguir siendo trimestral.
        self.assertEqual(frecuencia.split("-")[0], "QS")
        self.assertTrue(info["serie_regular"])

    def test_fechas_diarias_regulares(self):
        fechas = _fechas_diarias(15)
        info, _adv, _indice, frecuencia = temporal.construir_informacion_temporal(fechas, None, 15)
        self.assertEqual(frecuencia, "D")
        self.assertTrue(info["serie_regular"])

    def test_periodo_faltante_se_detecta_y_no_se_completa(self):
        fechas = _fechas_mensuales(6)
        del fechas[3]  # elimina 2024-04-01
        info, advertencias, _indice, _frecuencia = temporal.construir_informacion_temporal(fechas, "mensual", 5)
        self.assertFalse(info["serie_regular"])
        self.assertIn("2024-04-01", info["periodos_faltantes"])
        codigos = [a["codigo"] for a in advertencias]
        self.assertIn("PERIODOS_FALTANTES", codigos)

    def test_frecuencia_no_inferible_con_fechas_irregulares(self):
        fechas = ["2024-01-01", "2024-01-03", "2024-02-19"]
        info, advertencias, _indice, frecuencia = temporal.construir_informacion_temporal(fechas, None, 3)
        self.assertIsNone(frecuencia)
        self.assertIsNone(info["frecuencia_utilizada"])
        self.assertFalse(info["serie_regular"])
        codigos = [a["codigo"] for a in advertencias]
        self.assertIn("FRECUENCIA_NO_INFERIBLE", codigos)

    def test_frecuencia_explicita_compatible(self):
        fechas = _fechas_mensuales(10)
        info, _adv, _indice, frecuencia = temporal.construir_informacion_temporal(fechas, "MS", 10)
        self.assertEqual(frecuencia, "MS")
        self.assertEqual(info["frecuencia_solicitada"], "MS")

    def test_frecuencia_explicita_incompatible_lanza_error(self):
        fechas = _fechas_mensuales(10)
        with self.assertRaises(InconsistentFrequencyError):
            temporal.construir_informacion_temporal(fechas, "diaria", 10)

    def test_alias_pedagogico_de_frecuencia(self):
        fechas = _fechas_trimestrales(6)
        info, _adv, _indice, frecuencia = temporal.construir_informacion_temporal(fechas, "trimestral", 6)
        self.assertEqual(frecuencia, "QS")
        self.assertEqual(info["frecuencia_solicitada"], "QS")

    def test_serie_sin_fechas(self):
        info, advertencias, indice, frecuencia = temporal.construir_informacion_temporal(None, None, 10)
        self.assertFalse(info["fechas_proporcionadas"])
        self.assertIsNone(indice)
        self.assertIsNone(frecuencia)
        self.assertEqual(advertencias, [])


class GenerarFechasPronosticoTests(SimpleTestCase):
    def test_pronostico_mensual(self):
        indice = temporal.validar_fechas(_fechas_mensuales(12), 12)
        fechas_futuras = temporal.generar_fechas_pronostico(indice, "MS", 3)
        self.assertEqual(fechas_futuras, ["2025-01-01", "2025-02-01", "2025-03-01"])

    def test_pronostico_trimestral(self):
        indice = temporal.validar_fechas(_fechas_trimestrales(4), 4)
        fechas_futuras = temporal.generar_fechas_pronostico(indice, "QS", 2)
        self.assertEqual(len(fechas_futuras), 2)
        self.assertTrue(fechas_futuras[0] > indice[-1].strftime("%Y-%m-%d"))

    def test_pronostico_diario(self):
        indice = temporal.validar_fechas(_fechas_diarias(5), 5)
        fechas_futuras = temporal.generar_fechas_pronostico(indice, "D", 4)
        self.assertEqual(fechas_futuras, ["2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09"])

    def test_cantidad_correcta_de_fechas(self):
        indice = temporal.validar_fechas(_fechas_mensuales(12), 12)
        fechas_futuras = temporal.generar_fechas_pronostico(indice, "MS", 7)
        self.assertEqual(len(fechas_futuras), 7)

    def test_primera_fecha_posterior_a_la_ultima_observacion(self):
        indice = temporal.validar_fechas(_fechas_mensuales(12), 12)
        fechas_futuras = temporal.generar_fechas_pronostico(indice, "MS", 1)
        self.assertGreater(pd.Timestamp(fechas_futuras[0]), indice[-1])

    def test_frecuencia_desconocida_devuelve_none(self):
        indice = temporal.validar_fechas(_fechas_mensuales(12), 12)
        self.assertIsNone(temporal.generar_fechas_pronostico(indice, None, 3))
        self.assertIsNone(temporal.generar_fechas_pronostico(None, "MS", 3))

    def test_serializacion_iso(self):
        indice = temporal.validar_fechas(_fechas_mensuales(12), 12)
        fechas_futuras = temporal.generar_fechas_pronostico(indice, "MS", 2)
        for fecha in fechas_futuras:
            # No debe lanzar: formato ISO valido (YYYY-MM-DD).
            pd.Timestamp(fecha)
            self.assertRegex(fecha, r"^\d{4}-\d{2}-\d{2}$")
