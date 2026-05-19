from django.test import TestCase

from apps.herramientas.tools import ejecutar_herramienta


class HerramientasSeriesTests(TestCase):
    def test_acf_serie_constante_no_divide_por_varianza_cero(self):
        resultado = ejecutar_herramienta("acf", {"valores": [150, 150, 150, 150, 150, 150]})

        self.assertTrue(resultado["serie_constante"])
        self.assertEqual(resultado["q_sugerido"], 0)
        self.assertIn("varianza nula", resultado["interpretacion"])

    def test_descomposicion_detecta_volatilidad_extrema(self):
        resultado = ejecutar_herramienta(
            "descomposicion_visualizacion_serie",
            {"valores": [1, 10000, 1, 10000, 1, 10000], "frecuencia": 2},
        )

        estadisticos = resultado["estadisticos"]
        self.assertEqual(estadisticos["minimo"], 1)
        self.assertEqual(estadisticos["maximo"], 10000)
        self.assertEqual(estadisticos["rango"], 9999)
        self.assertTrue(estadisticos["volatilidad_alta"])

    def test_dickey_fuller_muestra_corta_entrega_diagnostico(self):
        resultado = ejecutar_herramienta(
            "modelo_dickey_fuller",
            {"valores": [100, 102, 98, 101, 99, 100]},
        )

        self.assertTrue(resultado["muestra_corta"])
        self.assertIn("es_estacionaria", resultado)
        self.assertIn("decision", resultado)
