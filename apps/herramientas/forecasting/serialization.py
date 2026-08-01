"""Serialización común de resultados normalizados del motor de pronóstico.

Las fachadas públicas pueden cambiar el nombre pedagógico del modelo o
reclasificar parámetros cuyos nombres corresponden a variables exógenas,
pero no deben mantener copias de la estructura estadística de cada
coeficiente.
"""

from collections.abc import Collection

from .schemas import ParametroEstimado


def serializar_parametro(
    parametro: ParametroEstimado,
    nombres_exogenas: Collection[str] = (),
) -> dict:
    """Convierte un parámetro estimado a la estructura pública común.

    ``nombres_exogenas`` permite reclasificar como ``exogena`` los nombres
    de columnas preservados por statsmodels. La función no modifica el
    objeto normalizado recibido.
    """
    tipo = "exogena" if parametro.nombre in nombres_exogenas else parametro.tipo
    intervalo = None
    if parametro.intervalo_confianza is not None:
        intervalo = {
            "inferior": parametro.intervalo_confianza[0],
            "superior": parametro.intervalo_confianza[1],
        }
    return {
        "nombre": parametro.nombre,
        "tipo": tipo,
        "coeficiente": parametro.coeficiente,
        "error_estandar": parametro.error_estandar,
        "estadistico": parametro.estadistico,
        "p_value": parametro.p_value,
        "intervalo_confianza": intervalo,
        "significativo_005": parametro.significativo_005,
    }

