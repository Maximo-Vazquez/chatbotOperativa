"""Fachada pública AR(p) sobre el núcleo estadístico compartido.

Conserva el contrato histórico de ``modelo_ar`` y configura el motor como
ARIMA(p,0,0), evitando una segunda implementación de ajuste, intervalos,
advertencias y Ljung-Box.
"""

from apps.herramientas.forecasting import diagnostics, engine, validation
from apps.herramientas.forecasting.exceptions import ForecastingError, InvalidOrderError
from apps.herramientas.forecasting.serialization import serializar_parametro


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "modelo_ar",
        "description": (
            "Ajusta un modelo autorregresivo AR(p), internamente ARIMA(p,0,0), "
            "y devuelve coeficientes, diagnóstico residual, pronóstico e intervalos."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "valores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Serie temporal estacionaria.",
                },
                "p": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Orden autorregresivo.",
                },
                "pasos_pronostico": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Cantidad de pasos a pronosticar (default 1).",
                },
                "nivel_confianza": {
                    "type": "number",
                    "minimum": 0.8,
                    "maximum": 0.999,
                    "description": "Nivel de confianza de los intervalos (default 0.95).",
                },
            },
            "required": ["valores", "p"],
        },
    },
}

TOOL_META = {
    "label": "Modelo AR",
    "description": "Ajuste de un modelo Autorregresivo AR(p)",
    "icon": "bx-trending-up",
    "color": "#0ea5e9",
}


def _ejecutar_modelo_ar(
    valores: list,
    p: int,
    pasos_pronostico: int = 1,
    nivel_confianza: float = 0.95,
) -> dict:
    """Valida y ajusta AR(p) mediante el motor común.

    Los errores de dominio se devuelven con ``codigo_error``. El último
    ``except Exception`` es exclusivamente la frontera pública y nunca
    expone trazas ni mensajes crudos de statsmodels.
    """
    try:
        validation.validar_orden_arima(p, 0, 0)
        if p < 1:
            raise InvalidOrderError("'p' debe ser al menos 1 para un modelo AR.")

        serie = validation.validar_serie(valores)
        validation.validar_serie_no_constante(serie)
        validation.validar_muestra_minima(serie.size, p, 0, 0)
        validation.validar_horizonte_pronostico(pasos_pronostico)
        nivel_confianza = validation.validar_nivel_confianza(nivel_confianza)

        resultado = engine.ajustar_arima(
            serie=serie,
            p=p,
            d=0,
            q=0,
            con_constante=True,
            pasos_pronostico=pasos_pronostico,
            nivel_confianza=nivel_confianza,
        )
        diagnostico = diagnostics.construir_diagnostico_residuos(
            resultado.residuos,
            d=0,
            p=p,
            q=0,
        )
    except ForecastingError as exc:
        return {"error": str(exc), "codigo_error": exc.codigo_error}
    except Exception:
        return {
            "error": f"No fue posible ajustar el modelo AR({p}) con los datos proporcionados.",
            "codigo_error": "ERROR_INESPERADO",
        }

    coeficientes = {
        parametro.nombre: parametro.coeficiente
        for parametro in resultado.parametros
    }
    pronostico = [paso.pronostico for paso in resultado.pronosticos]
    intervalos = [
        {
            "paso": paso.paso,
            "pronostico": paso.pronostico,
            "limite_inferior": paso.limite_inferior,
            "limite_superior": paso.limite_superior,
        }
        for paso in resultado.pronosticos
    ]

    return {
        # Claves históricas preservadas.
        "modelo": f"AR({p})",
        "orden_p": p,
        "n_observaciones": resultado.n_observaciones,
        "coeficientes": coeficientes,
        "aic": resultado.aic,
        "bic": resultado.bic,
        "mse_residuos": diagnostico["mse"],
        "media_residuos": diagnostico["media"],
        "varianza_residuos": diagnostico["varianza"],
        "ljung_box": diagnostico["ljung_box"],
        "pasos_pronostico": len(pronostico),
        "pronostico": pronostico,
        # Campos comunes aditivos.
        "orden": {"p": p, "d": 0, "q": 0},
        "representacion_interna": f"ARIMA({p},0,0)",
        "detalle_coeficientes": [
            serializar_parametro(parametro)
            for parametro in resultado.parametros
        ],
        "mse_residuos_entrenamiento": diagnostico["mse"],
        "diagnostico_residuos": diagnostico,
        "intervalos_pronostico": intervalos,
        "nivel_confianza": resultado.nivel_confianza,
        "tendencia_statsmodels": resultado.trend,
        "descripcion_tendencia": resultado.descripcion_tendencia,
        "informacion_ajuste": {
            "convergio": resultado.convergio,
            "metodo": "maxima verosimilitud",
            "n_observaciones_efectivas": resultado.n_observaciones_efectivas,
            "log_likelihood": resultado.log_likelihood,
            "enforce_stationarity": resultado.enforce_stationarity,
            "enforce_invertibility": resultado.enforce_invertibility,
        },
        "advertencias": list(resultado.advertencias) + list(diagnostico["advertencias"]),
    }


TOOL_FUNCTION = _ejecutar_modelo_ar
