"""Herramienta publica `modelo_arima` (function calling del chatbot).

Este archivo es solo la interfaz: valida los argumentos recibidos del LLM,
delega el ajuste estadistico en el nucleo compartido
(``apps.herramientas.forecasting``) y traduce el resultado normalizado al
diccionario JSON que consumen el LLM y el frontend (``home.html``). No
contiene logica de ajuste, diagnostico ni validacion propia: eso vive en
``forecasting/engine.py``, ``forecasting/diagnostics.py`` y
``forecasting/validation.py`` respectivamente, para poder reutilizarse desde
las futuras herramientas MA/SARIMA/ARIMAX/SARIMAX sin duplicar codigo.
"""

from apps.herramientas.forecasting import diagnostics, engine, evaluation, temporal, validation
from apps.herramientas.forecasting.exceptions import ForecastingError
from apps.herramientas.forecasting.schemas import ResultadoAjusteARIMA
from apps.herramientas.forecasting.serialization import serializar_parametro

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "modelo_arima",
        "description": (
            "Ajusta un modelo ARIMA(p, d, q) sobre una serie temporal: "
            "p ordenes autorregresivos (PACF), d diferenciaciones (Test ADF), "
            "q ordenes de medias moviles (ACF). Devuelve coeficientes con su "
            "significancia estadistica, AIC/BIC, intervalos de prediccion, "
            "validacion de residuos como ruido blanco (Ljung-Box) y un "
            "pronostico para los siguientes pasos. Opcionalmente acepta fechas "
            "y frecuencia para generar fechas de pronostico, y puede reservar "
            "las ultimas observaciones para evaluar la precision predictiva "
            "fuera de muestra (MAE/RMSE/MAPE) antes de reajustar con toda la serie."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "valores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Serie temporal original Y_t.",
                },
                "p": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Orden autorregresivo (sugerido por la PACF).",
                },
                "d": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 2,
                    "description": "Grado de integracion (diferenciaciones) segun ADF.",
                },
                "q": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Orden de medias moviles (sugerido por la ACF).",
                },
                "pasos_pronostico": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Cantidad de pasos a pronosticar (default 1).",
                },
                "con_constante": {
                    "type": "boolean",
                    "description": (
                        "Si True, incluye constante/drift cuando d>0 "
                        "(util para ARIMA(0,1,0) con drift)."
                    ),
                },
                "nivel_confianza": {
                    "type": "number",
                    "minimum": 0.8,
                    "maximum": 0.999,
                    "description": (
                        "Nivel de confianza para los intervalos de prediccion "
                        "(default 0.95)."
                    ),
                },
                "fechas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Lista opcional de fechas asociadas a cada valor. Debe "
                        "tener la misma longitud que 'valores' y estar ordenada "
                        "cronologicamente (no se reordena automaticamente). "
                        "Formato recomendado: ISO 8601 (YYYY-MM-DD, YYYY-MM o "
                        "YYYY-MM-DDTHH:MM:SS)."
                    ),
                },
                "frecuencia": {
                    "type": "string",
                    "description": (
                        "Frecuencia temporal opcional, como mensual, trimestral, "
                        "diaria, semanal, anual, horaria, o un alias de pandas "
                        "(D, W, MS, M, QS, Q, YS, Y, H). Si se omite y hay "
                        "fechas, se intenta inferir automaticamente."
                    ),
                },
                "evaluar_modelo": {
                    "type": "boolean",
                    "description": (
                        "Si es true, reserva las ultimas observaciones para "
                        "medir la precision fuera de muestra (MAE/RMSE/MAPE) "
                        "antes de reajustar el modelo con la serie completa "
                        "(default false)."
                    ),
                },
                "cantidad_prueba": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Cantidad de observaciones finales utilizadas como "
                        "prueba. Tiene prioridad sobre 'porcentaje_prueba'."
                    ),
                },
                "porcentaje_prueba": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 0.5,
                    "description": (
                        "Proporcion final de observaciones utilizada como "
                        "prueba cuando no se proporciona 'cantidad_prueba' "
                        "(estrictamente entre 0 y 0.5)."
                    ),
                },
            },
            "required": ["valores", "p", "d", "q"],
        },
    },
}

TOOL_META = {
    "label": "Modelo ARIMA",
    "description": "Ajuste ARIMA(p,d,q) con pronostico, intervalos y validacion de residuos",
    "icon": "bx-pulse",
    "color": "#ec4899",
}


def _construir_respuesta(
    resultado: ResultadoAjusteARIMA,
    informacion_temporal: dict,
    advertencias_temporales: list,
    indice_fechas,
    frecuencia_utilizada,
    evaluacion: dict,
) -> dict:
    p, d, q = resultado.orden
    diagnostico_residuos = diagnostics.construir_diagnostico_residuos(resultado.residuos, d, p, q)

    modelo_desc = resultado.modelo
    if resultado.trend == "c":
        modelo_desc += " con constante"
    elif resultado.trend == "t":
        modelo_desc += " con drift"

    coeficientes = {parametro.nombre: parametro.coeficiente for parametro in resultado.parametros}
    detalle_coeficientes = [serializar_parametro(parametro) for parametro in resultado.parametros]

    pronostico = [paso.pronostico for paso in resultado.pronosticos]
    fechas_pronostico = temporal.generar_fechas_pronostico(
        indice_fechas, frecuencia_utilizada, len(resultado.pronosticos)
    )

    intervalos_pronostico = []
    for indice, paso in enumerate(resultado.pronosticos):
        intervalos_pronostico.append({
            "paso": paso.paso,
            "fecha": fechas_pronostico[indice] if fechas_pronostico else None,
            "pronostico": paso.pronostico,
            "limite_inferior": paso.limite_inferior,
            "limite_superior": paso.limite_superior,
        })

    advertencias = list(resultado.advertencias) + list(advertencias_temporales)

    return {
        "modelo": modelo_desc,
        "orden": {"p": p, "d": d, "q": q},
        "n_observaciones": resultado.n_observaciones,
        "coeficientes": coeficientes,
        "aic": resultado.aic,
        "bic": resultado.bic,
        # `mse_residuos` se conserva por compatibilidad hacia atras: es el
        # mismo valor que `mse_residuos_entrenamiento`, el error cuadratico
        # medio de los residuos IN-SAMPLE. No es una medida de precision
        # predictiva fuera de muestra: para eso esta `evaluacion.metricas_prueba`,
        # calculada exclusivamente sobre observaciones que el modelo no vio
        # durante el ajuste (ver `evaluacion` mas abajo).
        "mse_residuos": diagnostico_residuos["mse"],
        "mse_residuos_entrenamiento": diagnostico_residuos["mse"],
        "media_residuos": diagnostico_residuos["media"],
        "varianza_residuos": diagnostico_residuos["varianza"],
        "ljung_box": diagnostico_residuos["ljung_box"],
        "pasos_pronostico": len(resultado.pronosticos),
        "pronostico": pronostico,
        "detalle_coeficientes": detalle_coeficientes,
        "diagnostico_residuos": diagnostico_residuos,
        "intervalos_pronostico": intervalos_pronostico,
        "nivel_confianza": resultado.nivel_confianza,
        "tendencia_statsmodels": resultado.trend,
        "descripcion_tendencia": resultado.descripcion_tendencia,
        "informacion_ajuste": {
            "convergio": resultado.convergio,
            "metodo": "maxima verosimilitud",
            "n_observaciones_efectivas": resultado.n_observaciones_efectivas,
            "log_likelihood": resultado.log_likelihood,
            "aic": resultado.aic,
            "bic": resultado.bic,
            "enforce_stationarity": resultado.enforce_stationarity,
            "enforce_invertibility": resultado.enforce_invertibility,
        },
        "evaluacion": evaluacion,
        "informacion_temporal": informacion_temporal,
        "fechas_pronostico": fechas_pronostico,
        "advertencias": advertencias,
    }


def _ejecutar_modelo_arima(
    valores: list,
    p: int,
    d: int,
    q: int,
    pasos_pronostico: int = 1,
    con_constante: bool = True,
    nivel_confianza: float = 0.95,
    fechas: list | None = None,
    frecuencia: str | None = None,
    evaluar_modelo: bool = False,
    cantidad_prueba: int | None = None,
    porcentaje_prueba: float | None = None,
) -> dict:
    try:
        validation.validar_orden_arima(p, d, q)
        serie = validation.validar_serie(valores)
        minimo_observaciones = validation.validar_muestra_minima(serie.size, p, d, q)
        validation.validar_serie_no_constante(serie)
        validation.validar_horizonte_pronostico(pasos_pronostico)
        nivel_confianza = validation.validar_nivel_confianza(nivel_confianza)

        informacion_temporal, advertencias_temporales, indice_fechas, frecuencia_utilizada = (
            temporal.construir_informacion_temporal(fechas, frecuencia, serie.size)
        )

        evaluacion = {"ejecutada": False}
        if evaluar_modelo:
            def _pronosticar_entrenamiento(entrenamiento, pasos):
                resultado_entrenamiento = engine.ajustar_arima(
                    serie=entrenamiento,
                    p=p,
                    d=d,
                    q=q,
                    con_constante=con_constante,
                    pasos_pronostico=pasos,
                    nivel_confianza=nivel_confianza,
                )
                return [paso.pronostico for paso in resultado_entrenamiento.pronosticos]

            evaluacion = evaluation.evaluar_holdout_temporal(
                serie=serie,
                minimo_observaciones_entrenamiento=minimo_observaciones,
                funcion_pronostico=_pronosticar_entrenamiento,
                cantidad_prueba=cantidad_prueba,
                porcentaje_prueba=porcentaje_prueba,
                fechas_indice=indice_fechas,
            )

        # Ajuste final: siempre sobre la serie completa. La evaluacion de
        # arriba (si se pidio) es una operacion historica aparte que nunca
        # produce el pronostico futuro devuelto al usuario.
        resultado = engine.ajustar_arima(
            serie=serie,
            p=p,
            d=d,
            q=q,
            con_constante=con_constante,
            pasos_pronostico=pasos_pronostico,
            nivel_confianza=nivel_confianza,
        )
    except ForecastingError as exc:
        return {"error": str(exc), "codigo_error": exc.codigo_error}
    except Exception:
        # Frontera de seguridad: cualquier fallo no anticipado se controla
        # aca sin exponer traceback ni detalles internos al usuario final.
        return {
            "error": f"No fue posible ajustar ARIMA({p},{d},{q}) con los datos proporcionados.",
            "codigo_error": "ERROR_INESPERADO",
        }

    return _construir_respuesta(
        resultado, informacion_temporal, advertencias_temporales,
        indice_fechas, frecuencia_utilizada, evaluacion,
    )


TOOL_FUNCTION = _ejecutar_modelo_arima
