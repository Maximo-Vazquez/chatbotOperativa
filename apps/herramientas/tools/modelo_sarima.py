"""Herramienta publica `modelo_sarima` (function calling del chatbot).

Fachada pedagogica sobre el mismo nucleo compartido que usan `modelo_arima` y
`modelo_ma` (`apps.herramientas.forecasting`). Un SARIMA(p,d,q)(P,D,Q)_s se
representa matematicamente como:

    Phi(B^s) * phi(B) * (1-B)^d * (1-B^s)^D * y_t = c + Theta(B^s) * theta(B) * e_t

y se ajusta con el mismo `engine.ajustar_arima(..., seasonal_order=(P,D,Q,s))`
que usa ARIMA/MA: `statsmodels.tsa.arima.model.ARIMA` es subclase de
`statsmodels.tsa.statespace.sarimax.SARIMAX` y acepta `seasonal_order` de
forma nativa (confirmado empiricamente en fase 5), por lo que no se
instancia `SARIMAX` por separado ni se reimplementa ajuste, pronostico,
intervalos, metricas, evaluacion temporal, fechas o diagnostico: todo eso
sigue viviendo en `forecasting/engine.py`, `forecasting/metrics.py`,
`forecasting/evaluation.py`, `forecasting/temporal.py` y
`forecasting/diagnostics.py` respectivamente. La validacion/analisis de
componentes estacionales (ordenes P/D/Q, periodicidad `s`, ciclos,
coherencia con la frecuencia) vive en `forecasting/seasonal.py` desde la
fase 7, compartida con `modelo_sarimax`; este archivo solo agrega
identificacion estacional (ACF en rezagos s/2s) y clasificacion de
coeficientes regulares/estacionales.
"""

from typing import Optional

from apps.herramientas.forecasting import diagnostics, engine, evaluation, seasonal, temporal, validation
from apps.herramientas.forecasting.exceptions import ForecastingError, InsufficientDataError
from apps.herramientas.forecasting.schemas import ResultadoAjusteARIMA
from apps.herramientas.forecasting.serialization import serializar_parametro

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "modelo_sarima",
        "description": (
            "Ajusta un modelo SARIMA(p,d,q)(P,D,Q,s) para series con patrones "
            "estacionales (internamente ARIMA con seasonal_order, sin variables "
            "exogenas). 's' es la longitud del ciclo estacional (12=mensual/anual, "
            "4=trimestral/anual, 7=diario/semanal), no la frecuencia de la serie. "
            "La serie deberia ser razonablemente regular y cubrir varios ciclos "
            "completos para una estimacion confiable. Devuelve coeficientes "
            "regulares y estacionales, diagnostico residual, metricas fuera de "
            "muestra, pronosticos e intervalos."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "valores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Serie temporal original Y_t.",
                },
                "p": {"type": "integer", "minimum": 0, "description": "Orden autorregresivo regular."},
                "d": {"type": "integer", "minimum": 0, "maximum": 2, "description": "Diferenciacion regular."},
                "q": {"type": "integer", "minimum": 0, "description": "Orden de medias moviles regular."},
                "P": {"type": "integer", "minimum": 0, "description": "Orden autorregresivo estacional."},
                "D": {"type": "integer", "minimum": 0, "maximum": 2, "description": "Diferenciacion estacional."},
                "Q": {"type": "integer", "minimum": 0, "description": "Orden de medias moviles estacional."},
                "s": {
                    "type": "integer",
                    "minimum": 2,
                    "description": (
                        "Longitud del ciclo estacional (no la frecuencia): "
                        "12 mensual con ciclo anual, 4 trimestral con ciclo anual, "
                        "7 diario con ciclo semanal, 24 horario con ciclo diario."
                    ),
                },
                "pasos_pronostico": {
                    "type": "integer", "minimum": 1, "maximum": 50,
                    "description": "Cantidad de pasos a pronosticar (default 1).",
                },
                "con_constante": {
                    "type": "boolean",
                    "description": "Si True (default), incluye un termino determinista compatible con d+D.",
                },
                "nivel_confianza": {
                    "type": "number", "minimum": 0.8, "maximum": 0.999,
                    "description": "Nivel de confianza para los intervalos de prediccion (default 0.95).",
                },
                "fechas": {
                    "type": "array", "items": {"type": "string"},
                    "description": (
                        "Lista opcional de fechas asociadas a cada valor, misma "
                        "longitud que 'valores', en orden cronologico estrictamente "
                        "creciente. Formato recomendado: ISO 8601."
                    ),
                },
                "frecuencia": {
                    "type": "string",
                    "description": (
                        "Frecuencia temporal opcional (mensual, trimestral, diaria, "
                        "semanal, anual, horaria, o alias de pandas). Se infiere de "
                        "las fechas si se omite."
                    ),
                },
                "evaluar_modelo": {
                    "type": "boolean",
                    "description": (
                        "Si es true, reserva las ultimas observaciones para medir "
                        "la precision fuera de muestra (MAE/RMSE/MAPE) antes de "
                        "reajustar con la serie completa (default false)."
                    ),
                },
                "cantidad_prueba": {
                    "type": "integer", "minimum": 1,
                    "description": "Cantidad de observaciones finales usadas como prueba.",
                },
                "porcentaje_prueba": {
                    "type": "number", "minimum": 0, "maximum": 0.5,
                    "description": "Proporcion final de observaciones usada como prueba (0 < x < 0.5).",
                },
            },
            "required": ["valores", "p", "d", "q", "P", "D", "Q", "s"],
        },
    },
}

TOOL_META = {
    "label": "Modelo SARIMA",
    "description": "Pronostico ARIMA con componentes estacionales",
    "icon": "bx-repeat",
    "color": "#06b6d4",
}


# "Configuracion demasiado compleja para la muestra": menos de este numero de
# observaciones (post-diferenciacion) por parametro estimado dispara una
# advertencia no bloqueante.
OBSERVACIONES_POR_PARAMETRO_RECOMENDADAS = 5


def _validar_ordenes(p: int, d: int, q: int, P: int, D: int, Q: int) -> None:
    """Valida p,d,q (reutilizando el chequeo generico de ARIMA) y P,D,Q (de `seasonal.py`)."""
    validation.validar_orden_arima(p, d, q)  # tipo, signo y d<=2 (reutilizado)
    seasonal.validar_techo_ordenes_regulares(p, q)
    seasonal.validar_ordenes_estacionales(P, D, Q)


def _parametros_estimados(p: int, q: int, P: int, Q: int, con_constante: bool) -> int:
    return p + q + P + Q + (1 if con_constante else 0)


def _minimo_tecnico(p: int, d: int, q: int, P: int, D: int, Q: int, s: int, con_constante: bool) -> int:
    return validation.calcular_minimo_observaciones_general(
        p, d, q, P=P, D=D, Q=Q, s=s, con_constante=con_constante
    )


def _construir_estacionariedad(valores: list, D: int, s: int) -> tuple[dict, list]:
    """ADF regular (reutilizado) + disclaimer estacional. El ADF regular no
    determina por si solo la necesidad de diferenciacion estacional (D)."""
    regular, advertencias = diagnostics.evaluar_estacionariedad_regular(valores)

    estacional = {
        "orden_D_solicitado": D,
        "periodicidad": s,
        "advertencia": (
            "El test ADF regular no determina por si solo la necesidad de "
            "diferenciacion estacional (D). Esa decision debe apoyarse en "
            "analisis estacional (ACF/PACF en rezagos multiplos de s), "
            "conocimiento del dominio, comparacion entre modelos y diagnostico de residuos."
        ),
    }
    advertencias.append({
        "codigo": "ADF_NO_DETERMINA_DIFERENCIACION_ESTACIONAL",
        "mensaje": (
            f"El ADF regular no certifica si D={D} es la eleccion correcta; "
            f"evalue la ACF en rezagos multiplos de s={s} o el diagnostico de residuos."
        ),
        "severidad": "advertencia",
    })
    return {"regular": regular, "estacional": estacional}, advertencias


def _construir_identificacion_estacional(valores: list, s: int) -> Optional[dict]:
    """ACF en rezagos s y 2s, solo informativa (reutiliza la herramienta `acf`)."""
    from apps.herramientas.tools import ejecutar_herramienta  # import diferido, ver modelo_ma.py

    resultado_acf = ejecutar_herramienta("acf", {"valores": valores, "lags": 2 * s})
    if "error" in resultado_acf or resultado_acf.get("serie_constante"):
        return None

    coeficientes = resultado_acf.get("acf", [])
    banda = resultado_acf.get("banda_confianza", 0)
    rezagos_evaluados = [r for r in (s, 2 * s) if r < len(coeficientes)]
    observaciones = []
    for rezago in rezagos_evaluados:
        valor = coeficientes[rezago]
        if abs(valor) > banda:
            observaciones.append(f"La ACF presenta senal en el rezago {rezago} (rho={valor}).")
        else:
            observaciones.append(f"La ACF no muestra senal clara en el rezago {rezago}.")

    return {
        "periodicidad": s,
        "rezagos_estacionales_evaluados": rezagos_evaluados,
        "observaciones": observaciones,
    }


def _construir_respuesta(
    resultado: ResultadoAjusteARIMA,
    valores_originales: list,
    informacion_temporal: dict,
    advertencias_temporales: list,
    indice_fechas,
    frecuencia_utilizada,
    evaluacion: dict,
    advertencias_previas: list,
    estacionariedad: dict,
    info_ciclos: dict,
    coherencia_estacional: dict,
    identificacion_estacional: Optional[dict],
) -> dict:
    p, d, q = resultado.orden
    P, D, Q, s = resultado.orden_estacional

    diagnostico_residuos = diagnostics.construir_diagnostico_residuos(
        resultado.residuos, d, p, q, P=P, Q=Q,
        descarte_inicial=d + D * s,
        periodo_estacional=s,
    )

    coeficientes = {parametro.nombre: parametro.coeficiente for parametro in resultado.parametros}
    coeficientes_regulares = {
        parametro.nombre: parametro.coeficiente
        for parametro in resultado.parametros
        if parametro.tipo in ("autorregresivo", "media_movil")
    }
    coeficientes_estacionales = {
        parametro.nombre: parametro.coeficiente
        for parametro in resultado.parametros
        if parametro.tipo in ("autorregresivo_estacional", "media_movil_estacional")
    }
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

    advertencias = (
        list(resultado.advertencias) + list(advertencias_temporales) + list(advertencias_previas)
    )

    return {
        "modelo": f"SARIMA({p},{d},{q})({P},{D},{Q},{s})",
        "representacion_interna": (
            "SARIMAX sin variables exogenas (via statsmodels.tsa.arima.model.ARIMA, "
            "subclase de SARIMAX)"
        ),
        "orden": {"p": p, "d": d, "q": q},
        "orden_estacional": {"P": P, "D": D, "Q": Q, "s": s},
        "diferenciacion": {"regular": d, "estacional": D, "periodicidad": s},
        "n_observaciones": resultado.n_observaciones,
        "n_ciclos_aproximados": info_ciclos["n_ciclos_aproximados"],
        "coeficientes": coeficientes,
        "coeficientes_regulares": coeficientes_regulares,
        "coeficientes_estacionales": coeficientes_estacionales,
        "detalle_coeficientes": detalle_coeficientes,
        "estacionariedad": estacionariedad,
        "coherencia_estacional": coherencia_estacional,
        "identificacion_estacional": identificacion_estacional,
        "aic": resultado.aic,
        "bic": resultado.bic,
        "mse_residuos": diagnostico_residuos["mse"],
        "mse_residuos_entrenamiento": diagnostico_residuos["mse"],
        "media_residuos": diagnostico_residuos["media"],
        "varianza_residuos": diagnostico_residuos["varianza"],
        "ljung_box": diagnostico_residuos["ljung_box"],
        "diagnostico_residuos": diagnostico_residuos,
        "evaluacion": evaluacion,
        "informacion_temporal": informacion_temporal,
        "pasos_pronostico": len(resultado.pronosticos),
        "pronostico": pronostico,
        "fechas_pronostico": fechas_pronostico,
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
        "advertencias": advertencias,
    }


def _ejecutar_modelo_sarima(
    valores: list,
    p: int,
    d: int,
    q: int,
    P: int,
    D: int,
    Q: int,
    s: int,
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
        _validar_ordenes(p, d, q, P, D, Q)
        seasonal.validar_periodicidad(s)
        serie = validation.validar_serie(valores)

        minimo_tecnico = _minimo_tecnico(p, d, q, P, D, Q, s, con_constante)
        if serie.size < minimo_tecnico:
            raise InsufficientDataError(
                f"Se necesitan al menos {minimo_tecnico} observaciones para ajustar "
                f"SARIMA({p},{d},{q})({P},{D},{Q},{s}); se recibieron {serie.size}."
            )

        validation.validar_serie_no_constante(serie)
        validation.validar_horizonte_pronostico(pasos_pronostico)
        nivel_confianza = validation.validar_nivel_confianza(nivel_confianza)

        info_ciclos, advertencias_previas = seasonal.analizar_ciclos(serie.size, s)
        parametros_estimados = _parametros_estimados(p, q, P, Q, con_constante)
        advertencias_previas.extend(
            validation.advertencia_configuracion_compleja(
                parametros_estimados, serie.size,
                codigo="CONFIGURACION_ESTACIONAL_COMPLEJA",
                multiplicador=OBSERVACIONES_POR_PARAMETRO_RECOMENDADAS,
            )
        )

        estacionariedad, advertencias_estacionariedad = _construir_estacionariedad(valores, D, s)
        advertencias_previas.extend(advertencias_estacionariedad)

        informacion_temporal, advertencias_temporales, indice_fechas, frecuencia_utilizada = (
            temporal.construir_informacion_temporal(fechas, frecuencia, serie.size)
        )

        coherencia_estacional, advertencias_coherencia = seasonal.clasificar_coherencia_estacional(
            frecuencia_utilizada, s
        )
        advertencias_previas.extend(advertencias_coherencia)

        identificacion_estacional = _construir_identificacion_estacional(valores, s)

        evaluacion = {"ejecutada": False}
        if evaluar_modelo:
            def _pronosticar_entrenamiento(entrenamiento, pasos):
                resultado_entrenamiento = engine.ajustar_arima(
                    serie=entrenamiento,
                    p=p, d=d, q=q,
                    con_constante=con_constante,
                    pasos_pronostico=pasos,
                    nivel_confianza=nivel_confianza,
                    seasonal_order=(P, D, Q, s),
                )
                return [paso.pronostico for paso in resultado_entrenamiento.pronosticos]

            # El entrenamiento debe conservar al menos el minimo tecnico Y al
            # menos dos ciclos estacionales completos: si la division deja
            # menos, la evaluacion se omite (no bloquea el ajuste final).
            minimo_entrenamiento = max(minimo_tecnico, 2 * s)
            evaluacion = evaluation.evaluar_holdout_temporal(
                serie=serie,
                minimo_observaciones_entrenamiento=minimo_entrenamiento,
                funcion_pronostico=_pronosticar_entrenamiento,
                cantidad_prueba=cantidad_prueba,
                porcentaje_prueba=porcentaje_prueba,
                fechas_indice=indice_fechas,
            )
            if evaluacion.get("ejecutada") and evaluacion["n_prueba"] < s:
                evaluacion.setdefault("advertencias", []).append({
                    "codigo": "PRUEBA_NO_CUBRE_CICLO_COMPLETO",
                    "mensaje": "El conjunto de prueba no cubre un ciclo estacional completo.",
                    "severidad": "informacion",
                })

        # Ajuste final: siempre con la serie completa, igual que en
        # `modelo_arima.py`/`modelo_ma.py`. La evaluacion de arriba es una
        # operacion historica aparte que nunca produce el pronostico futuro.
        resultado = engine.ajustar_arima(
            serie=serie,
            p=p, d=d, q=q,
            con_constante=con_constante,
            pasos_pronostico=pasos_pronostico,
            nivel_confianza=nivel_confianza,
            seasonal_order=(P, D, Q, s),
        )
    except ForecastingError as exc:
        return {"error": str(exc), "codigo_error": exc.codigo_error}
    except Exception:
        # Frontera de seguridad: cualquier fallo no anticipado se controla
        # aca sin exponer traceback ni detalles internos al usuario final.
        return {
            "error": (
                f"No fue posible ajustar SARIMA({p},{d},{q})({P},{D},{Q},{s}) "
                "con los datos proporcionados."
            ),
            "codigo_error": "ERROR_INESPERADO",
        }

    return _construir_respuesta(
        resultado, valores, informacion_temporal, advertencias_temporales,
        indice_fechas, frecuencia_utilizada, evaluacion, advertencias_previas,
        estacionariedad, info_ciclos, coherencia_estacional, identificacion_estacional,
    )


TOOL_FUNCTION = _ejecutar_modelo_sarima
