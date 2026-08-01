"""Herramienta publica `modelo_ma` (function calling del chatbot).

Fachada pedagogica sobre el mismo nucleo compartido que usa `modelo_arima`
(`apps.herramientas.forecasting`). Un MA(q) se representa internamente como
``ARIMA(0, 0, q)`` y se ajusta con el mismo `engine.ajustar_arima`; este
archivo no reimplementa ajuste, metricas, evaluacion temporal, fechas ni
diagnostico de residuos. Solo agrega lo especifico de MA: validacion de `q`,
evaluacion de estacionariedad (reutilizando la herramienta `modelo_dickey_fuller`
existente), informacion de invertibilidad, clasificacion de coeficientes MA y
una explicacion pedagogica de que MA(q) no es un promedio movil de suavizado.
"""

from apps.herramientas.forecasting import diagnostics, engine, evaluation, temporal, validation
from apps.herramientas.forecasting.exceptions import ForecastingError, InsufficientDataError, InvalidMAOrderError
from apps.herramientas.forecasting.schemas import ResultadoAjusteARIMA
from apps.herramientas.forecasting.serialization import serializar_parametro

# Maximo pedagogico de q: coincide con el techo por defecto que ya usan ACF/PACF
# (`min(20, n//4)`) para sugerir ordenes, asi una sugerencia de la ACF nunca
# excede este limite. Ordenes mayores tampoco serian razonables para el uso
# academico de esta herramienta y ya quedarian bloqueados por la regla de
# muestra minima antes de llegar a ser utiles.
Q_MAXIMO = 20

# Minimo tecnico: parametros a estimar (q coeficientes MA + 1 si hay constante)
# mas el mismo margen de 3 observaciones que usa `modelo_arima` para p+d+q+3.
MARGEN_TECNICO = 3

# Minimo recomendado para que el diagnostico de residuos (Ljung-Box, etc.) sea
# razonablemente confiable: una regla practica de "varias veces la cantidad
# de parametros MA", con un piso absoluto.
MULTIPLICADOR_RECOMENDADO = 4
PISO_RECOMENDADO = 15

EXPLICACION_MODELO = {
    "descripcion": (
        "El modelo MA(q) expresa cada valor como una constante mas errores "
        "(innovaciones) actuales y pasados, no valores pasados de la serie: "
        "y_t = mu + e_t + theta_1*e_(t-1) + ... + theta_q*e_(t-q)."
    ),
    "diferencia_promedio_movil": (
        "No es un promedio movil de suavizado: MA(q) es un modelo "
        "probabilistico que usa errores pasados para explicar la serie, no un "
        "promedio de observaciones consecutivas."
    ),
}

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "modelo_ma",
        "description": (
            "Ajusta un modelo de medias moviles MA(q) sobre una serie "
            "preferentemente estacionaria (internamente ARIMA(0,0,q)). Estima "
            "los coeficientes MA con su significancia, evalua residuos y "
            "estacionariedad, genera pronosticos e intervalos, y puede medir "
            "precision fuera de muestra. 'q' puede orientarse con la ACF. No "
            "es un promedio movil de suavizado."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "valores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Serie temporal, idealmente estacionaria.",
                },
                "q": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": Q_MAXIMO,
                    "description": "Orden de medias moviles (sugerido por la ACF). Minimo 1.",
                },
                "pasos_pronostico": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Cantidad de pasos a pronosticar (default 1).",
                },
                "con_constante": {
                    "type": "boolean",
                    "description": "Si True (default), incluye una constante (media del proceso).",
                },
                "nivel_confianza": {
                    "type": "number",
                    "minimum": 0.8,
                    "maximum": 0.999,
                    "description": "Nivel de confianza para los intervalos de prediccion (default 0.95).",
                },
                "fechas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Lista opcional de fechas asociadas a cada valor, misma "
                        "longitud que 'valores' y en orden cronologico estrictamente "
                        "creciente. Formato recomendado: ISO 8601."
                    ),
                },
                "frecuencia": {
                    "type": "string",
                    "description": (
                        "Frecuencia temporal opcional (mensual, trimestral, diaria, "
                        "semanal, anual, horaria, o alias de pandas). Si se omite y "
                        "hay fechas, se intenta inferir automaticamente."
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
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Cantidad de observaciones finales usadas como prueba. "
                        "Prioridad sobre 'porcentaje_prueba'."
                    ),
                },
                "porcentaje_prueba": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 0.5,
                    "description": (
                        "Proporcion final de observaciones usada como prueba "
                        "(estrictamente entre 0 y 0.5) cuando no se da 'cantidad_prueba'."
                    ),
                },
            },
            "required": ["valores", "q"],
        },
    },
}

TOOL_META = {
    "label": "Modelo MA",
    "description": "Ajuste de un modelo de medias moviles MA(q)",
    "icon": "bx-shuffle",
    "color": "#f97316",
}


def _validar_orden_q(q) -> None:
    """Valida `q`: reutiliza el chequeo generico de tipo/signo y agrega la regla propia de MA."""
    validation.validar_orden_arima(0, 0, q)  # tipo entero, no booleano, no negativo (reutilizado)
    if q < 1:
        raise InvalidMAOrderError(
            "El orden q de un modelo MA debe ser un entero mayor o igual que 1."
        )
    if q > Q_MAXIMO:
        raise InvalidMAOrderError(
            f"El orden q={q} supera el maximo admitido por esta herramienta ({Q_MAXIMO})."
        )


def _minimo_tecnico(q: int, con_constante: bool) -> int:
    """Minimo de observaciones para poder ajustar MA(q): parametros + margen tecnico."""
    parametros = q + (1 if con_constante else 0)
    return parametros + MARGEN_TECNICO


def _minimo_recomendado(q: int, con_constante: bool) -> int:
    """Minimo para que el diagnostico de residuos sea razonablemente confiable."""
    return max(_minimo_tecnico(q, con_constante), q * MULTIPLICADOR_RECOMENDADO + PISO_RECOMENDADO)


def _advertencia_no_estacionaria() -> dict:
    return {
        "codigo": "SERIE_NO_ESTACIONARIA_PARA_MA",
        "mensaje": (
            "Un modelo MA puro supone estacionariedad. Considere un modelo "
            "ARIMA con diferenciacion (d>0) si la serie muestra tendencia."
        ),
        "severidad": "advertencia_alta",
    }


def _evaluar_estacionariedad(valores: list) -> tuple[dict, list]:
    """Evalua estacionariedad reutilizando `diagnostics.evaluar_estacionariedad_regular`
    (que a su vez reutiliza la herramienta `modelo_dickey_fuller`; no
    reimplementa ADF). Esa funcion ya resuelve la advertencia comun
    `ADF_NO_EJECUTABLE`; aqui solo se agrega la advertencia especifica de MA.

    No diferencia la serie automaticamente ni convierte MA en ARIMA: solo
    informa y, si corresponde, agrega una advertencia de severidad alta.
    """
    estacionariedad, advertencias = diagnostics.evaluar_estacionariedad_regular(valores)

    if not estacionariedad["ejecutada"]:
        if estacionariedad.get("diagnostico_operativo") and not estacionariedad["estacionaria_aproximada"]:
            advertencias.append(_advertencia_no_estacionaria())
        return estacionariedad, advertencias

    if not estacionariedad["evidencia_estacionariedad"]:
        advertencias.append(_advertencia_no_estacionaria())
    return estacionariedad, advertencias


def _construir_invertibilidad(resultado: ResultadoAjusteARIMA) -> dict:
    """Informa la politica de invertibilidad aplicada por el motor; no analiza raices manualmente."""
    return {
        "forzada_por_statsmodels": resultado.enforce_invertibility,
        "configuracion": f"enforce_invertibility={resultado.enforce_invertibility}",
        "verificacion_manual": False,
        "interpretacion": (
            "El ajuste utiliza la restriccion de invertibilidad de statsmodels: "
            "el optimizador solo acepta soluciones cuyas raices de la parte MA "
            "queden fuera del circulo unitario."
        ),
    }


def _construir_identificacion(q_solicitado: int, valores: list) -> dict:
    """Compara `q` con la sugerencia de la ACF, sin recalcular la ACF (reutiliza la herramienta existente)."""
    from apps.herramientas.tools import ejecutar_herramienta  # import diferido, ver _evaluar_estacionariedad

    resultado_acf = ejecutar_herramienta("acf", {"valores": valores})
    q_sugerido = resultado_acf.get("q_sugerido") if "error" not in resultado_acf else None
    return {
        "q_solicitado": q_solicitado,
        "q_sugerido_acf": q_sugerido,
        "coincide_con_sugerencia": (q_sugerido == q_solicitado) if q_sugerido is not None else None,
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
) -> dict:
    p, d, q = resultado.orden
    diagnostico_residuos = diagnostics.construir_diagnostico_residuos(resultado.residuos, d, p, q)

    coeficientes = {parametro.nombre: parametro.coeficiente for parametro in resultado.parametros}
    coeficientes_ma = {
        parametro.nombre: parametro.coeficiente
        for parametro in resultado.parametros
        if parametro.tipo == "media_movil"
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

    invertibilidad = _construir_invertibilidad(resultado)
    identificacion = _construir_identificacion(q, valores_originales)

    advertencias = (
        list(resultado.advertencias) + list(advertencias_temporales) + list(advertencias_previas)
    )

    return {
        "modelo": f"MA({q})",
        "representacion_interna": f"ARIMA({p},{d},{q})",
        "orden": {"p": p, "d": d, "q": q},
        "orden_q": q,
        "n_observaciones": resultado.n_observaciones,
        "coeficientes": coeficientes,
        "coeficientes_ma": coeficientes_ma,
        "detalle_coeficientes": detalle_coeficientes,
        "estacionariedad": estacionariedad,
        "invertibilidad": invertibilidad,
        "identificacion": identificacion,
        "aic": resultado.aic,
        "bic": resultado.bic,
        # Ver nota equivalente en `modelo_arima.py`: `mse_residuos` (alias de
        # `mse_residuos_entrenamiento`) es error IN-SAMPLE, no predictivo.
        # La precision fuera de muestra esta en `evaluacion.metricas_prueba`.
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
        "explicacion_modelo": EXPLICACION_MODELO,
        "advertencias": advertencias,
    }


def _ejecutar_modelo_ma(
    valores: list,
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
        _validar_orden_q(q)
        serie = validation.validar_serie(valores)

        minimo_tecnico = _minimo_tecnico(q, con_constante)
        if serie.size < minimo_tecnico:
            raise InsufficientDataError(
                f"Se necesitan al menos {minimo_tecnico} observaciones para ajustar "
                f"MA({q}){' con constante' if con_constante else ''}; "
                f"se recibieron {serie.size}."
            )

        validation.validar_serie_no_constante(serie)
        validation.validar_horizonte_pronostico(pasos_pronostico)
        nivel_confianza = validation.validar_nivel_confianza(nivel_confianza)

        advertencias_previas = []
        minimo_recomendado = _minimo_recomendado(q, con_constante)
        if serie.size < minimo_recomendado:
            advertencias_previas.append({
                "codigo": "MUESTRA_MA_REDUCIDA",
                "mensaje": (
                    "El modelo puede ajustarse, pero la muestra es pequena "
                    f"respecto del orden q={q}: se recomiendan al menos "
                    f"{minimo_recomendado} observaciones para un diagnostico confiable."
                ),
                "severidad": "advertencia",
            })

        estacionariedad, advertencias_estacionariedad = _evaluar_estacionariedad(valores)
        advertencias_previas.extend(advertencias_estacionariedad)

        informacion_temporal, advertencias_temporales, indice_fechas, frecuencia_utilizada = (
            temporal.construir_informacion_temporal(fechas, frecuencia, serie.size)
        )

        evaluacion = {"ejecutada": False}
        if evaluar_modelo:
            def _pronosticar_entrenamiento(entrenamiento, pasos):
                resultado_entrenamiento = engine.ajustar_arima(
                    serie=entrenamiento,
                    p=0,
                    d=0,
                    q=q,
                    con_constante=con_constante,
                    pasos_pronostico=pasos,
                    nivel_confianza=nivel_confianza,
                )
                return [paso.pronostico for paso in resultado_entrenamiento.pronosticos]

            evaluacion = evaluation.evaluar_holdout_temporal(
                serie=serie,
                minimo_observaciones_entrenamiento=minimo_tecnico,
                funcion_pronostico=_pronosticar_entrenamiento,
                cantidad_prueba=cantidad_prueba,
                porcentaje_prueba=porcentaje_prueba,
                fechas_indice=indice_fechas,
            )

        # Ajuste final: siempre con la serie completa, igual que en
        # `modelo_arima.py`. La evaluacion de arriba es una operacion
        # historica aparte que nunca produce el pronostico futuro devuelto.
        resultado = engine.ajustar_arima(
            serie=serie,
            p=0,
            d=0,
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
            "error": f"No fue posible ajustar MA({q}) con los datos proporcionados.",
            "codigo_error": "ERROR_INESPERADO",
        }

    return _construir_respuesta(
        resultado, valores, informacion_temporal, advertencias_temporales,
        indice_fechas, frecuencia_utilizada, evaluacion, advertencias_previas, estacionariedad,
    )


TOOL_FUNCTION = _ejecutar_modelo_ma
