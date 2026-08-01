"""Herramienta publica `modelo_sarimax` (function calling del chatbot).

Fachada general sobre el mismo nucleo compartido que usan `modelo_ar`,
`modelo_ma`, `modelo_arima`, `modelo_sarima` y `modelo_arimax`
(`apps.herramientas.forecasting`). SARIMAX es una regresion con errores
temporales estacionales:

    y_t = beta_0 + sum_j(beta_j * x_j,t) + n_t,   n_t ~ SARIMA(p,d,q)(P,D,Q)_s

Componentes estacionales (P,D,Q,s) y variables exogenas son ambos
**opcionales e independientes**: segun cuales esten presentes, esta misma
fachada representa AR, MA, ARMA, ARIMA, SARIMA, ARIMAX o SARIMAX (ver
`_clasificar_tipo_modelo`). No es una reescritura de esas herramientas: es
la misma `engine.ajustar_arima(..., seasonal_order=..., exog=...)` que ya
usan, con ambos parametros simultaneamente opcionales (confirmado
empiricamente en esta fase: no hay exclusion mutua entre ellos). Toda la
logica de validacion/diagnostico/evaluacion/fechas se reutiliza de:

* `forecasting/validation.py` (serie, ordenes regulares, horizonte, confianza)
* `forecasting/seasonal.py` (ordenes y periodicidad estacional, ciclos, coherencia)
* `forecasting/exogenous.py` (estructura, alineacion, multicolinealidad, fuga)
* `forecasting/diagnostics.py` (residuos, Ljung-Box, clasificacion de parametros, ADF)
* `forecasting/evaluation.py` (holdout temporal)
* `forecasting/temporal.py` (fechas, frecuencia, fechas futuras)

Este archivo no reimplementa nada de lo anterior: solo combina las piezas,
detecta el tipo de modelo resultante y arma la respuesta.
"""

from typing import Optional

from apps.herramientas.forecasting import diagnostics, engine, evaluation, exogenous, seasonal, temporal, validation
from apps.herramientas.forecasting.exceptions import (
    ExogenousSingularMatrixError,
    ForecastingError,
    InsufficientDataError,
    SeasonalPeriodRequiredError,
)
from apps.herramientas.forecasting.schemas import ResultadoAjusteARIMA
from apps.herramientas.forecasting.serialization import serializar_parametro

EXPLICACION_MODELO = {
    "descripcion": (
        "SARIMAX combina una estructura ARIMA con componentes estacionales "
        "opcionales y variables externas (exogenas) opcionales: "
        "y_t = beta_0 + suma(beta_j * x_j,t) + n_t, con n_t ~ SARIMA(p,d,q)(P,D,Q)_s."
    ),
    "causalidad": (
        "Los coeficientes exogenos muestran asociaciones predictivas condicionadas "
        "al modelo. No demuestran causalidad."
    ),
}

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "modelo_sarimax",
        "description": (
            "Ajusta un modelo SARIMAX(p,d,q)(P,D,Q,s) general: combina componentes "
            "ARIMA regulares con estacionalidad opcional y variables exogenas "
            "opcionales. Segun que se provea, equivale a AR/MA/ARMA/ARIMA (sin "
            "estacionalidad ni exogenas), SARIMA (con estacionalidad), ARIMAX (con "
            "exogenas) o SARIMAX (con ambos). 's' es obligatorio solo si P, D o Q "
            "son mayores que 0; las exogenas futuras son obligatorias solo si se "
            "usaron exogenas historicas. Devuelve coeficientes, diagnostico, "
            "metricas fuera de muestra, pronosticos e intervalos. La asociacion de "
            "una variable exogena con la serie no implica causalidad."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "valores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Serie temporal objetivo Y_t.",
                },
                "p": {"type": "integer", "minimum": 0, "description": "Orden autorregresivo regular."},
                "d": {"type": "integer", "minimum": 0, "maximum": 2, "description": "Diferenciacion regular."},
                "q": {"type": "integer", "minimum": 0, "description": "Orden de medias moviles regular."},
                "P": {"type": "integer", "minimum": 0, "description": "Orden autorregresivo estacional (default 0)."},
                "D": {"type": "integer", "minimum": 0, "maximum": 2, "description": "Diferenciacion estacional (default 0)."},
                "Q": {"type": "integer", "minimum": 0, "description": "Orden de medias moviles estacional (default 0)."},
                "s": {
                    "type": "integer", "minimum": 2,
                    "description": (
                        "Longitud del ciclo estacional. Obligatorio solo si P, D o Q > 0 "
                        "(12 mensual/anual, 4 trimestral/anual, 7 diario/semanal)."
                    ),
                },
                "variables_exogenas_historicas": {
                    "type": "object",
                    "additionalProperties": {"type": "array", "items": {"type": "number"}},
                    "description": (
                        "Diccionario opcional {nombre_variable: [valores...]} con la "
                        "misma longitud que 'valores'. Si se omite, el modelo se ajusta "
                        "sin variables externas."
                    ),
                },
                "variables_exogenas_futuras": {
                    "type": "object",
                    "additionalProperties": {"type": "array", "items": {"type": "number"}},
                    "description": (
                        "Diccionario con las mismas variables que "
                        "'variables_exogenas_historicas', cada una con exactamente "
                        "'pasos_pronostico' valores. Obligatorio solo si se usaron "
                        "exogenas historicas; no se inventan valores futuros."
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
                    "description": "Fechas opcionales de la serie objetivo, misma longitud que 'valores'.",
                },
                "fechas_exogenas_historicas": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Fechas opcionales de las exogenas historicas (alineacion posicional si se omiten).",
                },
                "fechas_exogenas_futuras": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Fechas opcionales de las exogenas futuras; deben coincidir con las de pronostico.",
                },
                "frecuencia": {
                    "type": "string",
                    "description": "Frecuencia temporal opcional (mensual, diaria, alias de pandas, etc.).",
                },
                "evaluar_modelo": {
                    "type": "boolean",
                    "description": (
                        "Si es true, reserva las ultimas observaciones para medir "
                        "precision fuera de muestra antes de reajustar con la serie "
                        "completa (default false)."
                    ),
                },
                "cantidad_prueba": {
                    "type": "integer", "minimum": 1,
                    "description": "Cantidad de observaciones finales usadas como prueba.",
                },
                "porcentaje_prueba": {
                    "type": "number", "minimum": 0, "maximum": 0.5,
                    "description": "Proporcion final usada como prueba (0 < x < 0.5).",
                },
            },
            "required": ["valores", "p", "d", "q"],
        },
    },
}

TOOL_META = {
    "label": "Modelo SARIMAX",
    "description": "Pronostico estacional con variables externas",
    "icon": "bx-git-branch",
    "color": "#f43f5e",
}


def _clasificar_tipo_modelo(p: int, d: int, q: int, es_estacional: bool, tiene_exogenas: bool) -> str:
    """Identifica el nombre pedagogico de la configuracion, independiente de
    que la clase interna sea siempre la misma (`ARIMA`/`SARIMAX` de
    statsmodels): no se usa "SARIMAX" para todo solo porque la clase interna
    se llame asi.
    """
    if tiene_exogenas and es_estacional:
        return "SARIMAX"
    if tiene_exogenas:
        return "ARIMAX"
    if es_estacional:
        return "SARIMA"
    if d > 0:
        return "ARIMA"
    if p > 0 and q > 0:
        return "ARMA"
    if q > 0:
        return "MA"
    if p > 0:
        return "AR"
    return "ARIMA"  # (0,0,0) sin exogenas: caso degenerado, se etiqueta como ARIMA(0,0,0)


def _nombre_modelo(tipo: str, p: int, d: int, q: int, P: int, D: int, Q: int, s) -> str:
    if tipo in ("SARIMA", "SARIMAX"):
        return f"{tipo}({p},{d},{q})({P},{D},{Q},{s})"
    if tipo == "ARMA":
        return f"ARMA({p},{q})"
    if tipo == "MA":
        return f"MA({q})"
    if tipo == "AR":
        return f"AR({p})"
    return f"{tipo}({p},{d},{q})"  # ARIMA / ARIMAX


def _validar_ordenes(p: int, d: int, q: int, P: int, D: int, Q: int) -> None:
    validation.validar_orden_arima(p, d, q)
    seasonal.validar_techo_ordenes_regulares(p, q)
    seasonal.validar_ordenes_estacionales(P, D, Q)


def _construir_estacionariedad(valores: list, es_estacional: bool, D: int, s) -> tuple[dict, list]:
    regular, advertencias = diagnostics.evaluar_estacionariedad_regular(valores)

    if not es_estacional:
        return {"regular": regular, "estacional": None}, advertencias

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


def _construir_respuesta(
    resultado: ResultadoAjusteARIMA,
    tipo_modelo: str,
    nombres_exogenas: list,
    informacion_temporal: dict,
    advertencias_temporales: list,
    indice_fechas,
    frecuencia_utilizada,
    evaluacion: dict,
    advertencias_previas: list,
    estacionariedad: dict,
    info_ciclos: Optional[dict],
    coherencia_estacional: Optional[dict],
    diagnostico_multicolinealidad: Optional[dict],
    diagnostico_fuga: Optional[dict],
) -> dict:
    p, d, q = resultado.orden
    P, D, Q, s = resultado.orden_estacional if resultado.orden_estacional is not None else (0, 0, 0, 0)
    es_estacional = P > 0 or D > 0 or Q > 0
    tiene_exogenas = bool(nombres_exogenas)
    nombres_exogenas_set = set(nombres_exogenas)

    diagnostico_residuos = diagnostics.construir_diagnostico_residuos(
        resultado.residuos, d, p, q, P=P, Q=Q,
        descarte_inicial=d + D * s,
        periodo_estacional=s if es_estacional else None,
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
    coeficientes_exogenos = {
        parametro.nombre: parametro.coeficiente
        for parametro in resultado.parametros
        if parametro.nombre in nombres_exogenas_set
    }
    detalle_coeficientes = [
        serializar_parametro(parametro, nombres_exogenas_set)
        for parametro in resultado.parametros
    ]

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

    advertencias = list(resultado.advertencias) + list(advertencias_temporales) + list(advertencias_previas)

    explicacion_causalidad = EXPLICACION_MODELO["causalidad"] if tiene_exogenas else None

    return {
        "modelo": _nombre_modelo(tipo_modelo, p, d, q, P, D, Q, s if es_estacional else None),
        "tipo_modelo_detectado": tipo_modelo,
        "representacion_interna": (
            "statsmodels.tsa.arima.model.ARIMA (subclase de SARIMAX) con "
            f"seasonal_order={'(P,D,Q,s)' if es_estacional else '(0,0,0,0)'} y "
            f"exog={'si' if tiene_exogenas else 'None'}"
        ),
        "orden": {"p": p, "d": d, "q": q},
        "orden_estacional": {"P": P, "D": D, "Q": Q, "s": s if es_estacional else None},
        "variables_exogenas": {
            "utilizadas": tiene_exogenas,
            "nombres": nombres_exogenas,
            "cantidad": len(nombres_exogenas),
        },
        "n_observaciones": resultado.n_observaciones,
        "n_ciclos_aproximados": info_ciclos["n_ciclos_aproximados"] if info_ciclos else None,
        "coeficientes": coeficientes,
        "coeficientes_regulares": coeficientes_regulares,
        "coeficientes_estacionales": coeficientes_estacionales,
        "coeficientes_exogenos": coeficientes_exogenos,
        "detalle_coeficientes": detalle_coeficientes,
        "estacionariedad": estacionariedad,
        "coherencia_estacional": coherencia_estacional,
        "diagnostico_multicolinealidad": diagnostico_multicolinealidad,
        "diagnostico_fuga_informacion": diagnostico_fuga,
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
        "explicacion_modelo": {
            "descripcion": EXPLICACION_MODELO["descripcion"],
            "causalidad": explicacion_causalidad,
        },
        "advertencias": advertencias,
    }


def _ejecutar_modelo_sarimax(
    valores: list,
    p: int,
    d: int,
    q: int,
    P: int = 0,
    D: int = 0,
    Q: int = 0,
    s: Optional[int] = None,
    variables_exogenas_historicas: Optional[dict] = None,
    variables_exogenas_futuras: Optional[dict] = None,
    pasos_pronostico: int = 1,
    con_constante: bool = True,
    nivel_confianza: float = 0.95,
    fechas: list | None = None,
    fechas_exogenas_historicas: list | None = None,
    fechas_exogenas_futuras: list | None = None,
    frecuencia: str | None = None,
    evaluar_modelo: bool = False,
    cantidad_prueba: int | None = None,
    porcentaje_prueba: float | None = None,
) -> dict:
    try:
        _validar_ordenes(p, d, q, P, D, Q)
        es_estacional = P > 0 or D > 0 or Q > 0
        if es_estacional:
            if s is None:
                raise SeasonalPeriodRequiredError(
                    "La periodicidad s es obligatoria cuando existe algun componente estacional."
                )
            seasonal.validar_periodicidad(s)
        s_interno = s if es_estacional else 0

        serie = validation.validar_serie(valores)
        validation.validar_serie_no_constante(serie)
        validation.validar_horizonte_pronostico(pasos_pronostico)
        nivel_confianza = validation.validar_nivel_confianza(nivel_confianza)

        tiene_exogenas = bool(variables_exogenas_historicas)
        nombres_exogenas: list = []
        df_historico = None
        if tiene_exogenas:
            df_historico = exogenous.construir_exogenas_historicas(variables_exogenas_historicas, serie.size)
            nombres_exogenas = list(df_historico.columns)

        minimo_tecnico = validation.calcular_minimo_observaciones_general(
            p, d, q, P=P, D=D, Q=Q, s=s_interno, n_exogenas=len(nombres_exogenas), con_constante=con_constante,
        )
        tipo_modelo = _clasificar_tipo_modelo(p, d, q, es_estacional, tiene_exogenas)
        if serie.size < minimo_tecnico:
            raise InsufficientDataError(
                f"Se necesitan al menos {minimo_tecnico} observaciones para ajustar "
                f"{tipo_modelo}({p},{d},{q}); se recibieron {serie.size}."
            )

        advertencias_previas = []

        info_ciclos = None
        if es_estacional:
            info_ciclos, advertencias_ciclos = seasonal.analizar_ciclos(serie.size, s_interno)
            advertencias_previas.extend(advertencias_ciclos)

        parametros_estimados = p + q + P + Q + len(nombres_exogenas) + (1 if con_constante else 0)
        advertencias_previas.extend(
            validation.advertencia_configuracion_compleja(
                parametros_estimados, serie.size, codigo="MODELO_DEMASIADO_COMPLEJO"
            )
        )

        diagnostico_multicolinealidad = None
        diagnostico_fuga = None
        if tiene_exogenas:
            exogenous.validar_ausencia_de_duplicadas(df_historico)
            for nombre in exogenous.detectar_columnas_constantes(df_historico):
                mensaje = (
                    f"La variable '{nombre}' no cambia en la muestra y aporta poca "
                    "informacion identificable."
                )
                if con_constante:
                    mensaje += " Junto con la constante del modelo, puede duplicar el intercepto."
                advertencias_previas.append({"codigo": "EXOGENA_CONSTANTE", "mensaje": mensaje, "severidad": "advertencia"})

            diagnostico_multicolinealidad, advertencias_multi = exogenous.diagnosticar_multicolinealidad(
                df_historico, con_constante
            )
            if diagnostico_multicolinealidad["clasificacion"] == "matriz_degenerada":
                raise ExogenousSingularMatrixError(
                    "La matriz de variables exogenas (junto con la constante, si aplica) no "
                    "tiene rango completo: existe una combinacion lineal exacta entre columnas "
                    "(o con la constante) que impide identificar los coeficientes de forma unica."
                )
            advertencias_previas.extend(advertencias_multi)

            diagnostico_fuga, advertencias_fuga = exogenous.diagnosticar_fuga_informacion(df_historico, serie)
            advertencias_previas.extend(advertencias_fuga)

            advertencias_previas.append({
                "codigo": "ASOCIACION_NO_IMPLICA_CAUSALIDAD",
                "mensaje": EXPLICACION_MODELO["causalidad"],
                "severidad": "informacion",
            })

        estacionariedad, advertencias_estacionariedad = _construir_estacionariedad(valores, es_estacional, D, s_interno)
        advertencias_previas.extend(advertencias_estacionariedad)

        informacion_temporal, advertencias_temporales, indice_fechas, frecuencia_utilizada = (
            temporal.construir_informacion_temporal(fechas, frecuencia, serie.size)
        )

        if tiene_exogenas:
            alineacion_exogenas, advertencias_alineacion = exogenous.resolver_alineacion_exogenas(
                indice_fechas, fechas_exogenas_historicas
            )
            informacion_temporal["alineacion_exogenas"] = alineacion_exogenas
            advertencias_previas.extend(advertencias_alineacion)

        coherencia_estacional = None
        if es_estacional:
            coherencia_estacional, advertencias_coherencia = seasonal.clasificar_coherencia_estacional(
                frecuencia_utilizada, s_interno
            )
            advertencias_previas.extend(advertencias_coherencia)

        fechas_pronostico = temporal.generar_fechas_pronostico(indice_fechas, frecuencia_utilizada, pasos_pronostico)

        df_futuro = None
        if tiene_exogenas:
            exogenous.validar_fechas_exogenas_futuras(fechas_exogenas_futuras, fechas_pronostico)
            df_futuro = exogenous.construir_exogenas_futuras(variables_exogenas_futuras, nombres_exogenas, pasos_pronostico)

        seasonal_order = (P, D, Q, s_interno) if es_estacional else None

        evaluacion = {"ejecutada": False}
        if evaluar_modelo:
            exog_entrenamiento = None
            exog_prueba = None
            if tiene_exogenas:
                # `determinar_tamano_prueba` es pura y deterministica: se
                # invoca aca (ademas de dentro de `evaluar_holdout_temporal`)
                # solo para conocer de antemano el punto de corte y alinear
                # las exogenas -- no duplica la logica de evaluacion (mismos
                # argumentos, mismo resultado garantizado). Ver informe fase 6/7.
                n_prueba_estimado = evaluation.determinar_tamano_prueba(serie.size, cantidad_prueba, porcentaje_prueba)
                n_entrenamiento_estimado = serie.size - n_prueba_estimado
                exog_entrenamiento = df_historico.iloc[:n_entrenamiento_estimado].to_numpy()
                exog_prueba = df_historico.iloc[n_entrenamiento_estimado:].to_numpy()

            def _pronosticar_entrenamiento(entrenamiento_objetivo, pasos):
                resultado_entrenamiento = engine.ajustar_arima(
                    serie=entrenamiento_objetivo,
                    p=p, d=d, q=q,
                    con_constante=con_constante,
                    pasos_pronostico=pasos,
                    nivel_confianza=nivel_confianza,
                    seasonal_order=seasonal_order,
                    exog=exog_entrenamiento,
                    exog_futuro=exog_prueba,
                )
                return [paso.pronostico for paso in resultado_entrenamiento.pronosticos]

            minimo_entrenamiento = minimo_tecnico
            if es_estacional:
                minimo_entrenamiento = max(minimo_tecnico, 2 * s_interno)

            evaluacion = evaluation.evaluar_holdout_temporal(
                serie=serie,
                minimo_observaciones_entrenamiento=minimo_entrenamiento,
                funcion_pronostico=_pronosticar_entrenamiento,
                cantidad_prueba=cantidad_prueba,
                porcentaje_prueba=porcentaje_prueba,
                fechas_indice=indice_fechas,
            )
            if evaluacion.get("ejecutada"):
                if tiene_exogenas:
                    evaluacion["tipo"] = "condicionada_a_exogenas_observadas"
                    evaluacion.setdefault("advertencias", []).append({
                        "codigo": "EVALUACION_CON_EXOGENAS_OBSERVADAS",
                        "mensaje": (
                            "Las metricas de prueba se calculan suponiendo que las variables "
                            "exogenas del periodo de prueba son conocidas: miden precision "
                            "condicionada a esas exogenas observadas, no necesariamente la "
                            "precision cuando las exogenas futuras tambien deban pronosticarse."
                        ),
                        "severidad": "informacion",
                    })
                if es_estacional and evaluacion["n_prueba"] < s_interno:
                    evaluacion.setdefault("advertencias", []).append({
                        "codigo": "PRUEBA_NO_CUBRE_CICLO_COMPLETO",
                        "mensaje": "El conjunto de prueba no cubre un ciclo estacional completo.",
                        "severidad": "informacion",
                    })

        # Ajuste final: siempre con la serie completa y todas las exogenas
        # historicas, igual que en el resto de las fachadas. La evaluacion de
        # arriba es una operacion historica aparte que nunca produce el
        # pronostico futuro; las exogenas de prueba nunca se mezclan con las futuras.
        resultado = engine.ajustar_arima(
            serie=serie,
            p=p, d=d, q=q,
            con_constante=con_constante,
            pasos_pronostico=pasos_pronostico,
            nivel_confianza=nivel_confianza,
            seasonal_order=seasonal_order,
            exog=df_historico,
            exog_futuro=df_futuro,
        )
    except ForecastingError as exc:
        return {"error": str(exc), "codigo_error": exc.codigo_error}
    except Exception:
        # Frontera de seguridad: cualquier fallo no anticipado se controla
        # aca sin exponer traceback ni detalles internos al usuario final.
        return {
            "error": f"No fue posible ajustar el modelo SARIMAX({p},{d},{q}) con los datos proporcionados.",
            "codigo_error": "ERROR_INESPERADO",
        }

    return _construir_respuesta(
        resultado, tipo_modelo, nombres_exogenas, informacion_temporal, advertencias_temporales,
        indice_fechas, frecuencia_utilizada, evaluacion, advertencias_previas,
        estacionariedad, info_ciclos, coherencia_estacional, diagnostico_multicolinealidad, diagnostico_fuga,
    )


TOOL_FUNCTION = _ejecutar_modelo_sarimax
