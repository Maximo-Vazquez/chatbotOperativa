"""Herramienta publica `modelo_arimax` (function calling del chatbot).

Fachada pedagogica sobre el mismo nucleo compartido que usan `modelo_arima`,
`modelo_ma` y `modelo_sarima` (`apps.herramientas.forecasting`). ARIMAX es
una regresion dinamica:

    y_t = beta_0 + beta_1*x_1t + ... + beta_k*x_kt + n_t,   n_t ~ ARIMA(p,d,q)

Las variables exogenas explican parte de la serie; la estructura ARIMA
modela la dependencia temporal que queda en los errores. Los coeficientes
exogenos son asociacion predictiva dentro del modelo, no efectos causales.

Se ajusta con el mismo `engine.ajustar_arima(..., exog=..., exog_futuro=...)`
que usan ARIMA/MA/SARIMA: `statsmodels.tsa.arima.model.ARIMA` acepta `exog`
de forma nativa, asi que no se reimplementa ajuste, pronostico, intervalos,
metricas, evaluacion temporal, fechas ni diagnostico de residuos. Este
archivo solo agrega lo especifico de variables exogenas: validacion
estructural (`forecasting/exogenous.py`), alineacion temporal,
multicolinealidad, controles basicos de fuga de informacion y clasificacion
de coeficientes exogenos.
"""

from typing import Optional

from apps.herramientas.forecasting import diagnostics, engine, evaluation, exogenous, temporal, validation
from apps.herramientas.forecasting.exceptions import (
    ExogenousSingularMatrixError,
    ForecastingError,
    InsufficientDataError,
)
from apps.herramientas.forecasting.schemas import ResultadoAjusteARIMA
from apps.herramientas.forecasting.serialization import serializar_parametro

MARGEN_TECNICO = 3

EXPLICACION_MODELO = {
    "descripcion": (
        "ARIMAX combina variables externas (exogenas) con una estructura ARIMA "
        "que modela la dependencia temporal que queda en los errores: "
        "y_t = beta_0 + beta_1*x_1t + ... + beta_k*x_kt + n_t, con n_t ~ ARIMA(p,d,q)."
    ),
    "causalidad": (
        "Un coeficiente exogeno significativo indica asociacion predictiva dentro "
        "del modelo. No demuestra causalidad por si solo."
    ),
}

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "modelo_arimax",
        "description": (
            "Ajusta un modelo ARIMAX(p,d,q) usando variables exogenas historicas "
            "y futuras. Las exogenas historicas deben alinearse (por fecha o por "
            "posicion) con la serie objetivo, y se requieren exogenas futuras del "
            "mismo largo que 'pasos_pronostico' para poder pronosticar. Devuelve "
            "coeficientes (incluidos los exogenos, con su significancia), "
            "diagnostico de multicolinealidad y de posible fuga de informacion, "
            "metricas fuera de muestra, pronosticos e intervalos. La asociacion "
            "de una variable exogena con la serie no implica causalidad."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "valores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Serie temporal objetivo Y_t.",
                },
                "variables_exogenas_historicas": {
                    "type": "object",
                    "additionalProperties": {"type": "array", "items": {"type": "number"}},
                    "description": (
                        "Diccionario {nombre_variable: [valores...]} con la misma "
                        "longitud que 'valores', alineado por posicion (o por fecha "
                        "si se usan 'fechas'/'fechas_exogenas_historicas')."
                    ),
                },
                "variables_exogenas_futuras": {
                    "type": "object",
                    "additionalProperties": {"type": "array", "items": {"type": "number"}},
                    "description": (
                        "Diccionario con las mismas variables que "
                        "'variables_exogenas_historicas', cada una con exactamente "
                        "'pasos_pronostico' valores futuros conocidos o proyectados por "
                        "el usuario. Obligatorio para poder pronosticar: no se inventan "
                        "valores futuros."
                    ),
                },
                "p": {"type": "integer", "minimum": 0, "description": "Orden autorregresivo."},
                "d": {"type": "integer", "minimum": 0, "maximum": 2, "description": "Diferenciacion regular."},
                "q": {"type": "integer", "minimum": 0, "description": "Orden de medias moviles."},
                "pasos_pronostico": {
                    "type": "integer", "minimum": 1, "maximum": 50,
                    "description": "Cantidad de pasos a pronosticar (default 1).",
                },
                "con_constante": {
                    "type": "boolean",
                    "description": "Si True (default), incluye una constante/drift compatible con d.",
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
                    "description": (
                        "Fechas opcionales de las exogenas historicas. Si se omiten y hay "
                        "'fechas', se asume alineacion por posicion."
                    ),
                },
                "fechas_exogenas_futuras": {
                    "type": "array", "items": {"type": "string"},
                    "description": (
                        "Fechas opcionales de las exogenas futuras; deben coincidir con "
                        "las fechas de pronostico generadas a partir de la frecuencia."
                    ),
                },
                "frecuencia": {
                    "type": "string",
                    "description": "Frecuencia temporal opcional (mensual, diaria, alias de pandas, etc.).",
                },
                "evaluar_modelo": {
                    "type": "boolean",
                    "description": (
                        "Si es true, reserva las ultimas observaciones (objetivo y "
                        "exogenas) para medir precision fuera de muestra usando las "
                        "exogenas reales de ese tramo (default false)."
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
            "required": ["valores", "variables_exogenas_historicas", "p", "d", "q"],
        },
    },
}

TOOL_META = {
    "label": "Modelo ARIMAX",
    "description": "ARIMA con variables exogenas",
    "icon": "bx-git-merge",
    "color": "#8b5cf6",
}


def _construir_estacionariedad(valores: list) -> tuple[dict, list]:
    """ADF sobre el objetivo, reutilizando `diagnostics.evaluar_estacionariedad_regular`
    (que ya resuelve la advertencia comun `ADF_NO_EJECUTABLE`). Aqui solo se
    agrega la advertencia especifica de ARIMAX: la presencia de exogenas no
    elimina la necesidad de considerar la estructura temporal de la serie, y
    una regresion dinamica sobre una serie no estacionaria puede ser espuria.
    """
    estacionariedad, advertencias = diagnostics.evaluar_estacionariedad_regular(valores)

    if estacionariedad.get("ejecutada") and not estacionariedad.get("evidencia_estacionariedad"):
        advertencias.append({
            "codigo": "POSIBLE_REGRESION_ESPURIA",
            "mensaje": (
                "No hay evidencia de estacionariedad en la serie objetivo. Una "
                "regresion dinamica sobre series con tendencia puede producir "
                "relaciones espurias (coeficientes exogenos significativos sin una "
                "relacion real); interprete los resultados con cautela y considere "
                "aumentar 'd' si la tendencia es fuerte."
            ),
            "severidad": "advertencia",
        })
    return estacionariedad, advertencias


def _construir_respuesta(
    resultado: ResultadoAjusteARIMA,
    nombres_exogenas: list[str],
    informacion_temporal: dict,
    advertencias_temporales: list,
    indice_fechas,
    frecuencia_utilizada,
    evaluacion: dict,
    advertencias_previas: list,
    estacionariedad: dict,
    diagnostico_multicolinealidad: dict,
    diagnostico_fuga: dict,
) -> dict:
    p, d, q = resultado.orden
    diagnostico_residuos = diagnostics.construir_diagnostico_residuos(resultado.residuos, d, p, q)

    nombres_exogenas_set = set(nombres_exogenas)
    coeficientes = {parametro.nombre: parametro.coeficiente for parametro in resultado.parametros}
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

    return {
        "modelo": f"ARIMAX({p},{d},{q})",
        "representacion_interna": "ARIMA(p,d,q) con variables exogenas (statsmodels.tsa.arima.model.ARIMA, exog=...)",
        "orden": {"p": p, "d": d, "q": q},
        "variables_exogenas": {"nombres": nombres_exogenas, "cantidad": len(nombres_exogenas)},
        "n_observaciones": resultado.n_observaciones,
        "coeficientes": coeficientes,
        "coeficientes_exogenos": coeficientes_exogenos,
        "detalle_coeficientes": detalle_coeficientes,
        "estacionariedad": estacionariedad,
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
        "explicacion_modelo": EXPLICACION_MODELO,
        "advertencias": advertencias,
    }


def _ejecutar_modelo_arimax(
    valores: list,
    variables_exogenas_historicas: dict,
    p: int,
    d: int,
    q: int,
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
        validation.validar_orden_arima(p, d, q)
        serie = validation.validar_serie(valores)
        validation.validar_serie_no_constante(serie)
        validation.validar_horizonte_pronostico(pasos_pronostico)
        nivel_confianza = validation.validar_nivel_confianza(nivel_confianza)

        df_historico = exogenous.construir_exogenas_historicas(variables_exogenas_historicas, serie.size)
        nombres_exogenas = list(df_historico.columns)

        minimo_tecnico = validation.calcular_minimo_observaciones_general(
            p,
            d,
            q,
            n_exogenas=len(nombres_exogenas),
            con_constante=con_constante,
            margen=MARGEN_TECNICO,
        )
        if serie.size < minimo_tecnico:
            raise InsufficientDataError(
                f"Se necesitan al menos {minimo_tecnico} observaciones para ajustar "
                f"ARIMAX({p},{d},{q}) con {len(nombres_exogenas)} variable(s) exogena(s); "
                f"se recibieron {serie.size}."
            )

        exogenous.validar_ausencia_de_duplicadas(df_historico)

        advertencias_previas = []
        for nombre in exogenous.detectar_columnas_constantes(df_historico):
            mensaje = (
                f"La variable '{nombre}' no cambia en la muestra y aporta poca "
                "informacion identificable."
            )
            if con_constante:
                mensaje += " Junto con la constante del modelo, puede duplicar el intercepto."
            advertencias_previas.append({
                "codigo": "EXOGENA_CONSTANTE", "mensaje": mensaje, "severidad": "advertencia",
            })

        diagnostico_multicolinealidad, advertencias_multicolinealidad = exogenous.diagnosticar_multicolinealidad(
            df_historico, con_constante
        )
        if diagnostico_multicolinealidad["clasificacion"] == "matriz_degenerada":
            raise ExogenousSingularMatrixError(
                "La matriz de variables exogenas (junto con la constante, si aplica) no "
                "tiene rango completo: existe una combinacion lineal exacta entre columnas "
                "(o con la constante) que impide identificar los coeficientes de forma unica."
            )
        advertencias_previas.extend(advertencias_multicolinealidad)

        diagnostico_fuga, advertencias_fuga = exogenous.diagnosticar_fuga_informacion(df_historico, serie)
        advertencias_previas.extend(advertencias_fuga)

        advertencias_previas.append({
            "codigo": "ASOCIACION_NO_IMPLICA_CAUSALIDAD",
            "mensaje": EXPLICACION_MODELO["causalidad"],
            "severidad": "informacion",
        })

        estacionariedad, advertencias_estacionariedad = _construir_estacionariedad(valores)
        advertencias_previas.extend(advertencias_estacionariedad)

        informacion_temporal, advertencias_temporales, indice_fechas, frecuencia_utilizada = (
            temporal.construir_informacion_temporal(fechas, frecuencia, serie.size)
        )
        alineacion_exogenas, advertencias_alineacion = exogenous.resolver_alineacion_exogenas(
            indice_fechas, fechas_exogenas_historicas
        )
        informacion_temporal["alineacion_exogenas"] = alineacion_exogenas
        advertencias_previas.extend(advertencias_alineacion)

        fechas_pronostico = temporal.generar_fechas_pronostico(indice_fechas, frecuencia_utilizada, pasos_pronostico)
        exogenous.validar_fechas_exogenas_futuras(fechas_exogenas_futuras, fechas_pronostico)

        # Las exogenas futuras son obligatorias siempre que se vaya a
        # pronosticar (pasos_pronostico >= 1, que ya es el minimo admitido).
        df_futuro = exogenous.construir_exogenas_futuras(
            variables_exogenas_futuras, nombres_exogenas, pasos_pronostico
        )

        evaluacion = {"ejecutada": False}
        if evaluar_modelo:
            # `determinar_tamano_prueba` es una funcion pura y deterministica:
            # se invoca aca (ademas de dentro de `evaluar_holdout_temporal`)
            # solo para conocer de antemano el punto de corte y poder alinear
            # las exogenas -- no duplica la logica de evaluacion, la reutiliza
            # dos veces con los mismos argumentos (mismo resultado garantizado).
            n_prueba_estimado = evaluation.determinar_tamano_prueba(
                serie.size, cantidad_prueba, porcentaje_prueba
            )
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
                    exog=exog_entrenamiento,
                    exog_futuro=exog_prueba,
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
            if evaluacion.get("ejecutada"):
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

        # Ajuste final: siempre con toda la serie y todas las exogenas
        # historicas, igual que en `modelo_arima.py`/`modelo_ma.py`/
        # `modelo_sarima.py`. La evaluacion de arriba es una operacion
        # historica aparte que nunca produce el pronostico futuro; las
        # exogenas de prueba nunca se mezclan con las exogenas futuras.
        resultado = engine.ajustar_arima(
            serie=serie,
            p=p, d=d, q=q,
            con_constante=con_constante,
            pasos_pronostico=pasos_pronostico,
            nivel_confianza=nivel_confianza,
            exog=df_historico,
            exog_futuro=df_futuro,
        )
    except ForecastingError as exc:
        return {"error": str(exc), "codigo_error": exc.codigo_error}
    except Exception:
        # Frontera de seguridad: cualquier fallo no anticipado se controla
        # aca sin exponer traceback ni detalles internos al usuario final.
        return {
            "error": f"No fue posible ajustar ARIMAX({p},{d},{q}) con los datos proporcionados.",
            "codigo_error": "ERROR_INESPERADO",
        }

    return _construir_respuesta(
        resultado, nombres_exogenas, informacion_temporal, advertencias_temporales,
        indice_fechas, frecuencia_utilizada, evaluacion, advertencias_previas,
        estacionariedad, diagnostico_multicolinealidad, diagnostico_fuga,
    )


TOOL_FUNCTION = _ejecutar_modelo_arimax
