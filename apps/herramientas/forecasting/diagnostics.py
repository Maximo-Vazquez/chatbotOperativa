"""Diagnostico estadistico posterior al ajuste: residuos, Ljung-Box, clasificacion
de parametros y de advertencias emitidas por el optimizador de statsmodels.

Este modulo no ajusta modelos ni valida entradas; solo interpreta resultados
ya obtenidos (residuos, warnings capturados) y produce estructuras listas
para serializar a JSON.
"""

import warnings as warnings_module
from typing import Optional

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox


def calcular_estadisticos_residuos(residuos: np.ndarray) -> dict:
    """Media, varianza y MSE de un arreglo de residuos (siempre in-sample)."""
    cantidad = int(residuos.size)
    if cantidad == 0:
        return {"cantidad_residuos": 0, "media": 0.0, "varianza": 0.0, "mse": 0.0}
    return {
        "cantidad_residuos": cantidad,
        "media": float(np.mean(residuos)),
        "varianza": float(np.var(residuos, ddof=1)) if cantidad > 1 else 0.0,
        "mse": float(np.mean(residuos**2)),
    }


def seleccionar_lag_ljung_box(
    cantidad_residuos: int,
    model_df: int,
    lag_maximo: int = 10,
    periodo_estacional: Optional[int] = None,
) -> Optional[int]:
    """Elige un rezago valido para Ljung-Box, o ``None`` si no existe ninguno posible.

    Debe cumplirse simultaneamente: lag >= 1, lag > model_df (grados de
    libertad de la prueba positivos) y lag < cantidad_residuos.

    Si se provee ``periodo_estacional`` (``s`` de un modelo SARIMA), se
    prefiere ese rezago cuando es valido y no consume mas de la mitad de los
    residuos disponibles (para no sacrificar grados de libertad de la
    prueba); si no es viable, se recurre al criterio no estacional habitual.
    """
    if cantidad_residuos < 2:
        return None

    if periodo_estacional and periodo_estacional > 1:
        limite_proporcional = cantidad_residuos // 2
        if model_df < periodo_estacional < cantidad_residuos and periodo_estacional <= limite_proporcional:
            return periodo_estacional

    candidato = min(lag_maximo, max(1, cantidad_residuos // 5))
    candidato = max(candidato, model_df + 1)
    candidato = min(candidato, cantidad_residuos - 1)

    if candidato < 1 or candidato <= model_df or candidato >= cantidad_residuos:
        return None
    return candidato


def ejecutar_ljung_box(
    residuos: np.ndarray, model_df: int, periodo_estacional: Optional[int] = None
) -> dict:
    """Ejecuta Ljung-Box sobre los residuos con los grados de libertad del modelo.

    ``model_df`` debe ser la cantidad de parametros AR+MA (+ estacionales,
    para SARIMA) estimados, no incluye `d`/`D`: son ordenes de
    diferenciacion, no parametros estimados que consuman grados de libertad
    en la autocorrelacion residual. ``periodo_estacional`` (opcional) permite
    preferir un rezago relacionado con el ciclo estacional cuando es valido.
    """
    cantidad = int(residuos.size)
    lag = seleccionar_lag_ljung_box(cantidad, model_df, periodo_estacional=periodo_estacional)
    incluye_rezago_estacional = bool(periodo_estacional) and lag == periodo_estacional

    if lag is None:
        return {
            "ejecutado": False,
            "motivo": (
                "No existen suficientes residuos para seleccionar un rezago mayor "
                "que los grados de libertad del modelo."
            ),
            "lags": None,
            "model_df": model_df,
            "grados_libertad_prueba": None,
            "estadistico": None,
            "p_value": None,
            "autocorrelacion_significativa": None,
            "es_ruido_blanco": None,
            "incluye_rezago_estacional": False,
            "interpretacion": (
                "No se pudo evaluar la autocorrelacion residual con esta muestra."
            ),
        }

    resultado = acorr_ljungbox(residuos, lags=[lag], model_df=model_df, return_df=True)
    estadistico = float(resultado["lb_stat"].iloc[0])
    p_value = float(resultado["lb_pvalue"].iloc[0])
    autocorrelacion_significativa = bool(p_value < 0.05)

    return {
        "ejecutado": True,
        "lags": lag,
        "model_df": model_df,
        "grados_libertad_prueba": lag - model_df,
        "estadistico": round(estadistico, 6),
        "p_value": round(p_value, 6),
        "autocorrelacion_significativa": autocorrelacion_significativa,
        "incluye_rezago_estacional": incluye_rezago_estacional,
        # Alias de compatibilidad con el contrato anterior (frontend/consumidores
        # que leen `es_ruido_blanco`). Se documenta como una lectura parcial:
        # que no se detecte autocorrelacion no certifica que el modelo sea
        # valido en todos los demas aspectos (ver `interpretacion`).
        "es_ruido_blanco": not autocorrelacion_significativa,
        "interpretacion": (
            "Se detecto autocorrelacion residual significativa en el rezago evaluado: "
            "el modelo no captura toda la estructura de la serie."
            if autocorrelacion_significativa
            else "No se detecto autocorrelacion residual significativa en el rezago "
            "evaluado. Esto no certifica por si solo que el modelo sea valido: "
            "deben considerarse ademas la significancia de los parametros y la "
            "convergencia del ajuste."
        ),
    }


def construir_diagnostico_residuos(
    residuos_completos: np.ndarray,
    d: int,
    p: int,
    q: int,
    P: int = 0,
    Q: int = 0,
    descarte_inicial: Optional[int] = None,
    periodo_estacional: Optional[int] = None,
) -> dict:
    """Arma el diagnostico completo de residuos: descarte inicial, estadisticos y Ljung-Box.

    Descarte de residuos iniciales: con `d>0`, ARIMA usa la inicializacion
    difusa exacta de statsmodels para la parte diferenciada, lo cual reduce
    pero no elimina el transitorio de arranque del filtro de Kalman sobre las
    primeras observaciones. Por defecto se descartan los primeros `d`
    residuos (comportamiento original, sin cambios para ARIMA/MA). Para
    SARIMA, el llamador puede pasar `descarte_inicial` explicito (tipicamente
    `d + D*s`) para reflejar el transitorio mas largo que introduce ademas la
    diferenciacion estacional.

    `model_df = p + q + P + Q`: con `P=Q=0` (valor por defecto) es identico
    al calculo usado por ARIMA/MA.
    """
    n_descartar = descarte_inicial if descarte_inicial is not None else d
    if n_descartar > 0 and residuos_completos.size > n_descartar:
        residuos_diagnostico = residuos_completos[n_descartar:]
    else:
        residuos_diagnostico = residuos_completos

    mascara_finitos = np.isfinite(residuos_diagnostico)
    residuos_finitos = bool(np.all(mascara_finitos))
    cantidad_no_finitos = int(np.count_nonzero(~mascara_finitos))
    residuos_para_diagnostico = residuos_diagnostico[mascara_finitos]

    estadisticos = calcular_estadisticos_residuos(residuos_para_diagnostico)
    model_df = p + q + P + Q
    ljung_box = ejecutar_ljung_box(
        residuos_para_diagnostico,
        model_df=model_df,
        periodo_estacional=periodo_estacional,
    )

    advertencias = []
    if cantidad_no_finitos:
        advertencias.append({
            "codigo": "RESIDUOS_NO_FINITOS",
            "mensaje": (
                f"Se excluyeron {cantidad_no_finitos} residuo(s) no finito(s) del "
                "diagnóstico. El ajuste debe interpretarse con cautela."
            ),
            "categoria": "diagnostico",
            "severidad": "advertencia_alta",
        })
    if 0 < estadisticos["cantidad_residuos"] < 8:
        advertencias.append({
            "codigo": "MUESTRA_PEQUENA_DIAGNOSTICO",
            "mensaje": (
                "Quedan pocos residuos disponibles para el diagnostico tras excluir "
                "los primeros valores potencialmente inestables."
            ),
            "categoria": "diagnostico",
            "severidad": "advertencia",
        })

    return {
        "cantidad_residuos": estadisticos["cantidad_residuos"],
        "cantidad_residuos_originales": int(residuos_diagnostico.size),
        "residuos_finitos": residuos_finitos,
        "residuos_no_finitos_excluidos": cantidad_no_finitos,
        "media": round(estadisticos["media"], 6),
        "varianza": round(estadisticos["varianza"], 6),
        "mse": round(estadisticos["mse"], 6),
        "ljung_box": ljung_box,
        "advertencias": advertencias,
    }


def clasificar_parametro(nombre: str) -> str:
    """Clasifica un nombre de parametro de statsmodels en una categoria legible.

    statsmodels nombra los parametros estacionales con el infijo ``.S.``
    (p. ej. ``ar.S.L12``, ``ma.S.L12``): se verifican esos casos antes que
    los prefijos genericos ``ar.``/``ma.`` para no confundir un parametro
    estacional con uno regular.
    """
    nombre_normalizado = nombre.lower()
    if nombre_normalizado.startswith("ar.s."):
        return "autorregresivo_estacional"
    if nombre_normalizado.startswith("ma.s."):
        return "media_movil_estacional"
    if nombre_normalizado.startswith("ar."):
        return "autorregresivo"
    if nombre_normalizado.startswith("ma."):
        return "media_movil"
    if nombre_normalizado in {"const", "intercept"}:
        return "constante"
    if nombre_normalizado in {"x1", "drift", "trend"}:
        return "tendencia"
    if nombre_normalizado == "sigma2":
        return "varianza"
    return "otro"


def interpretar_resultado_adf(resultado_adf: dict) -> dict:
    """Traduce el resultado crudo de la herramienta `modelo_dickey_fuller` a una
    estructura ``estacionariedad`` comun, sin decidir advertencias especificas
    del modelo que la usa (eso lo decide cada herramienta llamadora segun su
    propio criterio pedagogico: MA y SARIMA reaccionan distinto ante una
    serie sin evidencia de estacionariedad).

    No ejecuta el test: solo interpreta el dict ya devuelto por esa
    herramienta, reutilizada tal cual via `ejecutar_herramienta` (no se
    reimplementa ADF aca).
    """
    if "error" in resultado_adf:
        return {
            "prueba": "ADF",
            "ejecutada": False,
            "motivo": resultado_adf["error"],
            "adf_no_ejecutable": True,
        }

    if resultado_adf.get("serie_constante"):
        return {
            "prueba": "ADF",
            "ejecutada": False,
            "motivo": "La serie es constante; no corresponde ejecutar ADF formal.",
        }

    if resultado_adf.get("diagnostico_operativo"):
        return {
            "prueba": "ADF",
            "ejecutada": False,
            "motivo": (
                "Muestra demasiado corta para ADF formal; se aplico un "
                "diagnostico operativo por inspeccion de tendencia."
            ),
            "diagnostico_operativo": True,
            "estacionaria_aproximada": bool(resultado_adf.get("es_estacionaria")),
        }

    evidencia_estacionariedad = bool(resultado_adf["es_estacionaria"])
    return {
        "prueba": "ADF",
        "ejecutada": True,
        "estadistico": resultado_adf["estadistico_adf"],
        "p_value": resultado_adf["p_value"],
        "nivel_significancia": resultado_adf["significancia"],
        "evidencia_estacionariedad": evidencia_estacionariedad,
        "interpretacion": (
            "Existe evidencia de estacionariedad (se rechaza la hipotesis de raiz unitaria)."
            if evidencia_estacionariedad
            else "No existe evidencia suficiente para considerar estacionaria la serie."
        ),
    }


def evaluar_estacionariedad_regular(valores: list) -> tuple[dict, list]:
    """Ejecuta ADF sobre `valores` (reutilizando la herramienta `modelo_dickey_fuller`,
    nunca reimplementada) e interpreta el resultado con `interpretar_resultado_adf`.

    Consolida el patron que antes se repetia identico en las fachadas MA,
    SARIMA y ARIMAX: llamar al ADF y, si no pudo ejecutarse, agregar la
    advertencia comun `ADF_NO_EJECUTABLE`. Cada fachada agrega despues, por
    su cuenta, la advertencia especifica que le corresponda segun el
    resultado (p. ej. recomendar `d>0` en MA, o advertir sobre regresion
    espuria en ARIMAX/SARIMAX): esa parte es deliberadamente distinta entre
    herramientas y no se generaliza aca.
    """
    # Import diferido: `apps.herramientas.tools` construye TOOL_REGISTRY
    # ejecutando los archivos de `tools/` durante su propia carga dinamica;
    # un import a nivel de modulo de `apps.herramientas.tools` crearia un
    # ciclo si esta funcion se llamara durante esa carga. Se ejecuta solo
    # cuando el nucleo realmente evalua estacionariedad (fase de ejecucion).
    from apps.herramientas.tools import ejecutar_herramienta

    resultado_adf = ejecutar_herramienta("modelo_dickey_fuller", {"valores": valores})
    estacionariedad = interpretar_resultado_adf(resultado_adf)

    advertencias = []
    if not estacionariedad.get("ejecutada") and estacionariedad.get("adf_no_ejecutable"):
        advertencias.append({
            "codigo": "ADF_NO_EJECUTABLE",
            "mensaje": "No fue posible ejecutar la prueba de Dickey-Fuller sobre esta serie.",
            "severidad": "advertencia",
        })
    return estacionariedad, advertencias


def clasificar_advertencia(mensaje: str, categoria: str) -> dict:
    """Traduce una advertencia cruda de statsmodels a un codigo e interpretacion estables."""
    texto = mensaje.lower()
    if categoria == "ConvergenceWarning" or "failed to converge" in texto:
        codigo = "CONVERGENCIA_NO_ALCANZADA"
        interpretado = "El optimizador no confirmo convergencia del ajuste."
    elif "non-stationary" in texto:
        codigo = "PARAMETROS_INICIALES_NO_ESTACIONARIOS"
        interpretado = (
            "Los parametros autorregresivos iniciales no eran estacionarios; "
            "el optimizador uso un punto de partida alternativo."
        )
    elif "non-invertible" in texto:
        codigo = "PARAMETROS_INICIALES_NO_INVERTIBLES"
        interpretado = (
            "Los parametros de media movil iniciales no eran invertibles; "
            "el optimizador uso un punto de partida alternativo."
        )
    elif "overflow" in texto or "invalid value" in texto or "divide by zero" in texto:
        codigo = "ADVERTENCIA_NUMERICA"
        interpretado = "Se detecto una condicion numerica inestable durante el ajuste."
    else:
        codigo = "ADVERTENCIA_STATSMODELS"
        interpretado = mensaje.strip()

    return {
        "codigo": codigo,
        "mensaje": interpretado,
        "categoria": categoria,
        "severidad": "advertencia",
    }


def clasificar_advertencias(advertencias_crudas: list) -> list[dict]:
    """Convierte una lista de ``warnings.WarningMessage`` capturados en advertencias
    estructuradas, deduplicadas por codigo (se conserva la primera ocurrencia)."""
    vistas = set()
    resultado = []
    for advertencia in advertencias_crudas:
        if isinstance(advertencia, warnings_module.WarningMessage):
            mensaje = str(advertencia.message)
            categoria = advertencia.category.__name__
        else:
            mensaje = str(advertencia)
            categoria = "Warning"
        estructurada = clasificar_advertencia(mensaje, categoria)
        if estructurada["codigo"] in vistas:
            continue
        vistas.add(estructurada["codigo"])
        resultado.append(estructurada)
    return resultado
