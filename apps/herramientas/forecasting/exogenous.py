"""Variables exogenas: validacion estructural, construccion de la tabla
comun (``pandas.DataFrame``), diagnostico de multicolinealidad y controles
basicos de fuga de informacion.

No depende de ARIMA/SARIMA en si: solo de estructuras tabulares (diccionario
de columnas -> DataFrame) y de la serie objetivo para los controles de fuga.
Pensado para reutilizarse sin cambios desde ARIMAX (esta fase) y, en el
futuro, SARIMAX.
"""

from typing import Optional

import numpy as np
import pandas as pd

from . import temporal
from .exceptions import (
    ExogenousColumnMismatchError,
    ExogenousDateMismatchError,
    ExogenousDuplicateError,
    ExogenousLengthMismatchError,
    ExogenousNonFiniteError,
    ExogenousRequiredError,
    ExogenousValueError,
    ForecastHorizonExogenousMismatchError,
    FutureExogenousRequiredError,
)

# ── Umbrales documentados (ver informe de fase 6 para la justificacion) ────
UMBRAL_CORRELACION_ALTA = 0.95
UMBRAL_NUMERO_CONDICION_MODERADO = 30
UMBRAL_NUMERO_CONDICION_ALTO = 100
UMBRAL_RAZON_ESCALAS_EXTREMA = 100.0
UMBRAL_CORRELACION_FUGA_IDENTICA = 1e-9  # tolerancia para "practicamente identica"
UMBRAL_CORRELACION_CASI_PERFECTA_OBJETIVO = 0.98

NOMBRES_SOSPECHOSOS_FUGA = {
    "objetivo_futuro", "venta_futura", "demanda_futura", "target", "y_futuro",
}


def _validar_columna_numerica(valores, nombre: str) -> np.ndarray:
    """Valida y convierte una columna exogena a ``ndarray[float]``."""
    if not isinstance(valores, (list, tuple)):
        raise ExogenousValueError(f"La variable exogena '{nombre}' debe ser una lista de numeros.")
    if len(valores) == 0:
        raise ExogenousValueError(f"La variable exogena '{nombre}' no puede estar vacia.")

    numeros = []
    for indice, valor in enumerate(valores):
        if isinstance(valor, bool):
            raise ExogenousValueError(
                f"El valor en la posicion {indice} de '{nombre}' es booleano; "
                "se esperaba un numero."
            )
        if isinstance(valor, (list, tuple, dict, set)):
            raise ExogenousValueError(
                f"La variable exogena '{nombre}' contiene una estructura anidada "
                f"invalida en la posicion {indice}."
            )
        if not isinstance(valor, (int, float)):
            raise ExogenousValueError(
                f"El valor en la posicion {indice} de '{nombre}' no es numerico: {valor!r}."
            )
        numeros.append(float(valor))

    arreglo = np.asarray(numeros, dtype=float)
    if np.isnan(arreglo).any():
        raise ExogenousNonFiniteError(f"La variable exogena '{nombre}' contiene valores NaN.")
    if np.isinf(arreglo).any():
        raise ExogenousNonFiniteError(f"La variable exogena '{nombre}' contiene valores infinitos.")
    return arreglo


def _validar_nombres(variables_exogenas: dict) -> list[str]:
    nombres = list(variables_exogenas.keys())
    for nombre in nombres:
        if not isinstance(nombre, str) or not nombre.strip():
            raise ExogenousValueError(
                "Los nombres de las variables exogenas deben ser cadenas de texto no vacias."
            )
    if len(set(nombres)) != len(nombres):
        raise ExogenousValueError("Los nombres de las variables exogenas deben ser unicos.")
    return nombres


def construir_exogenas_historicas(variables_exogenas_historicas, n_valores: int) -> pd.DataFrame:
    """Valida `variables_exogenas_historicas` (dict de columnas) y lo convierte
    a un DataFrame con las columnas en el orden de insercion recibido.
    """
    if not isinstance(variables_exogenas_historicas, dict):
        raise ExogenousValueError("'variables_exogenas_historicas' debe ser un diccionario de columnas.")
    if len(variables_exogenas_historicas) == 0:
        raise ExogenousRequiredError(
            "'variables_exogenas_historicas' debe contener al menos una variable."
        )

    nombres = _validar_nombres(variables_exogenas_historicas)
    columnas = {nombre: _validar_columna_numerica(variables_exogenas_historicas[nombre], nombre) for nombre in nombres}

    longitudes = {nombre: len(valores) for nombre, valores in columnas.items()}
    if len(set(longitudes.values())) > 1:
        detalle = ", ".join(f"'{n}'={l}" for n, l in longitudes.items())
        raise ExogenousLengthMismatchError(
            f"Las variables exogenas historicas no tienen todas la misma longitud ({detalle})."
        )

    longitud = next(iter(longitudes.values()))
    if longitud != n_valores:
        primer_nombre = nombres[0]
        raise ExogenousLengthMismatchError(
            f"La variable exogena '{primer_nombre}' tiene {longitud} observaciones, "
            f"pero la serie objetivo tiene {n_valores}."
        )

    return pd.DataFrame(columnas, columns=nombres)


def construir_exogenas_futuras(
    variables_exogenas_futuras,
    nombres_historicos: list[str],
    pasos_pronostico: int,
) -> pd.DataFrame:
    """Valida `variables_exogenas_futuras` contra las columnas historicas y
    devuelve un DataFrame con las columnas reordenadas para coincidir
    exactamente con el orden historico.
    """
    if not variables_exogenas_futuras:
        raise FutureExogenousRequiredError(
            "Para pronosticar con ARIMAX se necesitan valores futuros de todas "
            "las variables exogenas utilizadas en el ajuste."
        )
    if not isinstance(variables_exogenas_futuras, dict):
        raise ExogenousValueError("'variables_exogenas_futuras' debe ser un diccionario de columnas.")

    nombres_futuros = set(variables_exogenas_futuras.keys())
    nombres_esperados = set(nombres_historicos)
    if nombres_futuros != nombres_esperados:
        faltantes = nombres_esperados - nombres_futuros
        adicionales = nombres_futuros - nombres_esperados
        detalle = []
        if faltantes:
            detalle.append(f"faltan: {', '.join(sorted(faltantes))}")
        if adicionales:
            detalle.append(f"sobran: {', '.join(sorted(adicionales))}")
        raise ExogenousColumnMismatchError(
            "Las variables exogenas futuras deben coincidir exactamente con las "
            f"utilizadas en el ajuste ({'; '.join(detalle)})."
        )

    columnas = {
        nombre: _validar_columna_numerica(variables_exogenas_futuras[nombre], nombre)
        for nombre in nombres_historicos
    }

    longitudes = {nombre: len(valores) for nombre, valores in columnas.items()}
    if len(set(longitudes.values())) > 1:
        detalle = ", ".join(f"'{n}'={l}" for n, l in longitudes.items())
        raise ExogenousLengthMismatchError(
            f"Las variables exogenas futuras no tienen todas la misma longitud ({detalle})."
        )

    longitud = next(iter(longitudes.values()))
    if longitud != pasos_pronostico:
        raise ForecastHorizonExogenousMismatchError(
            f"Cada variable exogena futura debe tener exactamente {pasos_pronostico} "
            f"valores (uno por paso de pronostico); se recibieron {longitud}."
        )

    # Reordena para coincidir exactamente con el orden historico (nunca se
    # inventan, extrapolan ni repiten valores: son los mismos que llegaron).
    return pd.DataFrame(columnas, columns=nombres_historicos)


def detectar_columnas_constantes(df: pd.DataFrame, tolerancia: float = 1e-12) -> list[str]:
    """Nombres de columnas con varianza practicamente nula."""
    return [columna for columna in df.columns if np.ptp(df[columna].to_numpy(dtype=float)) <= tolerancia]


def detectar_columnas_duplicadas(df: pd.DataFrame, tolerancia: float = 1e-9) -> list[tuple[str, str]]:
    """Pares de columnas cuyos valores son practicamente identicos entre si."""
    duplicadas = []
    columnas = list(df.columns)
    for i in range(len(columnas)):
        for j in range(i + 1, len(columnas)):
            a = df[columnas[i]].to_numpy(dtype=float)
            b = df[columnas[j]].to_numpy(dtype=float)
            if np.allclose(a, b, atol=tolerancia, rtol=0):
                duplicadas.append((columnas[i], columnas[j]))
    return duplicadas


def validar_ausencia_de_duplicadas(df: pd.DataFrame) -> None:
    duplicadas = detectar_columnas_duplicadas(df)
    if duplicadas:
        var1, var2 = duplicadas[0]
        raise ExogenousDuplicateError(
            f"Las variables '{var1}' y '{var2}' contienen exactamente los mismos valores."
        )


def diagnosticar_multicolinealidad(df: pd.DataFrame, con_constante: bool) -> tuple[dict, list]:
    """Correlacion entre pares, rango de la matriz de diseno y numero de condicion.

    Umbrales documentados: ``|correlacion| >= 0.95`` -> se reporta el par;
    numero de condicion ``> 30`` -> posible multicolinealidad (advertencia);
    ``> 100`` o matriz sin rango completo -> clasificacion "alta"/"matriz_degenerada".
    No es una prueba formal (no reemplaza VIF exacto ni un analisis dedicado),
    pero es estable y no requiere dependencias nuevas.
    """
    advertencias = []
    matriz = df.to_numpy(dtype=float)
    n, k = matriz.shape

    correlaciones_altas = []
    if k >= 2 and n >= 2:
        corr = df.corr().to_numpy()
        for i in range(k):
            for j in range(i + 1, k):
                valor = corr[i, j]
                if np.isfinite(valor) and abs(valor) >= UMBRAL_CORRELACION_ALTA:
                    correlaciones_altas.append({
                        "variable_1": df.columns[i],
                        "variable_2": df.columns[j],
                        "correlacion": round(float(valor), 6),
                    })

    diseno = matriz if not con_constante else np.column_stack([np.ones(n), matriz])
    rango_completo = True
    numero_condicion = None
    matriz_degenerada = False
    if n > 0 and diseno.shape[1] > 0:
        rango = int(np.linalg.matrix_rank(diseno))
        rango_completo = rango == diseno.shape[1]
        try:
            valores_singulares = np.linalg.svd(diseno, compute_uv=False)
            minimo, maximo = float(valores_singulares[-1]), float(valores_singulares[0])
            if minimo <= 1e-10:
                matriz_degenerada = True
            else:
                numero_condicion = round(maximo / minimo, 4)
        except np.linalg.LinAlgError:
            matriz_degenerada = True

    if matriz_degenerada or not rango_completo:
        clasificacion = "matriz_degenerada"
    elif correlaciones_altas or (numero_condicion is not None and numero_condicion > UMBRAL_NUMERO_CONDICION_ALTO):
        clasificacion = "alta"
    elif numero_condicion is not None and numero_condicion > UMBRAL_NUMERO_CONDICION_MODERADO:
        clasificacion = "moderada"
    else:
        clasificacion = "sin_senales_relevantes"

    if correlaciones_altas:
        advertencias.append({
            "codigo": "EXOGENAS_ALTAMENTE_CORRELACIONADAS",
            "mensaje": (
                f"Se detectaron {len(correlaciones_altas)} par(es) de variables con "
                f"|correlacion| >= {UMBRAL_CORRELACION_ALTA}: la estimacion individual "
                "de sus coeficientes puede ser inestable."
            ),
            "severidad": "advertencia",
        })
    if numero_condicion is not None and numero_condicion > UMBRAL_NUMERO_CONDICION_MODERADO:
        advertencias.append({
            "codigo": "NUMERO_CONDICION_ALTO",
            "mensaje": (
                f"El numero de condicion de la matriz de diseno es {numero_condicion}, "
                f"por encima del umbral orientativo ({UMBRAL_NUMERO_CONDICION_MODERADO}): "
                "posible multicolinealidad."
            ),
            "severidad": "advertencia",
        })

    desvios = df.std(ddof=1).to_numpy(dtype=float)
    desvios_validos = desvios[(desvios > 0) & np.isfinite(desvios)]
    if len(desvios_validos) >= 2:
        razon = float(desvios_validos.max() / desvios_validos.min())
        if razon >= UMBRAL_RAZON_ESCALAS_EXTREMA:
            advertencias.append({
                "codigo": "ESCALAS_EXOGENAS_MUY_DIFERENTES",
                "mensaje": (
                    "Las variables exogenas tienen escalas muy distintas entre si "
                    f"(razon de desvios estandar ~{razon:.1f}x); el numero de condicion "
                    "puede estar influido por la escala y no solo por colinealidad real. "
                    "No se reescalan los datos automaticamente."
                ),
                "severidad": "advertencia",
            })

    resultado = {
        "rango_completo": rango_completo,
        "numero_condicion": numero_condicion,
        "correlaciones_altas": correlaciones_altas,
        "clasificacion": clasificacion,
        "advertencias": advertencias,
    }
    return resultado, advertencias


def diagnosticar_fuga_informacion(df: pd.DataFrame, objetivo: np.ndarray) -> tuple[dict, list]:
    """Controles basicos y prudentes de posible fuga de informacion.

    No pretende (ni puede) detectar toda fuga posible: solo señala los casos
    mas obvios (variable identica u observaciones con correlacion casi
    perfecta con el objetivo, nombres sugestivos). Nunca concluye fuga con
    certeza absoluta a partir de la correlacion o el nombre por si solos.
    """
    advertencias = []
    variables_identicas = []
    variables_casi_identicas = []
    nombres_sospechosos = []

    for columna in df.columns:
        valores = df[columna].to_numpy(dtype=float)
        if np.allclose(valores, objetivo, atol=UMBRAL_CORRELACION_FUGA_IDENTICA, rtol=0):
            variables_identicas.append(columna)
        elif np.ptp(valores) > 0 and np.ptp(objetivo) > 0:
            correlacion = float(np.corrcoef(valores, objetivo)[0, 1])
            if np.isfinite(correlacion) and abs(correlacion) >= UMBRAL_CORRELACION_CASI_PERFECTA_OBJETIVO:
                variables_casi_identicas.append({"variable": columna, "correlacion": round(correlacion, 6)})

        if columna.strip().lower() in NOMBRES_SOSPECHOSOS_FUGA:
            nombres_sospechosos.append(columna)

    if variables_identicas:
        advertencias.append({
            "codigo": "POSIBLE_FUGA_INFORMACION",
            "mensaje": (
                f"La(s) variable(s) {', '.join(variables_identicas)} son identicas a la "
                "serie objetivo: podria existir fuga de informacion (verifique que la "
                "variable estuviera realmente disponible antes de cada observacion)."
            ),
            "severidad": "advertencia_alta",
        })
    for entrada in variables_casi_identicas:
        advertencias.append({
            "codigo": "EXOGENA_CORRELACION_CASI_PERFECTA_OBJETIVO",
            "mensaje": (
                f"La variable '{entrada['variable']}' tiene una correlacion casi perfecta "
                f"({entrada['correlacion']}) con la serie objetivo. Esto no confirma fuga de "
                "informacion por si solo, pero amerita verificar su disponibilidad real en "
                "el momento del pronostico."
            ),
            "severidad": "advertencia",
        })
    if nombres_sospechosos:
        advertencias.append({
            "codigo": "POSIBLE_FUGA_INFORMACION",
            "mensaje": (
                f"El nombre de la(s) variable(s) {', '.join(nombres_sospechosos)} sugiere "
                "que podria derivar del futuro o del propio objetivo; verifique su "
                "disponibilidad real en el momento del pronostico. El nombre por si solo "
                "no es motivo de rechazo."
            ),
            "severidad": "advertencia",
        })

    resultado = {
        "variables_identicas_al_objetivo": variables_identicas,
        "variables_correlacion_casi_perfecta_objetivo": variables_casi_identicas,
        "nombres_sospechosos": nombres_sospechosos,
        "nota_disponibilidad_temporal": (
            "Cada variable exogena debe estar disponible en el momento real en que se "
            "genera el pronostico. Estos controles son basicos y no garantizan detectar "
            "toda fuga de informacion posible."
        ),
    }
    return resultado, advertencias


def resolver_alineacion_exogenas(indice_fechas, fechas_exogenas_historicas) -> tuple[dict, list]:
    """Decide y valida como se alinean las exogenas historicas con el objetivo.

    Extraido en fase 7 desde `tools/modelo_arimax.py` (donde vivia como
    logica privada de esa unica fachada) para que `modelo_arimax` y el nuevo
    `modelo_sarimax` lo reutilicen sin duplicar nada.

    Sin `fechas_exogenas_historicas`, se asume alineacion posicional (misma
    fila = misma observacion) y se documenta explicitamente en la salida. Con
    `fechas_exogenas_historicas`, deben coincidir exactamente (mismo orden y
    mismos valores) con las fechas ya validadas de la serie objetivo
    (`indice_fechas`, un `pandas.DatetimeIndex` o `None`).
    """
    if fechas_exogenas_historicas is None:
        advertencias = []
        if indice_fechas is not None:
            advertencias.append({
                "codigo": "ALINEACION_EXOGENAS_POR_POSICION",
                "mensaje": (
                    "La alineacion entre la serie objetivo y las variables exogenas se "
                    "realizo por posicion: no se proporcionaron fechas propias para las exogenas."
                ),
                "severidad": "informacion",
            })
        return {"metodo": "posicional", "fechas_exogenas_proporcionadas": False}, advertencias

    if indice_fechas is None:
        raise ExogenousDateMismatchError(
            "Se proporcionaron 'fechas_exogenas_historicas' sin proporcionar 'fechas' "
            "para la serie objetivo: no es posible verificar la alineacion temporal."
        )

    indice_exogenas = temporal.validar_fechas(fechas_exogenas_historicas, len(fechas_exogenas_historicas))
    if not indice_fechas.equals(indice_exogenas):
        raise ExogenousDateMismatchError(
            "Las fechas de las variables exogenas historicas no coinciden exactamente "
            "(mismo orden y mismos valores) con las fechas de la serie objetivo."
        )
    return {"metodo": "fechas_propias", "fechas_exogenas_proporcionadas": True}, []


def validar_fechas_exogenas_futuras(fechas_exogenas_futuras, fechas_pronostico) -> None:
    """Valida que `fechas_exogenas_futuras` (si se proveen) coincidan exactamente
    con las fechas de pronostico generadas a partir de la frecuencia detectada.

    Extraido en fase 7 desde `tools/modelo_arimax.py` por la misma razon que
    `resolver_alineacion_exogenas`.
    """
    if fechas_exogenas_futuras is None:
        return
    if fechas_pronostico is None:
        raise ExogenousDateMismatchError(
            "Se proporcionaron 'fechas_exogenas_futuras' pero no se pudo determinar una "
            "frecuencia para generar fechas de pronostico con las cuales compararlas."
        )
    if list(fechas_exogenas_futuras) != list(fechas_pronostico):
        raise ExogenousDateMismatchError(
            "'fechas_exogenas_futuras' no coincide con las fechas de pronostico generadas "
            "a partir de la frecuencia detectada."
        )
