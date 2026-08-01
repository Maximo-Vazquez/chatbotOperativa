# Informe del proyecto — Motor de pronóstico (MA/ARIMA/SARIMA/ARIMAX/SARIMAX)

Este archivo se actualiza al finalizar cada fase del trabajo. Cada sección es el
informe completo entregado al cierre de esa fase; no se reescriben fases
anteriores, solo se agregan las nuevas al final.

---

## Fase 1 — Diagnóstico técnico y arquitectura propuesta

**Alcance de la fase:** análisis de solo lectura del repositorio. No se
modificó, creó ni eliminó ningún archivo de código.

### 1. Resumen ejecutivo

El repositorio es un chatbot Django que expone herramientas de series
temporales (ACF, PACF, ADF, estabilización de media/varianza, descomposición,
AR, ARIMA) al modelo de lenguaje (DeepSeek, API compatible con OpenAI) vía
function calling. El mecanismo de registro dinámico (`TOOL_DEFINITION` /
`TOOL_META` / `TOOL_FUNCTION` en `apps/herramientas/tools/*.py`, cargado por
`apps/herramientas/tools.py`) es simple, funciona y **no debe romperse**.

La implementación estadística actual de ARIMA
(`apps/herramientas/tools/modelo_arima.py`) es correcta como MVP académico
pero **acopla estadística, validación y contrato de salida en una única
función**, sin capacidad de evaluación fuera de muestra, sin intervalos de
predicción, sin significancia de parámetros y con un manejo de Ljung-Box y
MSE que puede inducir a conclusiones erróneas si se generaliza sin cuidado
(MSE de entrenamiento presentado sin aclarar que no es error de pronóstico
real).

`statsmodels==0.14.6` (instalado; `requirements.txt` fija `0.14.4`,
compatible) ya soporta todo lo necesario:
`ARIMA(endog, exog=None, order=, seasonal_order=, trend=)` acepta
simultáneamente orden estacional y variables exógenas — **no hace falta
`SARIMAX` como clase separada ni dependencias nuevas**. Esto simplifica mucho
la arquitectura: un solo motor de ajuste puede cubrir MA, AR, ARIMA, SARIMA,
ARIMAX y SARIMAX variando parámetros.

Recomendación de dirección: **no reescribir**, sino extraer un paquete
`apps/herramientas/forecasting/` (motor, validaciones, métricas, diagnóstico,
esquemas, excepciones) del que `modelo_arima.py` pase a ser un adaptador
fino, y sobre el mismo motor construir `modelo_ma.py`, `modelo_sarima.py`,
`modelo_arimax.py`, `modelo_sarimax.py` como nuevos archivos de herramienta
(cada uno con su propio `TOOL_DEFINITION`/`TOOL_META`/`TOOL_FUNCTION`,
respetando el contrato de carga dinámica).

### 2. Arquitectura actual

```
apps/
  chatbot/          # app Django del chat: views, prompts, constants (modelos DeepSeek)
  herramientas/      # app Django de las "tools" de function calling
    tools.py         # loader dinámico + ejecutor + serializador JSON-safe
    tools/           # una herramienta por archivo (glob no recursivo, *.py, sin "_" inicial)
      acf.py
      pacf.py
      modelo_ar.py
      modelo_arima.py
      modelo_dickey_fuller.py
      descomposicion_visualizacion_serie.py
      estabilizacion_media.py
      estabilizacion_varianza.py
    models.py        # ToolCall (JSONField input/output, sin esquema fijo)
    views.py         # list_tools -> TOOL_META (para frontend)
  login/             # auth (allauth + Google)
main/settings.py      # Django settings, DB sqlite3 por defecto, DEEPSEEK_*
```

No hay una app "core"/"forecasting" separada; toda la lógica estadística vive
dentro de `apps/herramientas/tools/*.py`. No existe README.md en el repo
(solo `docs/chatbot/consigna.txt`, muy breve, y `AGENTS.md`/`CLAUDE.md`/
`GEMINI.md`/`QODER.md`, textos idénticos de guía para distintos agentes IA).

### 3. Flujo de Function Calling

1. **Definición de herramientas**: cada archivo en `apps/herramientas/tools/*.py`
   declara `TOOL_DEFINITION` (schema JSON estilo function calling de OpenAI),
   `TOOL_META` (label/icon/color para frontend) y `TOOL_FUNCTION`.
2. **Registro**: `apps/herramientas/tools.py:_cargar_modulos_de_herramientas()`
   hace `glob("*.py")` no recursivo sobre `tools/`, carga cada módulo con
   `importlib.util.spec_from_file_location` + `exec_module`, y arma
   `TOOL_DEFINITIONS`, `TOOL_META`, `TOOL_REGISTRY`.
3. **Selección por el LLM**: `apps/chatbot/views.py:_completion_kwargs()`
   agrega `tools=TOOL_DEFINITIONS, tool_choice="auto"`.
4. **Recepción de argumentos**: loop `while ... choice.finish_reason ==
   "tool_calls"` (máx. `MAX_TOOL_ITERS=9`), `json.loads(tc.function.arguments)`.
5. **Ejecución**: `ejecutar_herramienta(nombre, argumentos)` busca en
   `TOOL_REGISTRY` y llama `fn(**argumentos)`.
6. **Captura de excepciones**: única frontera — atrapa `TypeError` y
   `Exception` genérica, devuelve `{"error": "..."}`.
7. **Serialización JSON**: `_to_json_safe()` convierte tipos numpy/pandas a
   nativos.
8. **Vuelta al LLM**: mensaje `{"role":"tool", "content": json.dumps(result)}`.
9. **Presentación al usuario**: `TOOL_RENDERERS` en `home.html` (JS inline)
   tiene una función de render por herramienta; **si una herramienta nueva no
   tiene renderer dedicado, cae a un fallback genérico con el JSON crudo** —
   agregar MA/SARIMA/ARIMAX/SARIMAX no rompe el frontend.
10. **Dependencia directa del contrato de `modelo_arima`**: el renderer JS
    `modelo_arima(out)` y `ToolCall.output_data` (JSONField sin esquema)
    dependen de las claves exactas de salida.

### 4. Implementación estadística actual

**ARIMA (`modelo_arima.py`):** usa `statsmodels.tsa.arima.model.ARIMA`,
`order=(p,d,q)`, tendencia resuelta manualmente según `d` y `con_constante`
(`d=0→'c'`, `d=1→'t'`, `d≥2→'n'`). Calcula coeficientes, AIC/BIC, descarta los
primeros `d` residuos para diagnóstico, MSE **in-sample**, Ljung-Box con lag
heurístico (`max(1,min(10,len//5))`) **sin `model_df`**, pronóstico solo
puntual vía `.forecast()` (sin intervalos, sin significancia, sin captura de
`ConvergenceWarning`, sin evaluación fuera de muestra). Claves de salida
consumidas por el resto del sistema: `modelo, orden.{p,d,q}, n_observaciones,
coeficientes, aic, bic, mse_residuos, media_residuos, varianza_residuos,
ljung_box.{lags,estadistico,p_value,es_ruido_blanco,interpretacion},
pasos_pronostico, pronostico`.

**ACF/PACF:** bandas `±z/√n`, regla de corte simple para sugerir `q`/`p`
(heurística orientativa, no definitiva), bien manejado el caso de varianza
nula, límites de lags correctos para muestras pequeñas.

**Dickey-Fuller:** valida muestra mínima, maneja series constantes y muestras
cortas (fallback a diagnóstico operativo por pendiente si `adfuller` falla),
no distingue estacionariedad estacional de la regular.

**Descomposición:** exige mínimo 2 ciclos (`n >= frecuencia*2`), valida
positividad para modelo multiplicativo, no maneja NaN explícitamente.

### 5. Problemas encontrados

- **Errores:** ninguno funcional (los 11 tests existentes pasaban).
- **Limitaciones:** sin intervalos, sin significancia, sin evaluación
  fuera de muestra, sin fechas/frecuencia real.
- **Deuda técnica:** duplicación entre `modelo_ar.py` y `modelo_arima.py`;
  sin capa de validación/esquemas reutilizable.
- **Riesgos estadísticos:** MSE in-sample puede leerse como precisión
  predictiva; Ljung-Box sin `model_df` es menos conservador de lo correcto;
  sin captura de convergencia.
- **Riesgos de compatibilidad:** el renderer JS de `home.html` depende de
  nombres de clave exactos; `ToolCall.output_data` es JSONField sin esquema
  (sin riesgo de migración, pero sí de reportes/analytics futuros).

### 6. Arquitectura recomendada

```
apps/herramientas/
├── forecasting/
│   ├── __init__.py
│   ├── engine.py         # construcción/ajuste, forecast, intervalos
│   ├── validation.py     # validaciones de entrada
│   ├── metrics.py        # MAE/RMSE/MAPE
│   ├── diagnostics.py    # residuos, Ljung-Box, convergencia, significancia
│   ├── schemas.py        # entrada/salida común
│   └── exceptions.py     # jerarquía de excepciones propia
└── tools/
    ├── modelo_ar.py / modelo_arima.py        # existentes, migran al núcleo
    ├── modelo_ma.py / modelo_sarima.py       # nuevos
    └── modelo_arimax.py / modelo_sarimax.py  # nuevos
```

`forecasting/` como paquete hermano de `tools/` es seguro porque el loader
dinámico solo recorre `tools/*.py` de forma no recursiva.

### 7-8. Contratos de entrada/salida (resumen)

- **Común:** `valores, fechas?, frecuencia?, pasos_pronostico, nivel_confianza,
  evaluar_modelo?, cantidad_prueba?, porcentaje_prueba?`.
- **ARIMA:** `p,d,q,con_constante` (sin cambios). **MA:** `q,con_constante`.
  **SARIMA:** agrega `P,D,Q,s`. **ARIMAX:** agrega variables exógenas.
  **SARIMAX:** combinación de ambos.
- **Formato de exógenas — decisión:** diccionario por columnas
  (`{"temperatura":[...], "promocion":[...]}`), no matriz + nombres separados:
  más robusto para function calling (el LLM no tiene que sincronizar el orden
  de dos listas).
- **Salida común:** conserva TODAS las claves actuales de `modelo_arima` y
  agrega `orden_estacional, variables_exogenas, detalle_coeficientes,
  diagnostico_residuos, metricas_entrenamiento, metricas_prueba, evaluacion,
  intervalos_pronostico, fechas_pronostico, nivel_confianza,
  informacion_ajuste, advertencias` (lista estructurada
  `{codigo, mensaje, severidad}`).
- **Política de compatibilidad:** clave antigua conservada + clave nueva más
  precisa agregada, deprecación documentada, sin eliminación inmediata.

### 9-14. Excepciones, pruebas, plan de implementación, decisiones y riesgos

- **Excepciones:** jerarquía `ForecastingError` con subtipos por tipo de
  fallo (datos, configuración, muestra, exógenas, convergencia, numérico,
  pronóstico), todas capturables por el `except Exception` genérico ya
  existente en `tools.py`.
- **Pruebas actuales:** `django.test.TestCase`, 11 tests
  (`apps/herramientas/tests.py`, `apps/chatbot/tests.py`). Ejecutado
  `python manage.py test apps.herramientas apps.chatbot`: **11 tests, 0
  fallos, 0 errores, 0 omitidas, OK (11.5s)**.
- **Orden de implementación recomendado:** núcleo compartido → migración de
  ARIMA → métricas → evaluación temporal → fechas/frecuencia → MA → SARIMA →
  ARIMAX → SARIMAX → auditoría y documentación.
- **Decisiones recomendadas:** motor interno = `ARIMA` de statsmodels (no
  `SARIMAX` directo); `mse_residuos` se mantiene como alias; MAPE excluye
  observaciones con valor real ≈0 y reporta cuántas se excluyeron; framework
  de pruebas = `django.test.TestCase` (sin agregar `pytest`).
- **Riesgos pendientes:** no romper el renderer JS de `home.html`; secreto
  hardcodeado en `main/settings.py` (`CLAVE_API_DEEPSEEK_INVITADO`, hallazgo
  colateral de seguridad, fuera de alcance de esta tarea); confirmar que
  `statsmodels==0.14.4` (versión fijada en `requirements.txt`, distinta de la
  `0.14.6` instalada localmente) soporta `seasonal_order`/`exog` en el
  entorno real de despliegue.

### 15. Comandos ejecutados en esta fase

```
python --version                         → Python 3.11.4
pip show statsmodels numpy pandas Django → statsmodels 0.14.6, numpy 2.4.6, pandas 3.0.3, Django 5.2.6
python manage.py test apps.herramientas apps.chatbot -v 2  → Ran 11 tests — OK
inspect.signature(ARIMA.__init__) / SARIMAX.__init__       → confirma seasonal_order + exog soportados por ARIMA
ajuste ARIMA(1,1,1) + get_forecast().conf_int() + pvalues/bse/tvalues/arparams/maparams → todos disponibles
```

### Criterios para iniciar la fase 2 (definidos en esta fase)

- Acuerdo sobre formato de exógenas (diccionario por columnas) antes de
  `schemas.py`.
- Test de regresión fijado sobre `modelo_arima` actual antes de tocarlo.
- `forecasting/` con tests propios pasando en aislamiento.
- `python manage.py test` en 0 fallos tras cada etapa.
- Verificación manual de que el chat sigue renderizando igual tras la
  migración.

---

## Fase 2 — Núcleo estadístico compartido y refactor de `modelo_arima`

**Alcance de la fase:** implementación real de código. No se implementaron
MA, SARIMA, ARIMAX, SARIMAX, evaluación temporal, variables exógenas ni
componentes estacionales (quedan para fases posteriores).

### 1. Diagnóstico previo

Reverifiqué contra el código real (no solo contra el informe de fase 1):

- El loader dinámico y `modelo_arima.py` seguían exactamente como se
  documentó en fase 1.
- `requirements.txt` sigue fijando `statsmodels==0.14.4`; el entorno tenía
  `0.14.6` instalado. Confirmé en vivo que `ARIMA.fit()` expone
  `param_names`, `bse`, `pvalues`, `tvalues`, `conf_int()`,
  `get_forecast().conf_int()` y `mle_retvals['converged']` — todo lo
  necesario sin dependencias nuevas.
- No existe en el repo ninguna serie fija del "caso académico" de 24 meses
  (no hay fixtures ni datos de ejemplo versionados).
- Los 11 tests preexistentes pasaban antes de tocar nada.

### 2. Arquitectura implementada

Paquete nuevo `apps/herramientas/forecasting/`, hermano de `tools/` (el
loader dinámico solo recorre `tools/*.py`, nunca se confunde con una
herramienta):

- **`exceptions.py`** — jerarquía de excepciones de dominio, cada una con
  `codigo_error` propio (`ForecastingError` como base).
- **`schemas.py`** — dataclasses del resultado normalizado
  (`ParametroEstimado`, `PronosticoPaso`, `ResultadoAjusteARIMA`).
- **`validation.py`** — validación de serie, orden, muestra mínima,
  horizonte, nivel de confianza y resolución de tendencia. No conoce
  statsmodels.
- **`diagnostics.py`** — estadísticos de residuos, selección de rezago y
  ejecución de Ljung-Box con `model_df`, clasificación de parámetros y de
  advertencias de statsmodels.
- **`engine.py`** — único punto que instancia y ajusta `ARIMA`, captura
  advertencias, arma parámetros con inferencia estadística y genera
  pronóstico + intervalos vía `get_forecast()`.

`modelo_arima.py` quedó como interfaz fina: valida, delega en
`engine.ajustar_arima()`, arma el JSON de salida a partir de
`diagnostics.construir_diagnostico_residuos()`.

### 3. Archivos creados

```
apps/herramientas/forecasting/__init__.py
apps/herramientas/forecasting/exceptions.py
apps/herramientas/forecasting/schemas.py
apps/herramientas/forecasting/validation.py
apps/herramientas/forecasting/diagnostics.py
apps/herramientas/forecasting/engine.py
apps/herramientas/test_forecasting_core.py
apps/herramientas/test_forecasting_diagnostics.py
apps/herramientas/test_modelo_arima.py
```

### 4. Archivos modificados

- `apps/herramientas/tools/modelo_arima.py` — reescrito completo (134
  inserciones / 79 eliminaciones) para delegar en el núcleo, agregar
  `nivel_confianza` como argumento opcional y ampliar el contrato de salida.
  Único archivo de producción tocado.

No se modificó `tools.py` (loader), `models.py`, ni ningún archivo del
chatbot, frontend, Docker o CI.

### 5. Refactorización de ARIMA

**Antes:** una función de ~90 líneas con ajuste, Ljung-Box, forecast puntual
y armado de JSON todo junto; sin intervalos, sin significancia, sin captura
de advertencias, Ljung-Box con `lags` heurístico sin `model_df`.

**Ahora:** `_ejecutar_modelo_arima()` valida (`validation.*`) → delega en
`engine.ajustar_arima()` (que usa `diagnostics.*` para clasificar
parámetros/advertencias) → arma la respuesta JSON. Mismo modelo de fondo
(`statsmodels.tsa.arima.model.ARIMA`), misma regla de tendencia, pero ahora
con `get_forecast()`, intervalos, `detalle_coeficientes`, Ljung-Box con
`model_df=p+q` y advertencias de convergencia/estacionariedad/invertibilidad
capturadas.

### 6. Compatibilidad

- **Argumentos preservados:** `valores, p, d, q` obligatorios;
  `pasos_pronostico, con_constante` opcionales, mismos defaults. Agregado:
  `nivel_confianza` (default `0.95`), opcional.
- **Claves preservadas:** `modelo, orden, n_observaciones, coeficientes, aic,
  bic, mse_residuos, media_residuos, varianza_residuos, ljung_box,
  pasos_pronostico, pronostico`.
- **Claves nuevas:** `detalle_coeficientes, diagnostico_residuos,
  intervalos_pronostico, nivel_confianza, tendencia_statsmodels,
  descripcion_tendencia, informacion_ajuste, advertencias,
  mse_residuos_entrenamiento`.
- **Diferencia numérica esperable:** ninguna sistemática (mismo modelo de
  fondo); no se fijaron valores exactos en los tests.
- **Deprecación:** `mse_residuos` queda como alias documentado de
  `mse_residuos_entrenamiento`, sin fecha de eliminación en esta fase.
- **Riesgo de frontend detectado y mitigado:** el renderer JS
  `modelo_arima(out)` en `home.html` lee `ljung_box.es_ruido_blanco`. La
  nueva estructura de Ljung-Box (más rica, con `ejecutado`/
  `autocorrelacion_significativa`) **conserva `es_ruido_blanco`** como alias
  booleano explícito para no romper esa vista sin tocar `home.html` (fuera
  de alcance de esta fase).

### 7. Decisiones estadísticas

- **Tendencia:** se conserva `d=0→'c'`, `d=1→'t'`, `d≥2→'n'`,
  `con_constante=False→'n'`; ahora documentada en `descripcion_tendencia`.
- **Residuos iniciales:** se mantiene el descarte de los primeros `d`
  residuos. La inicialización difusa exacta de `ARIMA` mitiga pero no
  elimina el transitorio de arranque del filtro de Kalman; decisión
  documentada en el docstring de `diagnostics.construir_diagnostico_residuos()`.
- **`model_df`:** Ljung-Box usa `model_df = p + q` (no incluye `d`).
- **Selección de lag:** exige `lag > model_df` y `lag < cantidad_residuos`;
  si no existe lag válido, devuelve `{"ejecutado": False, "motivo": "..."}`
  en vez de forzar la prueba con grados de libertad inválidos (verificado:
  `acorr_ljungbox` no lanza excepción en ese caso, devuelve `NaN` —
  silenciosamente inválido si no se controla).
- **Intervalos:** vía `get_forecast(steps=...).conf_int(alpha=1-nivel_confianza)`,
  con validación de finitud y de `límite_inferior <= límite_superior`.
- **Significancia:** `p_value < 0.05` por parámetro, sin invalidar el modelo
  completo por un parámetro no significativo.
- **Estacionariedad/invertibilidad:** se mantienen `enforce_stationarity=True,
  enforce_invertibility=True` (comportamiento previo); ahora expuestos en
  `informacion_ajuste`, sin parámetros públicos nuevos para controlarlos.

### 8. Manejo de errores y advertencias

| Código | Excepción | Disparador |
|---|---|---|
| `SERIE_INVALIDA` | `InvalidSeriesError` | vacía, no numérica, booleana, NaN, Inf, constante |
| `ORDEN_INVALIDA` | `InvalidOrderError` | no entero, negativo, `d>2` |
| `MUESTRA_INSUFICIENTE` | `InsufficientDataError` | `n < p+d+q+3` |
| `HORIZONTE_INVALIDO` | `InvalidForecastHorizonError` | `pasos_pronostico` fuera de `[1,50]` o no entero |
| `NIVEL_CONFIANZA_INVALIDO` | `InvalidConfidenceLevelError` | fuera de `[0.80, 0.999]` |
| `ERROR_AJUSTE` | `ModelFitError` | falla el optimizador |
| `ERROR_NUMERICO` | `NumericalError` | `LinAlgError` (matriz singular) |
| `ERROR_PRONOSTICO_GENERACION` | `ForecastGenerationError` | pronóstico/intervalos no finitos o inconsistentes |
| `ERROR_INESPERADO` | genérico, sin traceback | cualquier fallo no anticipado |

Advertencias capturadas con `warnings.catch_warnings(record=True)` alrededor
de `.fit()`, clasificadas y deduplicadas por código
(`CONVERGENCIA_NO_ALCANZADA`, `PARAMETROS_INICIALES_NO_ESTACIONARIOS`,
`PARAMETROS_INICIALES_NO_INVERTIBLES`, `ADVERTENCIA_NUMERICA`), verificado con
casos reales que las disparan.

### 9. Pruebas agregadas (93 nuevas)

- **`test_forecasting_core.py`** (36 tests): validación de serie (vacía,
  texto, booleanos, NaN, Inf, constante), validación de orden, muestra
  mínima, horizonte, nivel de confianza, resolución de tendencia, jerarquía
  de excepciones, serialización.
- **`test_forecasting_diagnostics.py`** (21 tests): selección de rezago de
  Ljung-Box (incluyendo el caso `None`), ejecución de Ljung-Box, descarte de
  residuos iniciales, clasificación de parámetros, clasificación/
  deduplicación de advertencias reales de statsmodels.
- **`test_modelo_arima.py`** (36 tests): ARIMA(0,1,0) con y sin drift,
  AR(1,0,0), MA(0,0,1), ARIMA(1,0,1), `d=2`, horizontes de 1 y varios pasos,
  intervalos y su consistencia, nivel de confianza configurable, coeficientes
  simples y detallados, p-valores, IC de parámetros, AIC/BIC, diagnóstico
  residual, `model_df`, caso no ejecutable de Ljung-Box, captura de
  advertencias, errores controlados sin traceback, compatibilidad de claves
  antiguas, registro dinámico, serialización JSON completa, regresión con
  serie sintética de 24 observaciones (documentando que no existe la serie
  del caso académico en el repo) y regresión de la firma anterior sin
  `nivel_confianza`.

Series sintéticas con `numpy.random.default_rng(seed)` fijo; comparaciones
con `assertAlmostEqual`/`np.isfinite`/desigualdades, nunca igualdad exacta de
flotantes.

### 10. Comandos ejecutados

```
python manage.py check                              → 2 issues (warnings preexistentes de allauth, no relacionados)
python -m compileall -f apps/herramientas            → todos los .py compilan sin errores de sintaxis
python manage.py test apps.herramientas -v 1         → Ran 96 tests — OK (3 preexistentes + 93 nuevas)
python manage.py test apps.chatbot -v 1              → Ran 8 tests — OK (sin cambios, no tocado)
python manage.py test                                → Ran 109 tests — OK (proyecto completo, incluye apps.login)

smoke test manual (ejecutar_herramienta): ARIMA(0,1,0) con drift sobre 24 obs → JSON completo, json.dumps exitoso
smoke test manual: serie constante/vacía/NaN/booleanos/texto/órdenes inválidos/muestra insuficiente/horizonte/nivel de confianza inválidos → error controlado con codigo_error, sin excepción cruda
smoke test manual: ARIMA(2,0,2) sobre ruido corto (n=15) → advertencias reales capturadas (no estacionario, no invertible, no convergencia), convergio=False propagado
inspección statsmodels 0.14.6 instalado → param_names, bse, pvalues, tvalues, conf_int(), get_forecast().conf_int(), mle_retvals['converged'] disponibles
```

### 11. Resultado completo de las pruebas

```
apps.herramientas + apps.chatbot: 104 tests — 104 exitosas, 0 fallos, 0 errores, 0 omitidas — 4.85s
proyecto completo (incluye apps.login): 109 tests — 109 exitosas, 0 fallos, 0 errores, 0 omitidas — 4.98s
```

### 12. Riesgos pendientes

- `home.html` no fue tocado: la UI sigue mostrando solo lo que ya mostraba
  (no hay renderer nuevo para `detalle_coeficientes`/`intervalos_pronostico`/
  `advertencias`); esos datos ya viajan en el JSON pero solo se ven vía
  "Copiar todo" hasta que se actualice el frontend (fuera de alcance).
- `requirements.txt` sigue fijando `statsmodels==0.14.4` mientras el entorno
  de desarrollo tiene `0.14.6`; conviene validar en el entorno de despliegue
  real antes de la fase 3.
- No hay evaluación fuera de muestra, ni fechas/frecuencia, ni variables
  exógenas, ni componentes estacionales — deliberadamente fuera de esta fase.
- El "caso académico" de 24 meses mencionado en la consigna no existe como
  fixture en el repo; la prueba de regresión equivalente usa una serie
  sintética documentada como tal.

### 13. Preparación para la fase 3

La arquitectura queda lista para incorporar sin duplicar lógica:

- **Métricas** (MAE/RMSE/MAPE): nuevo `forecasting/metrics.py` consumiendo
  `ResultadoAjusteARIMA.pronosticos` y una serie de prueba real, sin tocar
  `engine.py`.
- **Evaluación temporal** (holdout): requiere que `engine.ajustar_arima()`
  acepte un tramo de entrenamiento separado del de prueba — encaja como
  parámetro adicional sin romper la firma actual.
- **Fechas y frecuencia**: `validation.py` puede extenderse con
  `validar_fechas()` sin afectar las validaciones existentes; `schemas.py`
  puede agregar `fechas_pronostico` a `PronosticoPaso` de forma aditiva.

No se implementó nada de esto todavía.

---

## Fase 3 — Métricas, evaluación temporal y fechas/frecuencia opcionales

**Alcance de la fase:** ampliación del núcleo compartido con MAE/RMSE/MAPE,
holdout temporal fuera de muestra, fechas y frecuencia opcionales, y fechas
futuras del pronóstico. No se implementaron MA, SARIMA, ARIMAX ni SARIMAX,
ni componentes estacionales, ni variables exógenas, ni selección automática
de órdenes.

### 1. Diagnóstico previo

Releí `informe.md` (fases 1 y 2) y reverifiqué contra el código real antes de
tocar nada:

- El núcleo de fase 2 (`apps/herramientas/forecasting/{exceptions,schemas,
  validation,diagnostics,engine}.py`) seguía exactamente como se documentó.
- `apps/herramientas/tools/modelo_arima.py` seguía delegando en ese núcleo,
  sin fechas, evaluación ni métricas fuera de muestra.
- `requirements.txt` ya incluye `pandas==2.2.3`, `python-dateutil` y `pytz`
  (usados por el chatbot en otras apps) — suficientes para todo lo pedido en
  esta fase; **no se agregó ninguna dependencia nueva**.
- Línea base ejecutada antes de modificar: `python manage.py test` → **109
  tests, 0 fallos, 0 errores, 0 omitidas, OK (4.92s)**.

### 2. Arquitectura implementada

- **`forecasting/metrics.py`** (nuevo) — funciones puras `calcular_mae`,
  `calcular_rmse`, `calcular_mape`, con validación compartida
  (`_validar_par`) de tipo, booleanos, NaN, Inf y longitud. No conoce
  statsmodels ni el motor de ajuste: la reutilizan tanto el diagnóstico
  in-sample como la evaluación fuera de muestra.
- **`forecasting/temporal.py`** (nuevo) — todo lo relacionado con
  `pandas.DatetimeIndex`: parseo/validación de fechas, normalización de
  alias de frecuencia, inferencia, detección de períodos faltantes y
  generación de fechas futuras. `engine.py` sigue siendo enteramente
  numérico; este módulo es el único que importa `pandas` para fechas.
- **`forecasting/evaluation.py`** (nuevo) — `determinar_tamano_prueba()` y
  `evaluar_holdout_temporal()`. Deliberadamente **agnóstico al modelo**:
  recibe una función `funcion_pronostico(entrenamiento, pasos)` provista por
  el llamador, así la división cronológica y las métricas se reutilizarán
  sin duplicarse desde MA/SARIMA/ARIMAX/SARIMAX en fases futuras.
- **`forecasting/exceptions.py`** (ampliado) — nuevas excepciones de dominio
  para métricas, evaluación, fechas y frecuencia (ver sección 9).
- **`tools/modelo_arima.py`** (ampliado) — orquesta: valida → arma
  `informacion_temporal` → si `evaluar_modelo`, corre el holdout con una
  función que envuelve `engine.ajustar_arima` sobre el tramo de
  entrenamiento → **siempre** reajusta con la serie completa para el
  pronóstico final → arma la respuesta con `evaluacion`,
  `informacion_temporal` y `fechas_pronostico`.

`engine.py` **no se modificó**: sigue siendo puramente numérico. La
evaluación temporal se logra llamando a `engine.ajustar_arima()` dos veces
desde `modelo_arima.py` (una vez con el tramo de entrenamiento dentro del
holdout, otra con la serie completa para el pronóstico final) en vez de
enseñarle al motor a conocer fechas o particiones — esto evita duplicar la
lógica de ajuste y mantiene el motor desacoplado de la evaluación.

### 3. Archivos creados

```
apps/herramientas/forecasting/metrics.py
apps/herramientas/forecasting/temporal.py
apps/herramientas/forecasting/evaluation.py
apps/herramientas/test_forecasting_metrics.py
apps/herramientas/test_forecasting_temporal.py
apps/herramientas/test_forecasting_evaluation.py
apps/herramientas/test_modelo_arima_fase3.py
```

### 4. Archivos modificados

- `apps/herramientas/forecasting/exceptions.py` — se agregaron
  `MetricCalculationError`, `MetricLengthMismatchError`,
  `InvalidEvaluationConfigurationError`, `InsufficientTrainingDataError`,
  `DateValidationError`, `DateLengthMismatchError`, `DuplicateDatesError`,
  `UnsortedDatesError`, `FrequencyValidationError`,
  `InconsistentFrequencyError`. Ninguna excepción de fase 2 se modificó ni
  se eliminó.
- `apps/herramientas/tools/modelo_arima.py` — `TOOL_DEFINITION` ampliado con
  `fechas, frecuencia, evaluar_modelo, cantidad_prueba, porcentaje_prueba`
  (todos opcionales); `_ejecutar_modelo_arima` y `_construir_respuesta`
  reescritos para orquestar evaluación temporal y fechas sin tocar la lógica
  de ajuste ARIMA en sí.

No se tocó `engine.py`, `validation.py`, `diagnostics.py`, `schemas.py`,
`tools.py` (loader), ni ningún archivo fuera de `apps/herramientas/`.

### 5. Métricas

- **Fórmulas:** MAE y RMSE estándar; MAPE = `100/n · Σ|((y-ŷ)/y)|` solo
  sobre observaciones con `y ≠ 0`.
- **Validación compartida:** secuencia no vacía, sin booleanos, solo
  numérica, sin NaN/Inf, misma longitud entre reales y pronosticados
  (`MetricLengthMismatchError` si difiere). Aplica por igual a MAE, RMSE y
  MAPE.
- **Política de MAPE con ceros:** se excluyen del cálculo únicamente las
  observaciones con valor real exactamente `0`; se informa
  `observaciones_excluidas_por_cero` y se agrega una advertencia
  `MAPE_VALORES_CERO_EXCLUIDOS`. Si el 100% de los reales son cero, `mape`
  es `None` con `mape_detalle.calculado = False` y un `motivo` explícito —
  nunca se devuelve infinito ni un número artificialmente grande.
- **Valores cercanos a cero:** tolerancia explícita
  `TOLERANCIA_MAPE_CERCANO_A_CERO = 1e-6` (documentada en el módulo). No se
  excluyen (solo los exactamente cero se excluyen): se incluyen en el
  cálculo y generan la advertencia `MAPE_VALORES_CERCANOS_A_CERO`.
- **Valores negativos:** no se rechazan; generan la advertencia conceptual
  `MAPE_VALORES_NEGATIVOS` ("MAPE puede resultar difícil de interpretar...").
- **Estructura de salida:** `calcular_mae`/`calcular_rmse` devuelven `float`
  nativo; `calcular_mape` devuelve `{"mape": float|None, "mape_detalle":
  {...}}` tal como pide la consigna.
- No se implementó sMAPE/MASE: la consigna la marca como opcional y no
  bloqueante; se priorizó MAE/RMSE/MAPE bien probadas.

### 6. Evaluación temporal

- **Estrategia:** holdout cronológico puro — `serie[:n_entrenamiento]` /
  `serie[n_entrenamiento:]`, nunca aleatorio, nunca mezclado (verificado con
  tests que comprueban que `valores_reales` de prueba es exactamente el
  tramo final de la serie, en orden).
- **Selección del tamaño de prueba** (`determinar_tamano_prueba`):
  prioridad `cantidad_prueba` > `porcentaje_prueba` > default documentado
  (20% de la serie, `PORCENTAJE_PRUEBA_DEFAULT = 0.2`). `porcentaje_prueba`
  debe cumplir `0 < x < 0.5` (límite elegido porque `>=50%` dejaría muy poco
  entrenamiento en las series cortas típicas de este chatbot académico).
  Nunca deja el conjunto de prueba vacío ni consume toda la serie.
- **Reajuste final:** `tools/modelo_arima.py` llama a
  `engine.ajustar_arima()` una segunda vez, siempre con la serie completa,
  **después** de la evaluación — el pronóstico futuro devuelto al usuario
  nunca proviene del modelo ajustado solo con entrenamiento (verificado con
  un test que compara el `pronostico` final con y sin `evaluar_modelo=True`:
  da exactamente igual).
- **Muestra insuficiente:** decisión documentada (en el docstring de
  `evaluation.py` y aquí): una configuración inválida de
  `cantidad_prueba`/`porcentaje_prueba` es un **error duro** (se comporta
  igual que un orden ARIMA inválido). En cambio, si la configuración es
  válida pero no queda entrenamiento suficiente (o el ajuste de evaluación
  falla), se **omite solo la evaluación** (`{"ejecutada": False, "motivo":
  "..."}`) y el ajuste final sobre toda la serie continúa con normalidad —
  tal como recomienda la consigna.
- **Separación de métricas:** `mse_residuos`/`mse_residuos_entrenamiento`
  (in-sample, sobre toda la serie de entrenamiento final) nunca se mezclan
  con `evaluacion.metricas_prueba` (MAE/RMSE/MAPE, calculadas exclusivamente
  sobre el holdout). Verificado con un test dedicado
  (`test_error_residual_separado_de_metricas_de_prueba`).

### 7. Fechas y frecuencia

- **Formatos aceptados:** `fechas` se parsea con `pandas.to_datetime`
  (acepta ISO 8601 — recomendado y documentado en la descripción del
  parámetro — y cualquier formato que pandas reconozca).
- **Inferencia:** con ≥3 fechas y sin `frecuencia` explícita, se usa
  `pandas.infer_freq`; si no puede determinarse, `frecuencia_utilizada` e
  `informacion_temporal.frecuencia_inferida` quedan en `None` y se agrega la
  advertencia `FRECUENCIA_NO_INFERIBLE` — nunca se inventa una frecuencia.
- **Alias:** `diaria, semanal, mensual, trimestral, anual, horaria` →
  `D, W, MS, QS, YS, H` respectivamente, centralizados en
  `temporal.ALIAS_FRECUENCIA`. También se aceptan códigos de pandas
  directos (`D, W, MS, M, QS, Q, YS, Y, H`).
- **Duplicados:** `DuplicateDatesError` → `{"error": "...", "codigo_error":
  "FECHAS_DUPLICADAS"}`. No se agregan ni promedian valores.
- **Desorden:** `UnsortedDatesError` → `FECHAS_DESORDENADAS`. No se
  reordena la serie automáticamente (se preserva la relación
  posición-valor tal como la mandó el usuario).
- **Períodos faltantes:** cuando hay una frecuencia determinada (explícita o
  inferida), se detectan con `pandas.date_range` vs. las fechas presentes;
  se listan (acotadas a 20) en `informacion_temporal.periodos_faltantes` y
  se agrega la advertencia `PERIODOS_FALTANTES` — **no se completan
  valores**, y el ajuste ARIMA continúa tratando las observaciones por
  posición (documentado explícitamente en el docstring de
  `construir_informacion_temporal`).
- **Frecuencia explícita incompatible:** `InconsistentFrequencyError` →
  `FRECUENCIA_INCONSISTENTE`, error duro (no se sustituye silenciosamente).
- **Serie sin fechas:** decisión documentada — no se agrega ninguna
  advertencia de primer nivel tipo `SERIE_SIN_FECHAS` (generaría ruido en
  casi todas las respuestas sin fechas); la ausencia queda autodocumentada
  en `informacion_temporal.fechas_proporcionadas = False`. Verificado con
  `test_serie_sin_fechas_no_genera_ruido_de_advertencias`.
- **Fechas futuras:** `generar_fechas_pronostico` usa
  `pandas.date_range(start=ultima_fecha, ...)` y devuelve `None` (no
  fechas inventadas) si no hay fechas o no hay frecuencia determinada.
  Formato ISO fecha-only salvo frecuencia horaria. Cada entrada de
  `intervalos_pronostico` incluye su `fecha` correspondiente (`null` si no
  hay fechas disponibles).

### 8. Contrato de ARIMA actualizado

- **Argumentos preservados:** `valores, p, d, q` obligatorios;
  `pasos_pronostico, con_constante, nivel_confianza` opcionales, sin cambios
  de nombre ni de default.
- **Argumentos nuevos (todos opcionales):** `fechas, frecuencia,
  evaluar_modelo (default false), cantidad_prueba (default null),
  porcentaje_prueba (default null)`.
- **Claves preservadas:** las 20 claves del contrato de fase 2 (`modelo,
  orden, n_observaciones, coeficientes, aic, bic, mse_residuos,
  mse_residuos_entrenamiento, media_residuos, varianza_residuos, ljung_box,
  pasos_pronostico, pronostico, detalle_coeficientes, diagnostico_residuos,
  intervalos_pronostico, nivel_confianza, tendencia_statsmodels,
  descripcion_tendencia, informacion_ajuste, advertencias`) — todas
  verificadas con un test de compatibilidad dedicado.
- **Claves nuevas:** `evaluacion, informacion_temporal, fechas_pronostico`.
  `intervalos_pronostico[i]` gana la clave `fecha` (aditiva, no rompe a
  quien ya leía `paso/pronostico/limite_inferior/limite_superior`).
- **Compatibilidad verificada:** una llamada idéntica a la de fase 2 (sin
  fechas/frecuencia/evaluar_modelo/cantidad_prueba/porcentaje_prueba) sigue
  funcionando exactamente igual, con las claves nuevas presentes en sus
  valores por defecto (`evaluacion: {"ejecutada": false}`,
  `informacion_temporal.fechas_proporcionadas: false`,
  `fechas_pronostico: null`).

### 9. Excepciones y advertencias

| Código | Excepción | Disparador |
|---|---|---|
| `METRICAS_ENTRADA_INVALIDA` | `MetricCalculationError` | valores no numéricos, booleanos, NaN, Inf, arreglo vacío en MAE/RMSE/MAPE |
| `METRICAS_LONGITUD_INCOMPATIBLE` | `MetricLengthMismatchError` | reales y pronosticados de distinta longitud |
| `CONFIGURACION_PRUEBA_INVALIDA` | `InvalidEvaluationConfigurationError` | `cantidad_prueba`/`porcentaje_prueba` inválidos o fuera de rango |
| `ENTRENAMIENTO_INSUFICIENTE` | `InsufficientTrainingDataError` | reservada para uso futuro (ver decisión en sección 6: hoy la muestra insuficiente no lanza, se omite la evaluación) |
| `FECHA_INVALIDA` | `DateValidationError` | fecha no interpretable, valores nulos en `fechas` |
| `FECHAS_LONGITUD_INCOMPATIBLE` | `DateLengthMismatchError` | `fechas` y `valores` de distinta longitud |
| `FECHAS_DUPLICADAS` | `DuplicateDatesError` | fechas repetidas |
| `FECHAS_DESORDENADAS` | `UnsortedDatesError` | fechas no estrictamente crecientes |
| `FRECUENCIA_INVALIDA` | `FrequencyValidationError` | alias/código de frecuencia no reconocido |
| `FRECUENCIA_INCONSISTENTE` | `InconsistentFrequencyError` | fechas incompatibles con la frecuencia explícita |

Advertencias nuevas, todas como lista estructurada `{codigo, mensaje,
severidad}`: `MAPE_VALORES_CERO_EXCLUIDOS`, `MAPE_VALORES_CERCANOS_A_CERO`,
`MAPE_VALORES_NEGATIVOS`, `FRECUENCIA_NO_INFERIBLE`, `PERIODOS_FALTANTES`.
Ninguna traza interna (traceback, rutas de archivo) se expone en ningún
caso; el `except Exception` genérico de `modelo_arima.py` (de fase 2) sigue
como última frontera de seguridad, ahora también cubriendo los nuevos
flujos.

### 10. Pruebas agregadas

- **`test_forecasting_metrics.py`** (33 tests): MAE (10), RMSE (9), MAPE
  (14) — valores conocidos, predicción perfecta, negativos, ceros,
  longitudes distintas, vacío, NaN, Inf, booleanos, tipo nativo Python,
  exclusión de ceros en MAPE, todos-cero → `null`, cercanos a cero,
  advertencias y conteo exacto de exclusiones.
- **`test_forecasting_temporal.py`** (29 tests): normalización de alias,
  validación de fechas (ISO, con hora, longitud, inválida, nula,
  desordenada, duplicada), información temporal (mensual/trimestral/diaria
  regulares, período faltante, frecuencia no inferible, frecuencia
  explícita compatible/incompatible, alias pedagógico, serie sin fechas) y
  fechas futuras (mensual, trimestral, diario, cantidad correcta, fecha
  posterior, frecuencia desconocida → `None`, formato ISO).
- **`test_forecasting_evaluation.py`** (21 tests): `determinar_tamano_prueba`
  (14 tests: cantidad/porcentaje válidos e inválidos, prioridad, límites,
  default, serie mínima) y `evaluar_holdout_temporal` con una función de
  pronóstico sintética (7 tests: últimas observaciones, orden preservado,
  entrenamiento insuficiente omite sin lanzar, configuración inválida se
  propaga, métricas presentes, serialización JSON).
- **`test_modelo_arima_fase3.py`** (27 tests): evaluación end-to-end vía
  `ejecutar_herramienta` (longitud de prueba, alineación de valores reales,
  métricas presentes, reajuste final con todas las observaciones,
  independencia del pronóstico futuro, separación residual/prueba,
  evaluación imposible con ajuste final posible, evaluación desactivada por
  defecto, porcentaje de prueba, serialización, configuración inválida),
  fechas/frecuencia end-to-end (inferida, alias, incompatible, duplicadas,
  desordenadas, período faltante, sin fechas), fechas futuras (mensual,
  fecha posterior, correspondencia con intervalos, `None` sin frecuencia,
  `None` sin fechas), compatibilidad (llamada de fase 2 intacta, claves
  nuevas con defaults, argumentos nuevos opcionales en `TOOL_DEFINITION`) y
  un caso académico sintético con evaluación (documentando, igual que en
  fase 2, que no existe la serie fija de 24 meses en el repositorio).

Series sintéticas con `numpy.random.default_rng(seed)` fijo y fechas
generadas con `pandas.date_range`; comparaciones con `assertAlmostEqual`/
`np.isfinite`/desigualdades, nunca igualdad exacta de flotantes.

### 11. Comandos ejecutados

```
python manage.py test                                            → (linea base, antes de tocar nada) Ran 109 tests — OK
python manage.py check                                            → 2 issues (warnings preexistentes de allauth, no relacionados)
python -m compileall -f apps/herramientas                         → todos los .py compilan sin errores de sintaxis
python manage.py test apps.herramientas.test_forecasting_metrics apps.herramientas.test_forecasting_temporal apps.herramientas.test_forecasting_evaluation apps.herramientas.test_modelo_arima_fase3 -v 2
                                                                    → 1 fallo inicial (ver mas abajo), corregido, luego 110/110 OK
python -m unittest ...                                             → falla con ImproperlyConfigured si no se define DJANGO_SETTINGS_MODULE (el proyecto requiere el bootstrap de Django; no aplica sin el); repetido con `DJANGO_SETTINGS_MODULE=main.settings` + `django.setup()` manual → Ran 83 tests — OK, 0 errores, 0 fallos
python manage.py test apps.herramientas -v 1                       → Ran 206 tests — OK
python manage.py test                                              → Ran 219 tests — OK (proyecto completo)

smoke test manual (ejecutar_herramienta): ARIMA(0,1,0) con drift, fechas mensuales reales, frecuencia="mensual", evaluar_modelo=True, cantidad_prueba=4
  → informacion_temporal, evaluacion (periodo_entrenamiento/prueba, fechas_prueba, metricas_prueba) y fechas_pronostico coinciden con el formato pedido; JSON serializable.
smoke test manual: fechas duplicadas/desordenadas/longitud incompatible/fecha invalida/frecuencia incompatible/frecuencia invalida/configuracion de evaluacion invalida
  → cada uno devuelve {"error":..., "codigo_error":...} sin excepcion cruda, con el codigo esperado.
smoke test manual: evaluar_modelo=True con cantidad_prueba=22 sobre 24 observaciones (entrenamiento insuficiente)
  → evaluacion.ejecutada=False con motivo, pero el ajuste final y el pronostico se generaron igual (comportamiento documentado en seccion 6).
```

**Un fallo detectado y corregido durante el desarrollo:** el test
`test_fechas_trimestrales_regulares` esperaba que `pandas.infer_freq`
devolviera `"QS"` exacto; en la práctica devuelve un código anclado
(`"QS-OCT"`), que es la frecuencia trimestral correcta pero con información
adicional del mes de anclaje. No era un bug del código de producción — se
corrigió la aserción del test para comparar solo la base de la frecuencia
(`frecuencia.split("-")[0] == "QS"`), y se volvió a ejecutar para confirmar
que pasaba.

### 12. Resultado completo de las pruebas

```
Antes de esta fase (linea base): 109 tests — 109 exitosas, 0 fallos, 0 errores, 0 omitidas — 4.92s
apps.herramientas (fase 1+2+3):  206 tests — 206 exitosas, 0 fallos, 0 errores, 0 omitidas — 1.65s
proyecto completo (todas las apps): 219 tests — 219 exitosas, 0 fallos, 0 errores, 0 omitidas — 5.46s
```

### 13. Riesgos pendientes

- `home.html` sigue sin renderer dedicado para `evaluacion`,
  `informacion_temporal` ni `fechas_pronostico`: esos datos viajan en el
  JSON y son legibles vía "Copiar todo", pero no se muestran en la tarjeta
  de resultado hasta que se actualice el frontend (fuera de alcance de esta
  fase, igual que en fase 2).
- `InsufficientTrainingDataError` quedó definida pero no se usa activamente
  en el flujo actual (la muestra insuficiente se maneja como
  `{"ejecutada": false}`, no como excepción) — queda documentada por si una
  fase futura necesita que la evaluación sea obligatoria.
- La detección de zona horaria es best-effort: `pandas.to_datetime` preserva
  tz-aware timestamps si son consistentes entre sí, pero no hay una
  validación explícita adicional de zona horaria más allá de lo que pandas
  ya impone.
- Sigue sin variables exógenas, componentes estacionales ni selección
  automática de órdenes — deliberadamente fuera de esta fase.
- El "caso académico" de 24 meses sigue sin existir como fixture en el
  repositorio; la prueba de regresión equivalente vuelve a usar una serie
  sintética documentada como tal.

### 14. Preparación para la fase 4

El núcleo queda preparado para incorporar una herramienta pública MA(q) sin
duplicar:

- **Ajuste:** `engine.ajustar_arima(serie, p=0, d=0, q=q, ...)` ya cubre
  MA(q) como caso particular de ARIMA; una futura `modelo_ma.py` puede
  llamarlo directamente (o `engine.py` puede ganar un alias
  `ajustar_ma(serie, q, ...)` que delegue en `ajustar_arima` con `p=d=0`).
- **Métricas:** `metrics.calcular_mae/rmse/mape` son agnósticas al modelo;
  no requieren cambios.
- **Evaluación:** `evaluation.evaluar_holdout_temporal` ya es agnóstica al
  modelo (recibe `funcion_pronostico` como parámetro); MA(q) solo necesita
  pasar una función que envuelva su propio ajuste, igual que hace
  `modelo_arima.py`.
- **Fechas:** `temporal.construir_informacion_temporal` y
  `temporal.generar_fechas_pronostico` no conocen ARIMA ni ningún orden
  específico; son reutilizables sin cambios.
- **Diagnóstico:** `diagnostics.construir_diagnostico_residuos(residuos, d,
  p, q)` ya acepta `p`/`q` como parámetros (no asume ARIMA): para MA(q) se
  invocaría con `p=0`.
- **Intervalos:** ya genéricos en `engine.ajustar_arima` vía
  `get_forecast().conf_int()`.

No se implementó MA todavía, tal como pedía la consigna.

---

## Fase 4 — Herramienta pública MA(q)

**Alcance de la fase:** nueva herramienta pública `modelo_ma`, construida
enteramente como fachada sobre el núcleo compartido de fases 2-3. No se
implementó SARIMA, ARIMAX ni SARIMAX.

### 1. Diagnóstico previo

Releí `informe.md` (fases 1-3) y reverifiqué contra el código real antes de
tocar nada. Línea base ejecutada: `python manage.py test` → **219 tests, 0
fallos, 0 errores, 0 omitidas, OK (5.80s)**. Confirmé en el propio núcleo:

- El motor se invoca siempre como `engine.ajustar_arima(serie, p, d, q,
  con_constante, pasos_pronostico, nivel_confianza)`; el orden interno es la
  tupla `(p, d, q)` — para MA(q) alcanza con llamarlo con `p=0, d=0, q=q`.
  No fue necesario tocar `engine.py`.
- Las validaciones viven en `forecasting/validation.py` como funciones
  puras (`validar_serie`, `validar_serie_no_constante`,
  `validar_orden_arima`, `validar_horizonte_pronostico`,
  `validar_nivel_confianza`); todas reutilizables tal cual.
- El diagnóstico de residuos (`diagnostics.construir_diagnostico_residuos`)
  ya recibe `p` y `q` por separado y calcula `model_df = p + q`: para MA
  alcanza con pasar `p=0`, sin tocar el módulo.
- Las métricas (`metrics.calcular_mae/rmse/mape`) y la evaluación temporal
  (`evaluation.evaluar_holdout_temporal`, agnóstica al modelo vía
  `funcion_pronostico`) ya estaban diseñadas en fase 3 para no depender de
  ARIMA — se confirmó que MA podía reutilizarlas sin ningún cambio.
- Fechas/frecuencia (`temporal.py`) tampoco conocen el modelo: reutilizables
  directamente.
- El ADF (`apps/herramientas/tools/modelo_dickey_fuller.py`) y la ACF
  (`apps/herramientas/tools/acf.py`) son herramientas públicas independientes
  ya registradas en `TOOL_REGISTRY`; para reutilizarlas sin duplicar su
  lógica, la única vía limpia es invocarlas por el mismo mecanismo que usa
  el chatbot: `apps.herramientas.tools.ejecutar_herramienta(nombre, args)`.
- La carga dinámica (`apps/herramientas/tools.py`) sigue haciendo
  `glob("*.py")` no recursivo sobre `tools/`, sin registro manual: un
  archivo nuevo ahí se descubre solo.

### 2. Arquitectura utilizada

`apps/herramientas/tools/modelo_ma.py` es una fachada pura: no ajusta
modelos, no calcula métricas, no implementa Ljung-Box, ACF ni ADF. Su
`_ejecutar_modelo_ma`:

1. Valida `q` (regla propia: entero, no booleano, `1 <= q <= 20`) reutilizando
   `validation.validar_orden_arima(0, 0, q)` para el chequeo genérico de
   tipo/signo antes de aplicar la regla específica de MA.
2. Valida la serie con las mismas funciones que usa ARIMA
   (`validation.validar_serie`, `validar_serie_no_constante`,
   `validar_horizonte_pronostico`, `validar_nivel_confianza`).
3. Evalúa estacionariedad llamando a
   `ejecutar_herramienta("modelo_dickey_fuller", {"valores": valores})` (el
   mismo camino que usaría el chatbot) — **no reimplementa ADF**.
4. Arma `informacion_temporal` con `temporal.construir_informacion_temporal`.
5. Si `evaluar_modelo`, llama a `evaluation.evaluar_holdout_temporal` pasando
   una función que envuelve `engine.ajustar_arima(..., p=0, d=0, q=q, ...)`
   sobre el tramo de entrenamiento — **no reimplementa el holdout**.
6. Ajusta la serie completa con el mismo `engine.ajustar_arima(..., p=0,
   d=0, q=q, ...)` para el pronóstico final.
7. Arma el diagnóstico con `diagnostics.construir_diagnostico_residuos`,
   clasifica coeficientes MA (ya vienen etiquetados `"media_movil"` por
   `diagnostics.clasificar_parametro`, sin cambios), compara `q` con la
   sugerencia de `ejecutar_herramienta("acf", {"valores": valores})`, informa
   la política de invertibilidad y agrega la explicación pedagógica.

El import de `ejecutar_herramienta` dentro de `_evaluar_estacionariedad` y
`_construir_identificacion` es **diferido** (dentro de la función, no a
nivel de módulo): `apps/herramientas/tools.py` construye `TOOL_REGISTRY`
ejecutando `modelo_ma.py` durante su propia carga dinámica, así que un
`import` a nivel de módulo de `apps.herramientas.tools` crearía un ciclo
(los nombres `ejecutar_herramienta`/`TOOL_REGISTRY` todavía no existirían en
ese momento). El import diferido rompe el ciclo sin ninguna duplicación de
lógica.

### 3. Archivos creados

```
apps/herramientas/tools/modelo_ma.py
apps/herramientas/test_modelo_ma.py
```

### 4. Archivos modificados

- `apps/herramientas/forecasting/exceptions.py` — se agregó
  `InvalidMAOrderError(InvalidOrderError)` con `codigo_error =
  "ORDEN_MA_INVALIDO"`, usada solo para la regla específica de MA (`q>=1`,
  `q<=20`); los problemas genéricos de tipo/signo de `q` siguen cayendo en
  el `InvalidOrderError` genérico (`ORDEN_INVALIDA`), reutilizado sin
  cambios. Ninguna excepción existente se modificó ni se eliminó.
- `apps/herramientas/forecasting/validation.py` — se generalizó el texto
  del mensaje de `validar_serie_no_constante` (antes decía "no corresponde
  ajustar un modelo ARIMA", ahora "no corresponde ajustar un modelo de esta
  familia (ARIMA/MA)"). Es una corrección imprescindible para que el mensaje
  tenga sentido también cuando lo dispara `modelo_ma`: no cambia
  `codigo_error` (sigue siendo `SERIE_INVALIDA`) ni ninguna clave de
  contrato; se verificó que ningún test existente comparaba el texto exacto
  antes de tocarlo.

No se tocó `engine.py`, `diagnostics.py`, `metrics.py`, `evaluation.py`,
`temporal.py`, `schemas.py`, `tools.py` (loader), `tools/modelo_arima.py`,
`tools/acf.py` ni `tools/modelo_dickey_fuller.py`.

### 5. Contrato de entrada

**Obligatorios:** `valores`, `q`.
**Opcionales:** `pasos_pronostico` (default 1), `con_constante` (default
true), `nivel_confianza` (default 0.95), `fechas`, `frecuencia`,
`evaluar_modelo` (default false), `cantidad_prueba`, `porcentaje_prueba` —
mismos nombres, tipos y defaults que en `modelo_arima`, para que un usuario
que ya conoce ARIMA no tenga que aprender un vocabulario nuevo.

### 6. Contrato de salida

Claves devueltas (superconjunto conceptual del ejemplo de la consigna):
`modelo` (`"MA(q)"`), `representacion_interna` (`"ARIMA(0,0,q)"`), `orden`,
`orden_q`, `n_observaciones`, `coeficientes`, `coeficientes_ma`,
`detalle_coeficientes`, `estacionariedad`, `invertibilidad`,
`identificacion`, `aic`, `bic`, `mse_residuos`,
`mse_residuos_entrenamiento`, `media_residuos`, `varianza_residuos`,
`ljung_box`, `diagnostico_residuos`, `evaluacion`, `informacion_temporal`,
`pasos_pronostico`, `pronostico`, `fechas_pronostico`,
`intervalos_pronostico`, `nivel_confianza`, `tendencia_statsmodels`,
`descripcion_tendencia`, `informacion_ajuste`, `explicacion_modelo`,
`advertencias`. Ejemplo real (MA(1) con fechas mensuales y evaluación,
recortado):

```json
{
  "modelo": "MA(1)",
  "representacion_interna": "ARIMA(0,0,1)",
  "orden_q": 1,
  "coeficientes_ma": {"ma.L1": 0.6539},
  "estacionariedad": {"prueba": "ADF", "ejecutada": true, "p_value": 0.13, "evidencia_estacionariedad": false},
  "invertibilidad": {"forzada_por_statsmodels": true, "verificacion_manual": false},
  "identificacion": {"q_solicitado": 1, "q_sugerido_acf": 1, "coincide_con_sugerencia": true},
  "evaluacion": {"ejecutada": true, "metricas_prueba": {"mae": 1.97, "rmse": 2.34, "mape": 9.02}},
  "fechas_pronostico": ["2025-01-01", "2025-02-01"],
  "explicacion_modelo": {"descripcion": "...", "diferencia_promedio_movil": "..."},
  "advertencias": [{"codigo": "SERIE_NO_ESTACIONARIA_PARA_MA", "severidad": "advertencia_alta"}]
}
```

### 7. Reutilización

| Responsabilidad | Módulo reutilizado | ¿Se tocó? |
|---|---|---|
| Ajuste | `forecasting/engine.py: ajustar_arima(p=0,d=0,q=q,...)` | No |
| Validación de serie/horizonte/confianza | `forecasting/validation.py` | Solo texto de un mensaje (ver sección 4) |
| Diagnóstico de residuos / Ljung-Box | `forecasting/diagnostics.py` | No |
| Métricas | `forecasting/metrics.py` | No |
| Evaluación temporal | `forecasting/evaluation.py` | No |
| Fechas/frecuencia | `forecasting/temporal.py` | No |
| Estacionariedad (ADF) | `tools/modelo_dickey_fuller.py` vía `ejecutar_herramienta` | No |
| Identificación (ACF) | `tools/acf.py` vía `ejecutar_herramienta` | No |
| Serialización JSON | `tools.py: _to_json_safe` (aplicado por `ejecutar_herramienta`) | No |

### 8. Decisiones estadísticas

- **Tamaño mínimo:** `minimo_tecnico = q + (1 si con_constante) + 3` (mismo
  margen de 3 que usa ARIMA para `p+d+q+3`); bloquea el ajuste
  (`InsufficientDataError`, `MUESTRA_INSUFICIENTE`) si no se cumple. Además,
  `minimo_recomendado = max(minimo_tecnico, 4q + 15)` genera una advertencia
  no bloqueante (`MUESTRA_MA_REDUCIDA`) cuando la muestra alcanza para
  ajustar pero es chica para un diagnóstico confiable — la distinción entre
  "no se puede" y "se puede pero con reservas" que pedía la consigna.
- **Constante:** `con_constante=True` → `trend="c"` (nunca `"t"`/drift,
  porque `d=0`); `con_constante=False` → `trend="n"`. Reutiliza
  `validation.resolver_tendencia(0, con_constante)` tal cual.
- **Estacionariedad:** se ejecuta ADF siempre (vía `modelo_dickey_fuller`).
  Si no hay evidencia de estacionariedad, se agrega una advertencia
  `severidad: "advertencia_alta"` con código
  `SERIE_NO_ESTACIONARIA_PARA_MA` recomendando ARIMA con `d>0` — **nunca se
  bloquea el ajuste ni se diferencia automáticamente** (se sigue ajustando
  MA(q) puro con `d=0`), tal como pedía la consigna.
- **Comportamiento ante serie no estacionaria:** solo advertencia, nunca
  conversión silenciosa a ARIMA. Verificado con test dedicado
  (`test_no_diferencia_automaticamente`: confirma `orden.d == 0` siempre).
- **Invertibilidad:** se informa `enforce_invertibility` (heredado del
  motor, sin cambios) y se aclara explícitamente `verificacion_manual:
  false`. **Decisión:** no se calculan raíces MA manualmente en esta fase —
  requeriría exponer el objeto de resultado crudo de statsmodels desde
  `engine.py` (hoy devuelve solo el `ResultadoAjusteARIMA` normalizado, sin
  el ajuste crudo), lo cual la consigna permite solo "si no agrega
  duplicación compleja"; se prefirió no ampliar el contrato interno del
  motor por esto y quedó documentado como limitación (sección 14).
- **`model_df`:** `diagnostics.construir_diagnostico_residuos(residuos, d=0,
  p=0, q=q)` ya calcula `model_df = p + q = q` sin ningún cambio.
- **Selección de lag:** reutiliza `diagnostics.seleccionar_lag_ljung_box`
  tal cual (exige `lag > model_df` y `lag < cantidad_residuos`; si no hay
  lag válido, `ejecutado: false` con `motivo`, nunca `NaN`).
- **ACF:** `identificacion.q_sugerido_acf` se informa solo como referencia
  (vía `ejecutar_herramienta("acf", ...)`); el `q` del usuario nunca se
  sobreescribe ni se declara "incorrecto" por no coincidir.

### 9. Manejo de errores

| Código | Origen | Nota |
|---|---|---|
| `ORDEN_MA_INVALIDO` | `InvalidMAOrderError` (nueva, subclase de `InvalidOrderError`) | `q<1` o `q>20` |
| `ORDEN_INVALIDA` | `InvalidOrderError` (reutilizada) | `q` booleano, no entero o negativo |
| `SERIE_INVALIDA` | `InvalidSeriesError` (reutilizada) | vacía, no numérica, booleana, NaN, Inf, constante |
| `MUESTRA_INSUFICIENTE` | `InsufficientDataError` (reutilizada) | `n < minimo_tecnico` — se decidió **no** crear `MUESTRA_INSUFICIENTE_MA` porque el problema es idéntico conceptualmente al de ARIMA (tamaño de muestra insuficiente para el orden pedido), solo cambia la fórmula que calcula el mínimo |
| `HORIZONTE_INVALIDO` / `NIVEL_CONFIANZA_INVALIDO` | reutilizadas | igual que ARIMA |
| `FECHA_INVALIDA` / `FECHAS_LONGITUD_INCOMPATIBLE` / `FECHAS_DUPLICADAS` / `FECHAS_DESORDENADAS` / `FRECUENCIA_INVALIDA` / `FRECUENCIA_INCONSISTENTE` | reutilizadas de fase 3 | sin cambios |
| `CONFIGURACION_PRUEBA_INVALIDA` | reutilizada de fase 3 | `cantidad_prueba`/`porcentaje_prueba` inválidos |
| `ERROR_AJUSTE` / `ERROR_NUMERICO` / `ERROR_PRONOSTICO_GENERACION` | reutilizadas de `engine.py` | cubren "AJUSTE_MA_FALLIDO"/"PRONOSTICO_MA_FALLIDO" de la consigna sin crear códigos nuevos, porque MA llama al mismo `engine.ajustar_arima` que ya los produce |
| `ADF_NO_EJECUTABLE` | advertencia (no excepción) | cuando `modelo_dickey_fuller` devuelve `error` |
| `SERIE_NO_ESTACIONARIA_PARA_MA` | advertencia `severidad: "advertencia_alta"` | serie sin evidencia de estacionariedad |
| `MUESTRA_MA_REDUCIDA` | advertencia | `n` entre el mínimo técnico y el recomendado |

No se creó ninguna excepción para "muestra insuficiente MA" ni para
"ajuste/pronóstico MA fallido": las genéricas del núcleo ya representan el
problema correctamente, tal como permitía la consigna.

### 10. Pruebas agregadas

**`test_modelo_ma.py`** (82 tests) organizados en los mismos grupos que pidió
la consigna: contrato (7: registro dinámico, `TOOL_DEFINITION`, `TOOL_META`,
ejecutable, obligatorios, opcionales, serializable), validación (16: `q` en
1/2/0/negativo/decimal/booleano/excesivo, serie vacía/corta/constante/texto/
booleanos/NaN/Inf, horizonte y nivel de confianza inválidos), MA(1) (9),
MA(2) (8), constante (4), estacionariedad (5: serie estacionaria ejecuta ADF,
serie con tendencia genera advertencia, recomendación de ARIMA, no
diferencia automáticamente, muestra pequeña no bloquea), invertibilidad (3),
Ljung-Box (5: `model_df=q`, lag>q, estructura completa, muestra pequeña sin
NaN, sin la frase "modelo válido"), evaluación temporal (9: desactivada,
activada, últimas observaciones, MAE/RMSE/MAPE, reajuste completo,
independencia del pronóstico futuro, cantidad/porcentaje de prueba,
entrenamiento insuficiente, MAPE con cero), fechas (9: mensuales, frecuencia
inferida/explícita, fechas futuras, intervalos con fecha, duplicadas,
desordenadas, período faltante, sin fechas), pedagógicas (4: explicación de
errores pasados, diferencia con promedio móvil, advertencia de
estacionariedad, identificación vía ACF sin forzar selección) y regresión
(2: `modelo_arima` sigue funcionando, ambas herramientas conviven en
`TOOL_REGISTRY` junto con `modelo_ar`/`acf`/`modelo_dickey_fuller`).

Series MA(1)/MA(2) sintéticas construidas manualmente con
`numpy.random.default_rng(seed)` fijo (sin dependencia nueva: no se usó
`statsmodels.tsa.arima_process` porque una construcción manual con ruido
blanco ya alcanzaba y evita acoplar los tests a otra API de statsmodels).

### 11. Comandos ejecutados

```
python manage.py test                                          → (linea base, antes de tocar nada) Ran 219 tests — OK
python manage.py test apps.herramientas.test_modelo_ma -v 2     → Ran 82 tests — OK (0 fallos, 0 errores, en el primer intento)
python manage.py check                                          → 2 issues (warnings preexistentes de allauth, no relacionados)
python -m compileall -f apps/herramientas                        → todos los .py compilan sin errores, incluye tools/modelo_ma.py
python -c "...unittest con django.setup() manual..."             → Ran 82 tests — OK, 0 errores, 0 fallos (equivalente a `python -m unittest` con el bootstrap que Django exige)
python manage.py test apps.herramientas -v 1                     → Ran 288 tests — OK
python manage.py test                                            → Ran 301 tests — OK (proyecto completo)
```

**Smoke tests manuales adicionales** (antes de escribir la suite formal):
MA(1) con serie sintética completa (coeficientes, estacionariedad,
invertibilidad, identificación, advertencias, pronóstico, JSON); batería de
validaciones inválidas (`q` en 0/-1/True/1.5/25, series vacía/constante/
corta/texto/bool/NaN/Inf, horizonte y confianza inválidos); MA(2) con
`evaluar_modelo=True`; `con_constante=False`; muestra pequeña con
advertencia; MA(1) con fechas mensuales + frecuencia + evaluación
combinadas, verificando `limite_inferior <= pronostico <= limite_superior`
en todos los pasos.

### 12. Resultado completo de las pruebas

```
Antes de esta fase (linea base): 219 tests — 219 exitosas, 0 fallos, 0 errores, 0 omitidas — 5.80s
apps.herramientas.test_modelo_ma (aislado): 82 tests — 82 exitosas, 0 fallos, 0 errores, 0 omitidas — 2.17s
apps.herramientas (fases 1-4 completas): 288 tests — 288 exitosas, 0 fallos, 0 errores, 0 omitidas — 3.35s
proyecto completo (todas las apps): 301 tests — 301 exitosas, 0 fallos, 0 errores, 0 omitidas — 7.65s
```

### 13. Compatibilidad

`modelo_arima` no se tocó salvo lo indicado en la sección 4 (un mensaje de
texto genérico en `validation.py`, sin cambio de `codigo_error` ni de
ninguna clave). Se verificó explícitamente con
`RegresionARIMATests.test_modelo_arima_sigue_funcionando` (ajusta
ARIMA(0,1,0) con drift y confirma claves/longitud de pronóstico) y con la
suite completa de fases 1-3 (219 tests) corriendo sin cambios dentro de la
suite de 301. `modelo_ar`, `acf`, `pacf`, `modelo_dickey_fuller`,
`descomposicion_visualizacion_serie`, `estabilizacion_media/varianza` y los
8 tests de `apps.chatbot` (incluyendo el flujo completo de function calling
mockeado) siguen pasando sin cambios.

### 14. Limitaciones

- No se calculan raíces MA manualmente (ver decisión en sección 8):
  `invertibilidad.verificacion_manual` es siempre `false`. Ampliar esto
  requeriría que `engine.py` exponga el objeto de ajuste crudo de
  statsmodels (o sus raíces) en `ResultadoAjusteARIMA`, lo cual se dejó
  pendiente para no tocar el contrato interno del motor en esta fase.
- `home.html` no tiene un renderer dedicado para `modelo_ma`: como con
  cualquier herramienta sin renderer propio, cae al fallback genérico de
  JSON crudo del frontend (comportamiento ya existente, documentado desde
  fase 1, no es una regresión).
- La ACF usada para `q_sugerido_acf` sigue siendo una heurística de corte
  simple (documentado desde fase 1): la comparación con `q_solicitado` es
  solo informativa, nunca determinante.
- Sigue sin SARIMA, ARIMAX, SARIMAX, componentes estacionales, variables
  exógenas ni selección automática de órdenes — deliberadamente fuera de
  esta fase.

### 15. Preparación para la fase 5

El núcleo está preparado para incorporar SARIMA (`order=(p,d,q)`,
`seasonal_order=(P,D,Q,s)`) sin duplicar:

- **Ajuste:** `statsmodels.tsa.arima.model.ARIMA` (usado por
  `engine.ajustar_arima`) ya acepta `seasonal_order` de forma nativa (se
  confirmó en la inspección de fase 1: `ARIMA.__init__` tiene el parámetro
  `seasonal_order=(0,0,0,0)` por defecto). `engine.ajustar_arima` necesitaría
  un parámetro opcional `seasonal_order` que se pase directo al constructor;
  no requiere una segunda función de ajuste.
- **Métricas/Evaluación:** sin cambios, ya son agnósticas al orden.
- **Fechas:** sin cambios; la validación de períodos faltantes ya usa la
  frecuencia detectada, compatible con series estacionales.
- **Diagnóstico:** `construir_diagnostico_residuos` tomaría
  `model_df = p+q+P+Q` en vez de `p+q` — un ajuste menor y localizado, no
  una reescritura.
- **Intervalos:** sin cambios, ya genéricos vía `get_forecast()`.

---

## Fase 5 — Herramienta pública SARIMA(p,d,q)(P,D,Q)_s

**Alcance de la fase:** nueva herramienta pública `modelo_sarima`, más una
ampliación puntual y compatible del núcleo compartido (`seasonal_order`
opcional). No se implementó ARIMAX, variables exógenas ni la herramienta
pública SARIMAX.

### 1. Diagnóstico previo

Releí `informe.md` (fases 1-4) y reverifiqué contra el código real. Línea
base ejecutada: `python manage.py test` → **301 tests, 0 fallos, 0 errores,
0 omitidas, OK (7.36s)**. Verifiqué en el propio núcleo:

- El motor (`engine.ajustar_arima`) seguía sin aceptar `seasonal_order`;
  `order=(p,d,q)` era la única representación de orden.
- `resolver_tendencia(d, con_constante)` solo consideraba `d`, no `d+D`.
- `diagnostics.construir_diagnostico_residuos` calculaba `model_df=p+q` y
  descartaba siempre `d` residuos iniciales; `clasificar_parametro` no
  distinguía parámetros estacionales (`ar.S.L12` caía en "autorregresivo").
- Ljung-Box (`seleccionar_lag_ljung_box`) no tenía forma de preferir un
  rezago relacionado con un ciclo estacional.
- La evaluación temporal y las fechas/frecuencia ya eran completamente
  agnósticas al modelo (confirmado en fases 3-4): reutilizables sin tocar.
- Confirmé empíricamente (ver sección 3) que `ARIMA` de statsmodels ya
  soporta `seasonal_order` de forma nativa y que la restricción de
  tendencia de statsmodels depende de `d+D` (no solo `d`): el propio mensaje
  de error de `ARIMA.__init__` lo dice explícitamente ("models with
  integration (d>0) or seasonal integration (D>0)").

### 2. Arquitectura utilizada

Se amplió `engine.ajustar_arima` con un parámetro opcional
`seasonal_order: tuple[int,int,int,int] | None = None` (default `None` →
`(0,0,0,0)`, comportamiento idéntico al de fases 2-4 para ARIMA/MA). Con
`seasonal_order` provisto, se pasa directo al constructor de `ARIMA` junto
con `order=(p,d,q)`. La tendencia se resuelve con `resolver_tendencia(d,
con_constante, D=D)` (generalizada para considerar `d+D`, con `D=0` por
defecto). `ResultadoAjusteARIMA` gana el campo `orden_estacional` (opcional,
`None` para ARIMA/MA) para que el resultado normalizado sea autoconsistente.

`apps/herramientas/tools/modelo_sarima.py` es una fachada pura, del mismo
estilo que `modelo_ma.py`: valida sus propios parámetros (P, D, Q, s,
ciclos, complejidad, coherencia frecuencia-periodicidad), reutiliza
`engine.ajustar_arima(..., seasonal_order=(P,D,Q,s))` para el ajuste,
`evaluation.evaluar_holdout_temporal` para la evaluación (pasando
`seasonal_order` dentro de la función de pronóstico, igual que hace MA con
`p=0,d=0`), `temporal.*` para fechas y `diagnostics.construir_diagnostico_residuos`
(ahora con `P`, `Q` y `descarte_inicial=d+D*s` explícitos) para el
diagnóstico. La estacionariedad regular se evalúa reutilizando
`modelo_dickey_fuller` (vía `ejecutar_herramienta`, igual que MA) y la
identificación estacional reutiliza `acf` pidiendo explícitamente
`lags=2*s`.

**Refactor retroactivo en `modelo_ma.py`:** se extrajo la interpretación del
dict crudo de ADF (antes duplicada inline en `modelo_ma.py`) a una función
nueva y compartida, `diagnostics.interpretar_resultado_adf`, que ahora usan
tanto `modelo_ma.py` como `modelo_sarima.py`. Se verificó con la suite
completa de `test_modelo_ma.py` (82 tests) que el comportamiento observable
no cambió.

### 3. Clase de statsmodels seleccionada

**`statsmodels.tsa.arima.model.ARIMA`** (la misma que ya usa `engine.py`
desde fase 2), **no** se instancia `SARIMAX` por separado. Justificación,
confirmada empíricamente en esta fase:

```python
>>> from statsmodels.tsa.arima.model import ARIMA
>>> from statsmodels.tsa.statespace.sarimax import SARIMAX
>>> issubclass(ARIMA, SARIMAX)
True
```

`ARIMA` es literalmente una subclase de `SARIMAX` y acepta `seasonal_order`
de forma nativa (probado con `ARIMA(serie, order=(1,0,0),
seasonal_order=(1,1,0,12), trend='c')` → ajusta correctamente y expone
`ar.S.L12` en `param_names`). Usar `ARIMA` en vez de instanciar `SARIMAX`
directamente evita: (a) una segunda función de ajuste con su propia lógica
de tendencia/advertencias/forecast (duplicación explícitamente prohibida),
y (b) dos rutas de código divergentes para el mismo problema matemático. La
clave `representacion_interna` de la salida dice explícitamente "SARIMAX
sin variables exógenas (vía statsmodels.tsa.arima.model.ARIMA, subclase de
SARIMAX)" para ser transparente sobre qué se ejecuta realmente, tal como
sugiere la consigna.

### 4. Archivos creados

```
apps/herramientas/tools/modelo_sarima.py
apps/herramientas/test_modelo_sarima.py
```

### 5. Archivos modificados

- `apps/herramientas/forecasting/schemas.py` — `ResultadoAjusteARIMA` gana
  el campo `orden_estacional: Optional[tuple[int,int,int,int]]`.
- `apps/herramientas/forecasting/validation.py` — `resolver_tendencia` gana
  el parámetro opcional `D: int = 0`; con `D=0` (default) el comportamiento
  es idéntico al de fases 2-4. La restricción ahora se evalúa sobre `d+D`
  (confirmado que así lo exige statsmodels) y la descripción menciona el
  origen de la diferenciación ("vía diferenciación regular"/"estacional").
- `apps/herramientas/forecasting/diagnostics.py` — `clasificar_parametro`
  distingue `ar.S.`/`ma.S.` (`autorregresivo_estacional`/
  `media_movil_estacional`) antes de los prefijos genéricos `ar.`/`ma.`;
  `seleccionar_lag_ljung_box` y `ejecutar_ljung_box` ganan el parámetro
  opcional `periodo_estacional` (default `None`, sin cambio de
  comportamiento si no se usa) y `ejecutar_ljung_box` agrega la clave
  `incluye_rezago_estacional`; `construir_diagnostico_residuos` gana los
  parámetros opcionales `P=0, Q=0, descarte_inicial=None,
  periodo_estacional=None` (con los defaults, `model_df=p+q` y el descarte
  usa `d`, igual que antes); se agregó la función nueva
  `interpretar_resultado_adf` (extraída de `modelo_ma.py`, ver sección 2).
- `apps/herramientas/forecasting/engine.py` — `ajustar_arima` gana el
  parámetro opcional `seasonal_order` (ver sección 2).
- `apps/herramientas/forecasting/exceptions.py` — se agregaron
  `InvalidSeasonalOrderError` (`ORDEN_ESTACIONAL_INVALIDO`) e
  `InvalidSeasonalPeriodError` (`PERIODICIDAD_INVALIDA`), ambas subclases
  de `InvalidOrderError`.
- `apps/herramientas/tools/modelo_ma.py` — `_evaluar_estacionariedad` se
  reescribió para delegar en `diagnostics.interpretar_resultado_adf` (ver
  sección 2); el resto del archivo no cambió.

Todos los cambios al núcleo son **aditivos con defaults neutros**
(parámetros opcionales que preservan el comportamiento anterior cuando se
omiten); se verificó con la suite completa de fases 1-4 (301 tests)
corriendo sin cambios dentro de la suite final de 396.

### 6. Contrato de entrada

**Obligatorios:** `valores, p, d, q, P, D, Q, s` (los ocho, sin defaults,
tal como pedía la consigna). **Opcionales:** `pasos_pronostico` (1),
`con_constante` (true), `nivel_confianza` (0.95), `fechas`, `frecuencia`,
`evaluar_modelo` (false), `cantidad_prueba`, `porcentaje_prueba` — mismos
nombres/tipos/defaults que ARIMA y MA.

### 7. Contrato de salida

Claves principales: `modelo` (`"SARIMA(p,d,q)(P,D,Q,s)"`),
`representacion_interna`, `orden`, `orden_estacional`, `diferenciacion`
(`regular`/`estacional`/`periodicidad`), `n_observaciones`,
`n_ciclos_aproximados`, `coeficientes`, `coeficientes_regulares`,
`coeficientes_estacionales`, `detalle_coeficientes`, `estacionariedad`
(`regular`/`estacional`), `coherencia_estacional`,
`identificacion_estacional`, `aic`, `bic`, `mse_residuos`,
`mse_residuos_entrenamiento`, `diagnostico_residuos` (con `ljung_box`
incluyendo `incluye_rezago_estacional`), `evaluacion`,
`informacion_temporal`, `pasos_pronostico`, `pronostico`,
`fechas_pronostico`, `intervalos_pronostico`, `nivel_confianza`,
`tendencia_statsmodels`, `descripcion_tendencia`, `informacion_ajuste`,
`advertencias`. Ejemplo real (SARIMA(1,1,1)(1,1,1,12) mensual, recortado):

```json
{
  "modelo": "SARIMA(1,1,1)(1,1,1,12)",
  "orden_estacional": {"P": 1, "D": 1, "Q": 1, "s": 12},
  "diferenciacion": {"regular": 1, "estacional": 1, "periodicidad": 12},
  "n_ciclos_aproximados": 4.0,
  "coeficientes_regulares": {"ar.L1": 0.086, "ma.L1": -0.9406},
  "coeficientes_estacionales": {"ar.S.L12": -0.7214, "ma.S.L12": -0.181},
  "ljung_box": {"ejecutado": true, "lags": 12, "model_df": 4, "incluye_rezago_estacional": true, "p_value": 0.766},
  "estacionariedad": {"estacional": {"orden_D_solicitado": 1, "periodicidad": 12}},
  "advertencias": [{"codigo": "ADF_NO_DETERMINA_DIFERENCIACION_ESTACIONAL", "severidad": "advertencia"}]
}
```

### 8. Validaciones estacionales

- **Órdenes P, D, Q:** enteros, no booleanos, no negativos
  (`InvalidSeasonalOrderError`, `ORDEN_ESTACIONAL_INVALIDO`); máximos
  pedagógicos propios de esta herramienta (no de ARIMA, que no fija techo a
  p/q): `P<=3, Q<=3, D<=1` (se eligió `D<=1` en vez de 2: `D=2` exige `2*s`
  observaciones extra solo para diferenciar, poco realista para series
  académicas cortas). `p<=5, q<=5` también acotados aquí (con el código
  genérico `ORDEN_INVALIDA`, no el estacional, porque no son órdenes
  estacionales).
- **Periodicidad `s`:** entero, no booleano, `s>=2`, `s<=366`
  (`InvalidSeasonalPeriodError`, `PERIODICIDAD_INVALIDA`).
- **Complejidad:** `minimo_tecnico = d + D*s + (p+q+P+Q+constante) + 3`
  (bloquea con `InsufficientDataError`/`MUESTRA_INSUFICIENTE` si no se
  cumple); si se cumple pero `n < parametros*5`, advertencia no bloqueante
  `CONFIGURACION_ESTACIONAL_COMPLEJA`.
- **Ciclos:** `n_ciclos_aproximados = n/s`; `n < 2*s` →
  `CICLOS_ESTACIONALES_INSUFICIENTES` (`advertencia_alta`, documentando
  explícitamente que 2 ciclos son un mínimo técnico orientativo, no una
  garantía estadística); `2*s <= n < 3*s` → `CICLOS_ESTACIONALES_LIMITADOS`
  (advertencia suave, recomienda 3+ ciclos).
- **Frecuencia vs. `s`:** `coherencia_estacional` clasifica la combinación
  como `"habitual"` (mensual+12, trimestral+4, diario+7, horario+24,
  semanal+52) o `"poco_habitual"` (matemáticamente válida pero menos común,
  con advertencia informativa `FRECUENCIA_Y_PERIODICIDAD_INUSUALES`); no se
  rechaza ninguna combinación matemáticamente posible. No se implementó una
  tercera categoría "inválida": cualquier incompatibilidad real ya la
  bloquea `FRECUENCIA_INCONSISTENTE` (fechas vs. frecuencia explícita, de
  fase 3) antes de llegar a esta clasificación, así que una tercera
  categoría ahí habría sido inalcanzable en la práctica.
- **Tamaño de muestra para evaluación temporal:** se pasa
  `minimo_observaciones_entrenamiento = max(minimo_tecnico, 2*s)` a
  `evaluation.evaluar_holdout_temporal`, así una partición que deje el
  entrenamiento con menos de 2 ciclos completos omite la evaluación
  (`ejecutada: false`) sin bloquear el ajuste final, igual que cualquier
  otro caso de entrenamiento insuficiente ya manejado por ese módulo.

### 9. Decisiones estadísticas

- **Tendencia:** `resolver_tendencia(d, con_constante, D=D)` evalúa
  `d+D` (confirmado que es la restricción real de statsmodels, no solo
  `d`); con `con_constante=True` y `d+D>=2`, se resuelve a `trend="n"`
  automáticamente y sin error (verificado con test dedicado).
- **Diferenciación regular y estacional:** nunca se difieren los datos
  manualmente antes de pasarlos al modelo; siempre `order=(p,d,q)` +
  `seasonal_order=(P,D,Q,s)` sobre la serie original, delegando toda la
  diferenciación en statsmodels (evita doble diferenciación y
  desalineación de pronósticos).
- **ADF vs. diferenciación estacional:** se documenta explícitamente
  (`estacionariedad.estacional.advertencia` + advertencia
  `ADF_NO_DETERMINA_DIFERENCIACION_ESTACIONAL`, siempre presente) que el
  ADF regular no certifica si `D` es la elección correcta; no se
  implementaron OCSB/Canova-Hansen/HEGY (fuera de alcance, sin dependencias
  nuevas).
- **ACF/PACF estacionales:** `identificacion_estacional` evalúa
  únicamente los rezagos `s` y `2s` (vía `ejecutar_herramienta("acf",
  {"lags": 2*s})`) como observación informativa; nunca selecciona `P`/`Q`
  automáticamente.
- **`model_df`:** `p+q+P+Q` (no incluye `d`, `D` ni `s`), vía
  `diagnostics.construir_diagnostico_residuos(..., P=P, Q=Q)`.
- **Selección de lag de Ljung-Box:** se prefiere el rezago `s` si es válido
  (`model_df < s < cantidad_residuos` y `s` no consume más de la mitad de
  los residuos); si no, se recurre al criterio no estacional habitual; si
  ninguno es válido, no se ejecuta la prueba (nunca `NaN`).
- **Cantidad de ciclos:** ver sección 8; documentado que 2 ciclos es un
  piso orientativo, no una garantía.

### 10. Reutilización

| Responsabilidad | Módulo reutilizado | ¿Se tocó? |
|---|---|---|
| Ajuste | `forecasting/engine.py: ajustar_arima(seasonal_order=...)` | Sí, aditivo (ver sección 5) |
| Validación de serie/horizonte/confianza/orden base | `forecasting/validation.py` | Sí, aditivo (`resolver_tendencia` con `D`) |
| Diagnóstico de residuos / Ljung-Box / clasificación de parámetros | `forecasting/diagnostics.py` | Sí, aditivo (ver sección 5) |
| Métricas | `forecasting/metrics.py` | No |
| Evaluación temporal | `forecasting/evaluation.py` | No |
| Fechas/frecuencia | `forecasting/temporal.py` | No |
| Estacionariedad (ADF) | `tools/modelo_dickey_fuller.py` vía `ejecutar_herramienta` + `diagnostics.interpretar_resultado_adf` | No (ADF); compartida (interpretación) |
| Identificación (ACF) | `tools/acf.py` vía `ejecutar_herramienta` | No |
| Serialización JSON | `tools.py: _to_json_safe` | No |

### 11. Advertencias y errores

**Errores (nuevos):** `ORDEN_ESTACIONAL_INVALIDO`
(`InvalidSeasonalOrderError`), `PERIODICIDAD_INVALIDA`
(`InvalidSeasonalPeriodError`). **Errores reutilizados sin cambios:**
`ORDEN_INVALIDA`, `MUESTRA_INSUFICIENTE`, `SERIE_INVALIDA`,
`HORIZONTE_INVALIDO`, `NIVEL_CONFIANZA_INVALIDO`, todos los de
fechas/frecuencia (fase 3), `CONFIGURACION_PRUEBA_INVALIDA`, y
`ERROR_AJUSTE`/`ERROR_NUMERICO`/`ERROR_PRONOSTICO_GENERACION` de
`engine.py` (cubren "AJUSTE_SARIMA_FALLIDO"/"PRONOSTICO_SARIMA_FALLIDO" de
la consigna sin crear códigos nuevos, mismo razonamiento que en fase 4 para
MA). No se creó `MUESTRA_INSUFICIENTE_SARIMA`,
`FRECUENCIA_ESTACIONAL_INCONSISTENTE` ni `TENDENCIA_SARIMA_INVALIDA` como
excepciones separadas: los problemas que cubrirían ya los representan
correctamente `InsufficientDataError`, `InconsistentFrequencyError` (fase
3) y la resolución automática de tendencia (sin error posible, siempre
converge a un `trend` válido), respectivamente.

**Advertencias (todas como `{codigo, mensaje, severidad}`):**
`CICLOS_ESTACIONALES_INSUFICIENTES` (`advertencia_alta`),
`CICLOS_ESTACIONALES_LIMITADOS`, `CONFIGURACION_ESTACIONAL_COMPLEJA`,
`FRECUENCIA_Y_PERIODICIDAD_INUSUALES` (`informacion`),
`PRUEBA_NO_CUBRE_CICLO_COMPLETO` (`informacion`, dentro de
`evaluacion.advertencias`), `ADF_NO_DETERMINA_DIFERENCIACION_ESTACIONAL`,
`ADF_NO_EJECUTABLE` (reutilizada de MA). La convergencia
(`CONVERGENCIA_NO_ALCANZADA`) y los parámetros iniciales
estacionales-no-invertibles/no-estacionarios **reutilizan los códigos
genéricos ya existentes** de `diagnostics.clasificar_advertencia`: se
verificó empíricamente que los mensajes de advertencia de statsmodels para
componentes estacionales ("Non-stationary starting autoregressive
parameters...", "Non-invertible starting MA parameters...") son
textualmente idénticos a los del componente regular — statsmodels no
distingue cuál componente disparó la advertencia en el texto del mensaje.
Crear códigos `PARAMETROS_ESTACIONALES_NO_INVERTIBLES`/
`PARAMETROS_ESTACIONALES_NO_ESTACIONARIOS` habría exigido inventar una
distinción que la fuente de datos no ofrece; se prefirió no fabricar una
precisión diagnóstica que no existe. Ninguna advertencia se silencia.

### 12. Pruebas agregadas

**`test_modelo_sarima.py`** (95 tests) organizados en los grupos de la
consigna: contrato (7), órdenes (14: todos en cero salvo estacional, P=1,
Q=1, D=1, negativo, decimal, booleano, excesivo, s en 1/0/negativo/decimal/
booleano, configuración demasiado compleja), serie mensual s=12 (10), serie
trimestral s=4 (6), serie diaria s=7 (4), diferenciación estacional (6:
D=0, D=1, d=1&D=1, tendencia incompatible se resuelve sin error,
información de diferenciación, ausencia de doble diferenciación manual),
ciclos (7: menos de uno, entre uno y dos, exactamente dos, más de tres,
rechazo técnico, `n_ciclos_aproximados` correcto), coeficientes (8: AR/MA
regular y estacional, clasificación, p-valores, IC, significancia),
Ljung-Box (7: `model_df=p+q+P+Q`, lag>model_df, uso de rezago estacional
cuando es viable, lag alternativo cuando `s` es demasiado grande, muestra
pequeña sin NaN, interpretación prudente, serializable), evaluación
temporal (11: desactivada, activada, holdout cronológico, ciclos
suficientes/insuficientes, prueba menor a un ciclo, MAE/RMSE/MAPE,
reajuste final, pronóstico futuro separado, fechas de prueba alineadas),
fechas (9: mensuales, duplicadas, desordenadas, períodos faltantes,
frecuencia explícita compatible/incompatible/no inferible, sin fechas),
advertencias (5: ciclos insuficientes, configuración compleja, frecuencia/s
inusuales, ADF no determina D, advertencias no silenciadas) y regresión (3:
`modelo_arima` y `modelo_ma` siguen funcionando, las tres conviven en
`TOOL_REGISTRY`).

Dos fallos detectados y corregidos durante el desarrollo (ninguno en código
de producción): (1) un test de "frecuencia no inferible" construía fechas
que quedaban desordenadas por un error de años base en la construcción del
fixture (disparaba `FECHAS_DESORDENADAS` en vez de ejercitar la inferencia)
— corregido ajustando las fechas del fixture para mantener orden
cronológico; (2) un test de "lag alternativo cuando `s` es demasiado
grande" asumía que `s=12` sería inválido con `n=30`, pero el cálculo
(`cantidad_residuos//2=15 >= 12`) mostró que sí es válido — corregido
reduciendo la serie a `n=20` para que el rezago estacional realmente supere
la mitad de los residuos disponibles.

Series estacionales sintéticas construidas manualmente (tendencia +
componente sinusoidal de período `s` + ruido) con
`numpy.random.default_rng(seed)` fijo, sin dependencia nueva.

### 13. Comandos ejecutados

```
python manage.py test                                              → (linea base) Ran 301 tests — OK
python manage.py test apps.herramientas.test_modelo_sarima -v 2     → 1 error + 1 fallo (ver seccion 12), corregidos, luego Ran 95 tests — OK
python manage.py check                                              → 2 issues (warnings preexistentes de allauth, no relacionados)
python -m compileall -f apps/herramientas                            → todos los .py compilan sin errores, incluye tools/modelo_sarima.py
python manage.py test apps.herramientas -v 1                         → Ran 383 tests — OK
python manage.py test                                                → Ran 396 tests — OK (proyecto completo)
```

**Smoke tests manuales adicionales** (antes de la suite formal): SARIMA
mensual (1,1,1)(1,1,1,12) completo con coeficientes/estacionariedad/
invertibilidad/identificación/Ljung-Box coincidiendo con el ejemplo de la
consigna (`lags:12`, `incluye_rezago_estacional:true`); batería de
validaciones inválidas (`s` en 0/1/-3/2.5/True, `P`/`D`/`Q` inválidos/
excesivos, `p` excesivo); serie trimestral con fechas reales y frecuencia
inferida `QS-OCT`; serie diaria con fechas y ciclo semanal; fechas
duplicadas/desordenadas/frecuencia incompatible; regresión de
`modelo_arima`/`modelo_ma`.

### 14. Resultado completo de las pruebas

```
Antes de esta fase (linea base): 301 tests — 301 exitosas, 0 fallos, 0 errores, 0 omitidas — 7.36s
apps.herramientas.test_modelo_sarima (aislado): 95 tests — 95 exitosas, 0 fallos, 0 errores, 0 omitidas — 8.20s
apps.herramientas (fases 1-5 completas): 383 tests — 383 exitosas, 0 fallos, 0 errores, 0 omitidas — 11.23s
proyecto completo (todas las apps): 396 tests — 396 exitosas, 0 fallos, 0 errores, 0 omitidas — 15.44s
```

### 15. Compatibilidad

`modelo_arima` no se tocó. `modelo_ma` tuvo un refactor interno (extracción
de `_evaluar_estacionariedad` hacia `diagnostics.interpretar_resultado_adf`)
sin cambio de comportamiento observable, verificado con sus 82 tests
propios corriendo sin cambios. Los cambios al núcleo compartido
(`schemas.py`, `validation.py`, `diagnostics.py`, `engine.py`) son todos
aditivos con parámetros opcionales de default neutro. Se verificó
explícitamente con `RegresionARIMAMATests` (`test_modelo_arima_sigue_funcionando`,
`test_modelo_ma_sigue_funcionando`, `test_las_tres_herramientas_conviven_en_el_registro`)
y con la suite completa de fases 1-4 (301 tests) corriendo sin cambios
dentro de la suite final de 396.

### 16. Limitaciones

- **Pocos ciclos:** con menos de 2 ciclos se advierte (`advertencia_alta`)
  pero no se bloquea si el motor puede ajustar; con 2-3 ciclos la
  estimación sigue siendo estadísticamente débil aunque ya no dispare la
  advertencia más fuerte.
- **Modelos complejos:** el límite `CONFIGURACION_ESTACIONAL_COMPLEJA` es
  una heurística (`n < parametros*5`), no una prueba formal de
  identificabilidad; configuraciones cerca del límite pueden converger mal
  igual (las advertencias de convergencia de statsmodels seguirían
  apareciendo, no se silencian).
- **ADF regular vs. diferenciación estacional:** se documenta la limitación
  explícitamente (sección 9), pero no hay ninguna prueba estadística
  formal de diferenciación estacional implementada (OCSB, Canova-Hansen,
  HEGY quedan fuera de alcance).
- **Una única periodicidad estacional:** como en cualquier SARIMA clásico,
  solo se admite un `s`; series con múltiple estacionalidad (p. ej. diaria
  + semanal + anual simultáneas) no están cubiertas.
- **Ausencia de selección automática robusta:** `identificacion_estacional`
  y `q_sugerido_acf` (ya existente en MA) son solo orientativos; no hay
  búsqueda combinatoria de órdenes (deliberadamente fuera de alcance, y
  explícitamente prohibida por la consigna).
- **Advertencias estacionales no distinguibles del componente regular:**
  ver sección 11 — statsmodels no lo permite vía el texto del mensaje.

### 17. Preparación para la fase 6

El núcleo está preparado para incorporar ARIMAX (`exog=variables_exogenas_historicas`
para el ajuste, `exog=variables_exogenas_futuras` para el pronóstico) sin
duplicar:

- **Ajuste:** `ARIMA`/`SARIMAX` ya aceptan `exog` de forma nativa (confirmado
  en la inspección de fase 1). `engine.ajustar_arima` necesitaría un
  parámetro opcional `exog: np.ndarray | None = None` pasado al
  constructor, y `get_forecast(steps=..., exog=exog_futuro)` para el
  pronóstico — misma estrategia de "parámetro opcional con default neutro"
  usada en esta fase para `seasonal_order`.
- **Métricas/Evaluación:** sin cambios; `evaluation.evaluar_holdout_temporal`
  ya es agnóstica y solo necesitaría que la `funcion_pronostico` de la
  fachada ARIMAX alinee las exógenas de entrenamiento/prueba (lógica de la
  fachada, no del núcleo).
- **Fechas:** sin cambios.
- **Diagnóstico:** `construir_diagnostico_residuos` ya acepta `P`/`Q`
  explícitos; para ARIMAX no cambia `model_df` (las exógenas no son
  "autorregresivas" ni "de medias móviles" a efectos de Ljung-Box, así que
  seguiría siendo `p+q(+P+Q)`).
- **Intervalos:** sin cambios.

No se implementó ARIMAX todavía, tal como pedía la consigna.

---

## Fase 6 — Herramienta pública ARIMAX (variables exógenas)

**Alcance de la fase:** nueva herramienta pública `modelo_arimax`, más una
ampliación puntual y compatible del núcleo compartido (`exog`/`exog_futuro`
opcionales en el motor) y un módulo nuevo `forecasting/exogenous.py`
compartido para validación estructural, multicolinealidad y controles
básicos de fuga de información. No se implementaron componentes
estacionales dentro de ARIMAX ni la herramienta pública SARIMAX.

### 1. Diagnóstico previo

Releí `informe.md` (fases 1-5) y reverifiqué contra el código real. Línea
base ejecutada: `python manage.py test` → **396 tests, 0 fallos, 0 errores,
0 omitidas, OK (16.36s)**. Confirmé en el propio núcleo:

- El motor (`engine.ajustar_arima`) recibía la serie objetivo como `serie:
  np.ndarray` y no aceptaba variables exógenas.
- Seguía usando `statsmodels.tsa.arima.model.ARIMA` (subclase de `SARIMAX`,
  confirmado en fase 5).
- Las fechas se representan con `pandas.DatetimeIndex` (`temporal.py`); el
  holdout (`evaluation.evaluar_holdout_temporal`) divide `entrenamiento =
  serie[:n]` / `prueba = serie[n:]` cronológicamente, nunca al azar.
- Los coeficientes se arman genéricamente en `engine._construir_parametros`
  a partir de `ajuste.param_names`/`.params`/`.bse`/`.pvalues`/`.tvalues`/
  `.conf_int()`, y se clasifican con `diagnostics.clasificar_parametro`.
- Verifiqué empíricamente (ver sección 3) que `ARIMA(exog=...)` acepta un
  `pandas.DataFrame` con columnas nombradas y preserva esos nombres en
  `param_names` (p. ej. `"temperatura"`, `"promocion"`), en vez de los
  genéricos `"x1"/"x2"` que usa si se le pasa un array plano sin nombres.
- Las herramientas se siguen registrando por descubrimiento de archivos en
  `apps/herramientas/tools.py` (`glob("*.py")` no recursivo), sin registro
  manual.

### 2. Arquitectura utilizada

Se amplió `engine.ajustar_arima` con dos parámetros opcionales: `exog:
np.ndarray | pd.DataFrame | None = None` (pasado directo al constructor de
`ARIMA`) y `exog_futuro` (usado únicamente en
`ajuste.get_forecast(steps=..., exog=exog_futuro)`, nunca en el ajuste).
Ambos con default `None`, sin cambiar el comportamiento de ARIMA/MA/SARIMA
que no los usan.

Se creó `apps/herramientas/forecasting/exogenous.py`, un módulo nuevo y
compartido (no depende de ARIMA/SARIMA, solo de estructuras tabulares) con:
validación estructural de exógenas históricas/futuras y su conversión a
`pandas.DataFrame`; detección de columnas constantes y duplicadas;
diagnóstico de multicolinealidad (correlación, rango, número de condición,
escalas); y controles básicos de fuga de información. Se diseñó así
(separado de `tools/modelo_arimax.py`) precisamente para que una futura
`modelo_sarimax.py` pueda reutilizarlo sin duplicar nada.

`apps/herramientas/tools/modelo_arimax.py` es la fachada: valida sus propios
parámetros exógenos, alinea fechas de exógenas con el objetivo, construye
`seasonal_order` no aplica aquí (ARIMAX no lo usa), llama a
`engine.ajustar_arima(..., exog=..., exog_futuro=...)`, reutiliza
`evaluation.evaluar_holdout_temporal` (alineando las exógenas de
entrenamiento/prueba por fuera, ver sección 10), `temporal.*` para fechas y
`diagnostics.construir_diagnostico_residuos`/`interpretar_resultado_adf`
para diagnóstico y estacionariedad. Clasifica los coeficientes exógenos
comparando el nombre del parámetro contra los nombres de columnas conocidos
(no se tocó `diagnostics.clasificar_parametro`: los nombres de variables
exógenas son arbitrarios y ya caen en `"otro"` por defecto; la fachada los
reetiqueta a `"exogena"` al serializar, sin necesidad de que el núcleo
compartido conozca nombres específicos de columnas).

**Bug real encontrado y corregido durante el desarrollo:** la primera
versión de la fachada calculaba `diagnostico_multicolinealidad` pero nunca
verificaba su campo `clasificacion == "matriz_degenerada"` antes de
intentar el ajuste. Con una columna exógena constante y `con_constante=True`
(caso de prueba manual antes de escribir la suite formal), la matriz de
diseño quedaba con rango deficiente (la columna constante es linealmente
dependiente de la columna de unos del intercepto) y el ajuste fallaba con
una excepción cruda de statsmodels, capturada solo por el `except Exception`
genérico (`ERROR_INESPERADO`) en vez de un error de dominio claro. Se
corrigió agregando la verificación explícita que lanza
`ExogenousSingularMatrixError` (`EXOGENAS_MATRIZ_DEGENERADA`) antes de
llegar al ajuste, y se agregaron pruebas dedicadas
(`VariableConstanteTests.test_constante_junto_con_con_constante_true_es_degenerada`)
para que no vuelva a pasar inadvertido.

### 3. Clase de statsmodels utilizada

**`statsmodels.tsa.arima.model.ARIMA`** (la misma de fases 2-5), no se
instanció `SARIMAX` por separado. Confirmado empíricamente:

```python
>>> from statsmodels.tsa.arima.model import ARIMA
>>> import numpy as np, pandas as pd
>>> exog_df = pd.DataFrame({'temperatura': [...], 'promocion': [...]})
>>> m = ARIMA(list(y), exog=exog_df, order=(1,0,1), trend='c')
>>> m.fit().param_names
['const', 'temperatura', 'promocion', 'ar.L1', 'ma.L1', 'sigma2']
```

`ARIMA` acepta `exog` de forma nativa y, pasándole un `DataFrame` con
columnas nombradas, preserva esos nombres en la salida (en vez de
`"x1"/"x2"` genéricos con un array plano) — exactamente lo que necesita
`coeficientes_exogenos` para mostrar nombres reales. Usar `ARIMA` en vez de
`SARIMAX` directo evita una segunda función de ajuste con su propia lógica
de tendencia/advertencias/forecast (duplicación explícitamente prohibida) y
mantiene una única ruta de código para ARIMA/MA/SARIMA/ARIMAX.

### 4. Archivos creados

```
apps/herramientas/forecasting/exogenous.py
apps/herramientas/tools/modelo_arimax.py
apps/herramientas/test_modelo_arimax.py
```

### 5. Archivos modificados

- `apps/herramientas/forecasting/exceptions.py` — se agregaron 9 excepciones
  nuevas para variables exógenas (`ExogenousRequiredError`,
  `FutureExogenousRequiredError`, `ExogenousValueError`,
  `ExogenousNonFiniteError`, `ExogenousLengthMismatchError`,
  `ExogenousColumnMismatchError`, `ExogenousDuplicateError`,
  `ExogenousSingularMatrixError`, `ExogenousDateMismatchError`,
  `ForecastHorizonExogenousMismatchError`), todas heredando directo de
  `ForecastingError` (no forman una jerarquía natural entre sí, a diferencia
  de las de orden en fases 4-5). Ninguna excepción existente se modificó.
- `apps/herramientas/forecasting/engine.py` — `ajustar_arima` gana los
  parámetros opcionales `exog` y `exog_futuro` (ver sección 2); import
  nuevo de `pandas` (ya es dependencia del proyecto, usada en `temporal.py`
  desde fase 3) solo para el type hint.

Todos los cambios al núcleo son aditivos con defaults neutros; se verificó
con la suite completa de fases 1-5 (396 tests) corriendo sin cambios dentro
de la suite final de 497.

### 6. Contrato de entrada

**Obligatorios:** `valores, variables_exogenas_historicas, p, d, q`.
**Condicionalmente obligatorio:** `variables_exogenas_futuras` (requerido
siempre que se vaya a pronosticar; dado que `pasos_pronostico` ya exige
mínimo 1 desde fase 2, en la práctica es efectivamente obligatorio salvo
que se agregue en el futuro la posibilidad de horizonte 0). **Opcionales:**
`pasos_pronostico` (1), `con_constante` (true), `nivel_confianza` (0.95),
`fechas`, `fechas_exogenas_historicas`, `fechas_exogenas_futuras`,
`frecuencia`, `evaluar_modelo` (false), `cantidad_prueba`,
`porcentaje_prueba`.

### 7. Formato de exógenas

**Estructura pública:** diccionario por columnas
(`{"nombre": [valores...]}`) tanto para históricas como futuras — el único
formato público, sin una segunda representación equivalente (matriz +
nombres separados), tal como pedía la consigna, por las mismas razones ya
documentadas en fase 1 (preserva nombres, reduce errores de orden, es claro
para function calling).

**Transformación interna:** `exogenous.construir_exogenas_historicas`/
`construir_exogenas_futuras` convierten el diccionario a un
`pandas.DataFrame`, nunca expuesto directamente al usuario.

**Orden:** se conserva el orden de inserción del diccionario histórico
(`nombres = list(variables_exogenas_historicas.keys())`); las futuras se
**reordenan** para coincidir exactamente con ese orden histórico
(verificado con `test_orden_interno_corregido_de_forma_segura`, que manda
las futuras en un orden distinto y confirma que el resultado no cambia);
columnas faltantes o adicionales en las futuras son error
(`EXOGENAS_COLUMNAS_INCOMPATIBLES`), nunca se completan ni se ignoran en
silencio. La respuesta expone el orden final en
`variables_exogenas.nombres`.

**Alineación temporal:** sin `fechas_exogenas_historicas`, se asume
alineación posicional (documentado en
`informacion_temporal.alineacion_exogenas` y, si hay `fechas` para el
objetivo, con una advertencia informativa `ALINEACION_EXOGENAS_POR_POSICION`).
Con `fechas_exogenas_historicas`, deben coincidir exactamente (mismo orden,
mismos valores) con las fechas del objetivo, si no
`ExogenousDateMismatchError` (`FECHAS_EXOGENAS_DESALINEADAS`). Con
`fechas_exogenas_futuras`, deben coincidir exactamente con las fechas de
pronóstico generadas a partir de la frecuencia detectada/solicitada, mismo
código de error si no.

### 8. Contrato de salida

Claves principales: `modelo` (`"ARIMAX(p,d,q)"`), `representacion_interna`,
`orden`, `variables_exogenas` (`nombres`/`cantidad`), `n_observaciones`,
`coeficientes`, `coeficientes_exogenos`, `detalle_coeficientes`,
`estacionariedad`, `diagnostico_multicolinealidad`,
`diagnostico_fuga_informacion`, `aic`, `bic`, `mse_residuos`,
`mse_residuos_entrenamiento`, `diagnostico_residuos` (con `ljung_box`),
`evaluacion` (con `tipo: "condicionada_a_exogenas_observadas"` cuando se
ejecuta), `informacion_temporal` (con `alineacion_exogenas` anidado),
`pasos_pronostico`, `pronostico`, `fechas_pronostico`,
`intervalos_pronostico`, `nivel_confianza`, `tendencia_statsmodels`,
`descripcion_tendencia`, `informacion_ajuste`, `explicacion_modelo`
(`descripcion`/`causalidad`), `advertencias`. Ejemplo real (recortado):

```json
{
  "modelo": "ARIMAX(1,0,1)",
  "variables_exogenas": {"nombres": ["temperatura", "promocion"], "cantidad": 2},
  "coeficientes_exogenos": {"temperatura": 1.497322, "promocion": 7.947914},
  "diagnostico_multicolinealidad": {"rango_completo": true, "numero_condicion": 150.47, "clasificacion": "alta"},
  "diagnostico_fuga_informacion": {"variables_identicas_al_objetivo": [], "nombres_sospechosos": []},
  "evaluacion": {"ejecutada": true, "tipo": "condicionada_a_exogenas_observadas", "metricas_prueba": {"mae": 1.01, "rmse": 1.08, "mape": 1.17}},
  "explicacion_modelo": {"causalidad": "Un coeficiente exogeno significativo indica asociacion predictiva... No demuestra causalidad por si solo."},
  "advertencias": [{"codigo": "NUMERO_CONDICION_ALTO", "severidad": "advertencia"}, {"codigo": "ASOCIACION_NO_IMPLICA_CAUSALIDAD", "severidad": "informacion"}]
}
```

### 9. Validaciones

- **Longitudes:** cada columna histórica debe tener la misma longitud entre
  sí y contra `valores` (`EXOGENA_LONGITUD_INCOMPATIBLE`); cada columna
  futura debe tener exactamente `pasos_pronostico` valores
  (`HORIZONTE_EXOGENAS_INCOMPATIBLE` si no).
- **Columnas:** futuras deben tener exactamente las mismas variables que las
  históricas, ni de más ni de menos (`EXOGENAS_COLUMNAS_INCOMPATIBLES`).
- **Tipos:** rechaza no numéricos, booleanos disfrazados de número y
  estructuras anidadas (`EXOGENA_NO_NUMERICA`).
- **Finitud:** rechaza NaN/Inf en cualquier columna, histórica o futura
  (`EXOGENA_NO_FINITA`).
- **Horizonte:** ver longitudes.
- **Fechas:** ver sección 7.
- **Constantes:** `exogenous.detectar_columnas_constantes` → advertencia
  `EXOGENA_CONSTANTE` (no bloquea, salvo que además cause matriz degenerada
  junto con la constante del modelo, ver sección 2 sobre el bug corregido).
- **Duplicados:** `exogenous.detectar_columnas_duplicadas` → error duro
  `EXOGENAS_DUPLICADAS` (nunca se elige una silenciosamente).

### 10. Evaluación temporal

- **División conjunta:** `evaluation.evaluar_holdout_temporal` sigue siendo
  agnóstica al modelo (fase 3-4) y no conoce exógenas. Para alinear la
  partición de las exógenas con la del objetivo, la fachada llama
  **primero** a `evaluation.determinar_tamano_prueba(serie.size,
  cantidad_prueba, porcentaje_prueba)` — la misma función pura que
  `evaluar_holdout_temporal` invoca internamente — para conocer de antemano
  `n_entrenamiento` y recortar `df_historico` en `exog_entrenamiento`/
  `exog_prueba` exactamente en el mismo punto de corte. Al ser una función
  determinística llamada con los mismos argumentos, el resultado coincide
  siempre con el que usa `evaluar_holdout_temporal` internamente: **no es
  duplicar la lógica de división, es invocar la misma función pura dos
  veces** para poder alinear una segunda estructura (las exógenas) que el
  módulo de evaluación, deliberadamente, no conoce.
- **Uso de exógenas de prueba:** la función de pronóstico que recibe
  `evaluar_holdout_temporal` ajusta con `exog=exog_entrenamiento` y
  pronostica con `exog_futuro=exog_prueba` (los valores **reales** del
  tramo de prueba, nunca inventados).
- **Reajuste:** después de evaluar, el ajuste final siempre usa
  `exog=df_historico` completo (todas las observaciones) y
  `exog_futuro=df_futuro` (las exógenas futuras genuinas) — nunca se
  mezclan las exógenas de prueba con las futuras del pronóstico final
  (verificado con `test_entrenamiento_sin_datos_futuros_del_objetivo`, que
  compara el pronóstico final con y sin evaluación: da igual).
- **Interpretación condicionada:** cuando `evaluacion.ejecutada`, se agrega
  `"tipo": "condicionada_a_exogenas_observadas"` y una advertencia
  informativa `EVALUACION_CON_EXOGENAS_OBSERVADAS` aclarando que la
  precisión medida asume que las exógenas de prueba son conocidas, no
  necesariamente la precisión cuando también haya que pronosticar esas
  exógenas.

### 11. Multicolinealidad

- **Métodos:** matriz de correlación por pares (`DataFrame.corr()`), rango
  de la matriz de diseño (`numpy.linalg.matrix_rank`, incluyendo la columna
  de unos si `con_constante=True`) y número de condición vía SVD
  (`numpy.linalg.svd`, cociente entre el mayor y el menor valor singular).
  No se usó VIF de statsmodels (`variance_inflation_factor`) para no atar
  el diagnóstico a una función que itera columna por columna con su propio
  supuesto de intercepto; la combinación correlación+rango+número de
  condición ya cubre los mismos casos (par correlacionado, colinealidad
  múltiple, degeneración exacta) sin ambigüedad de interpretación.
- **Umbrales documentados** (constantes en `exogenous.py`):
  `|correlación| >= 0.95` → se reporta el par y advertencia
  `EXOGENAS_ALTAMENTE_CORRELACIONADAS`; número de condición `> 30` →
  advertencia `NUMERO_CONDICION_ALTO`; `> 100` o matriz sin rango completo →
  clasificación `"alta"`/`"matriz_degenerada"`; razón entre desvíos
  estándar `>= 100x` → advertencia `ESCALAS_EXOGENAS_MUY_DIFERENTES` (no se
  reescala nada automáticamente).
- **Limitaciones:** es una heurística basada en umbrales fijos, no un
  análisis formal de identificabilidad; el número de condición está
  influido por la escala de las variables (de ahí la advertencia aparte); no
  sustituye un VIF exacto por variable.

### 12. Fuga de información

- **Controles implementados:** variable histórica idéntica al objetivo
  (`np.allclose`, tolerancia `1e-9`) → `POSIBLE_FUGA_INFORMACION`
  (`advertencia_alta`); correlación `>= 0.98` con el objetivo (sin ser
  idéntica) → `EXOGENA_CORRELACION_CASI_PERFECTA_OBJETIVO` (`advertencia`,
  nunca una afirmación absoluta); nombres sospechosos (`objetivo_futuro`,
  `venta_futura`, `demanda_futura`, `target`, `y_futuro`) →
  `POSIBLE_FUGA_INFORMACION` (`advertencia`, nunca rechazo solo por el
  nombre).
- **Controles no posibles:** no hay forma de verificar automáticamente que
  una variable estuviera *realmente* disponible en el momento histórico de
  cada observación (eso depende de cómo el usuario recolectó los datos, no
  de los valores en sí); por eso `diagnostico_fuga_informacion` siempre
  incluye `nota_disponibilidad_temporal` como recordatorio explícito, y
  ninguna función de este módulo afirma haber detectado fuga con certeza.
- **Causalidad:** `explicacion_modelo.causalidad` y la advertencia
  `ASOCIACION_NO_IMPLICA_CAUSALIDAD` (siempre presente, severidad
  `"informacion"`) dejan explícito que un coeficiente significativo es
  asociación predictiva dentro del modelo, no un efecto causal.

### 13. Decisiones estadísticas

- **Tendencia:** reutiliza `validation.resolver_tendencia(d, con_constante)`
  sin cambios (ARIMAX no tiene componente estacional, `D=0` implícito).
- **Estacionariedad:** se reutiliza `diagnostics.interpretar_resultado_adf`
  sobre la serie objetivo (mismo mecanismo que MA/SARIMA, vía
  `ejecutar_herramienta("modelo_dickey_fuller", ...)`); si no hay evidencia
  de estacionariedad, se agrega `POSIBLE_REGRESION_ESPURIA` (advertencia,
  no bloqueante) explicando que una regresión dinámica sobre series con
  tendencia puede producir relaciones espurias. No se exige que cada
  variable exógena sea estacionaria por separado (tal como indicaba la
  consigna) ni se implementaron pruebas de cointegración.
- **Coeficientes exógenos:** clasificados como `"exogena"` por la fachada
  (comparando el nombre del parámetro contra los nombres de columna
  conocidos), con el mismo detalle estadístico completo que cualquier otro
  parámetro (coeficiente, error estándar, estadístico t, p-valor, IC 95%,
  significancia al 5%).
- **`model_df`:** `p + q` (sin incluir las exógenas), vía
  `diagnostics.construir_diagnostico_residuos(residuos, d, p, q)` sin
  parámetros adicionales — Ljung-Box evalúa autocorrelación residual, no
  significancia de la regresión exógena.
- **Ljung-Box:** reutilizado sin cambios; la interpretación nunca afirma
  "modelo válido" únicamente por esta prueba (mismo texto prudente de
  fases 2-5).
- **Intervalos:** genéricos vía `get_forecast(steps=..., exog=exog_futuro)`,
  sin cambios respecto de ARIMA/MA/SARIMA salvo el parámetro `exog` extra.

### 14. Reutilización

| Responsabilidad | Módulo reutilizado | ¿Se tocó? |
|---|---|---|
| Ajuste | `forecasting/engine.py: ajustar_arima(exog=..., exog_futuro=...)` | Sí, aditivo |
| Validación de serie/horizonte/confianza/orden | `forecasting/validation.py` | No |
| Validación/transformación de exógenas | `forecasting/exogenous.py` (nuevo) | — |
| Métricas | `forecasting/metrics.py` | No |
| Evaluación temporal | `forecasting/evaluation.py` (+ `determinar_tamano_prueba` invocada dos veces, ver sección 10) | No |
| Fechas/frecuencia | `forecasting/temporal.py` | No |
| Diagnóstico de residuos / Ljung-Box / clasificación de parámetros | `forecasting/diagnostics.py` | No |
| Estacionariedad (ADF) | `tools/modelo_dickey_fuller.py` vía `ejecutar_herramienta` + `diagnostics.interpretar_resultado_adf` | No |
| Serialización JSON | `tools.py: _to_json_safe` | No |

### 15. Advertencias y errores

**Errores (nuevos, ver sección 5):** `EXOGENAS_HISTORICAS_REQUERIDAS`,
`EXOGENAS_FUTURAS_REQUERIDAS`, `EXOGENA_NO_NUMERICA`, `EXOGENA_NO_FINITA`,
`EXOGENA_LONGITUD_INCOMPATIBLE`, `EXOGENAS_COLUMNAS_INCOMPATIBLES`,
`EXOGENAS_DUPLICADAS`, `EXOGENAS_MATRIZ_DEGENERADA`,
`FECHAS_EXOGENAS_DESALINEADAS`, `HORIZONTE_EXOGENAS_INCOMPATIBLE`.
**Reutilizados sin cambios:** `ORDEN_INVALIDA`, `MUESTRA_INSUFICIENTE`,
`SERIE_INVALIDA`, `HORIZONTE_INVALIDO`, `NIVEL_CONFIANZA_INVALIDO`, todos
los de fechas/frecuencia de fase 3, `CONFIGURACION_PRUEBA_INVALIDA`, y
`ERROR_AJUSTE`/`ERROR_NUMERICO`/`ERROR_PRONOSTICO_GENERACION` de
`engine.py` (cubren "AJUSTE_ARIMAX_FALLIDO"/"PRONOSTICO_ARIMAX_FALLIDO" sin
crear códigos nuevos, mismo razonamiento que en fases 4-5).

**Advertencias (todas `{codigo, mensaje, severidad}`):**
`EXOGENA_CONSTANTE`, `EXOGENAS_ALTAMENTE_CORRELACIONADAS`,
`NUMERO_CONDICION_ALTO`, `ESCALAS_EXOGENAS_MUY_DIFERENTES`,
`POSIBLE_FUGA_INFORMACION`, `EXOGENA_CORRELACION_CASI_PERFECTA_OBJETIVO`,
`EVALUACION_CON_EXOGENAS_OBSERVADAS` (dentro de `evaluacion.advertencias`),
`ASOCIACION_NO_IMPLICA_CAUSALIDAD` (siempre presente, `"informacion"`),
`ALINEACION_EXOGENAS_POR_POSICION` (`"informacion"`),
`POSIBLE_REGRESION_ESPURIA` (código adicional, no listado literalmente en
la consigna pero justificado por la sección de estacionariedad),
`ADF_NO_EJECUTABLE` (reutilizada). Ninguna se silencia.

### 16. Pruebas agregadas

**`test_modelo_arimax.py`** (101 tests): contrato (7), estructurales
históricas (13: una/dos variables válidas, diccionario vacío, columna
vacía, longitudes históricas distintas, longitud incompatible con
objetivo, nombre vacío, valor no numérico/booleano/NaN/Inf, estructura
anidada), exógenas futuras (10: estructura válida, falta completa/una
columna, columna adicional, longitud menor/mayor al horizonte, NaN/Inf,
orden interno corregido, pronóstico sin inventar valores), variable
relevante (7: ajuste, coeficiente presente, signo razonable, pronóstico e
intervalos finitos, evaluación disponible, serializable), variable binaria
(4), múltiples variables (5: nombres preservados, orden estable,
coeficientes, detalle estadístico, multicolinealidad), variable constante
(4, incluyendo la regresión del bug real encontrado), duplicación (2),
multicolinealidad (5: independientes, correlación >0.95, número de
condición alto, matriz sin rango completo, escalas muy diferentes), fuga de
información (6: idéntica, casi idéntica, nombre sospechoso, variable
legítima no rechazada, advertencia sin afirmación absoluta, nota de
disponibilidad temporal), evaluación temporal (9), fechas (11), coeficientes
(7), diagnóstico (6), explicación/causalidad (2) y regresión (4: ARIMA, MA
y SARIMA siguen funcionando, las cuatro herramientas conviven en
`TOOL_REGISTRY`).

Series sintéticas construidas manualmente (tendencia + `beta * exógena` +
ruido, variantes continua/binaria/multivariable) con
`numpy.random.default_rng(seed)` fijo, sin dependencia nueva.

### 17. Comandos ejecutados

```
python manage.py test                                          → (linea base) Ran 396 tests — OK
python manage.py test apps.herramientas.test_modelo_arimax -v 2 → Ran 101 tests — OK (0 fallos, 0 errores, en el primer intento tras corregir el bug de la seccion 2)
python manage.py check                                          → 2 issues (warnings preexistentes de allauth, no relacionados)
python -m compileall -f apps/herramientas                        → todos los .py compilan sin errores, incluye exogenous.py y tools/modelo_arimax.py
python manage.py test apps.herramientas -v 1                     → Ran 484 tests — OK
python manage.py test                                            → Ran 497 tests — OK (proyecto completo)
```

**Smoke tests manuales adicionales** (antes de la suite formal, donde se
encontró el bug de la sección 2): ARIMAX(1,0,1) con dos exógenas
(temperatura continua, promoción binaria) end-to-end; batería de
validaciones estructurales inválidas (vacío, longitudes, tipos, NaN/Inf,
anidado); exógenas futuras faltantes/incompletas/con columna extra;
columnas duplicadas y constantes; fuga por variable idéntica y nombre
sospechoso; evaluación con exógenas reales del tramo de prueba; fechas de
exógenas históricas alineadas/desalineadas y futuras compatibles/incompatibles.

### 18. Resultado completo de las pruebas

```
Antes de esta fase (linea base): 396 tests — 396 exitosas, 0 fallos, 0 errores, 0 omitidas — 16.36s
apps.herramientas.test_modelo_arimax (aislado): 101 tests — 101 exitosas, 0 fallos, 0 errores, 0 omitidas — 4.20s
apps.herramientas (fases 1-6 completas): 484 tests — 484 exitosas, 0 fallos, 0 errores, 0 omitidas — 15.54s
proyecto completo (todas las apps): 497 tests — 497 exitosas, 0 fallos, 0 errores, 0 omitidas — 19.71s
```

### 19. Compatibilidad

`modelo_arima`, `modelo_ma` y `modelo_sarima` no se tocaron. Los cambios al
núcleo (`exceptions.py`, `engine.py`) son aditivos con parámetros/clases
opcionales de default neutro. Verificado explícitamente con
`RegresionARIMAMASARIMATests` (las tres herramientas anteriores siguen
funcionando, las cuatro conviven en `TOOL_REGISTRY`) y con la suite
completa de fases 1-5 (396 tests) corriendo sin cambios dentro de la suite
final de 497.

### 20. Limitaciones

- **Necesidad de exógenas futuras:** ARIMAX no puede pronosticar sin ellas;
  no se inventan, extrapolan ni repiten valores (deliberado, por diseño).
- **Multicolinealidad:** heurística basada en umbrales fijos (sección 11),
  no un análisis formal; sensible a la escala de las variables.
- **Causalidad:** los coeficientes son asociación predictiva dentro del
  modelo, nunca efectos causales confirmados.
- **Fuga no detectable en general:** solo se cubren los casos más obvios
  (identidad exacta, correlación casi perfecta, nombre sugestivo); una
  variable con fuga sutil (p. ej. calculada con información parcial del
  futuro de forma no obvia) no se detecta.
- **Regresiones espurias:** advertida cuando el objetivo no muestra
  evidencia de estacionariedad, pero no se verifica automáticamente si las
  propias exógenas también tienen tendencia (lo que agravaría el riesgo).
- **Incertidumbre de las exógenas futuras:** los intervalos de predicción
  de ARIMAX asumen que los valores futuros de las exógenas son exactos; no
  propagan la incertidumbre de esas exógenas (que en la práctica suelen ser
  ellas mismas una proyección) hacia el intervalo del objetivo.

### 21. Preparación para la fase 7

El núcleo está preparado para combinar `order=(p,d,q)` +
`seasonal_order=(P,D,Q,s)` + `exog=variables_exogenas` (SARIMAX completo)
sin duplicar nada:

- **Ajuste:** `engine.ajustar_arima` ya acepta `seasonal_order` (fase 5) y
  `exog`/`exog_futuro` (esta fase) simultáneamente como parámetros
  independientes — nada impide pasar ambos a la vez en una misma llamada
  (no se probó explícitamente en esta fase por estar fuera de alcance, pero
  la implementación no tiene ninguna exclusión mutua entre ambos).
- **Métricas/Evaluación:** sin cambios; la fachada SARIMAX combinaría el
  patrón de alineación de exógenas de esta fase con el análisis de ciclos
  estacionales de fase 5.
- **Fechas:** sin cambios.
- **Diagnóstico:** `construir_diagnostico_residuos` ya acepta `P`/`Q`
  (fase 5) y el descarte inicial ya generalizado (`descarte_inicial=d+D*s`);
  SARIMAX solo necesitaría combinar ambos parámetros sets tal cual, sin
  modificar `diagnostics.py`.
- **Intervalos:** sin cambios.
- **Exógenas:** `forecasting/exogenous.py` ya es completamente agnóstico de
  si el modelo tiene o no componente estacional — reutilizable sin cambios.

No se implementó la herramienta pública SARIMAX todavía, tal como pedía la
consigna.

---

## Fase 7 — Herramienta pública SARIMAX general y consolidación del núcleo

**Alcance de la fase:** nueva herramienta pública `modelo_sarimax` que
combina `order`, `seasonal_order` y `exog` simultáneamente, más una
consolidación real (no cosmética) de lógica que estaba duplicada entre las
fachadas de fases 4-6. No se realizó todavía la auditoría final ni casos
límite exhaustivos (fase 8).

### 1. Diagnóstico previo

Releí `informe.md` (fases 1-6) y reverifiqué contra el código real. Línea
base ejecutada: `python manage.py test` → **497 tests, 0 fallos, 0 errores,
0 omitidas, OK (19.50s)**. La inspección confirmó algo que las fases
anteriores no habían resuelto del todo: **existía duplicación residual real
entre fachadas**, no solo teórica:

- `modelo_sarima.py` tenía, como funciones privadas propias, toda la
  validación/análisis estacional (`_validar_ordenes`, `_validar_periodicidad`,
  `_analizar_ciclos`, `_analizar_complejidad`,
  `_clasificar_coherencia_estacional`) — nada de esto vivía en un módulo
  compartido, así que una futura `modelo_sarimax.py` lo habría tenido que
  copiar.
- `modelo_ma.py`, `modelo_sarima.py` y `modelo_arimax.py` tenían, **cada
  una por separado**, el mismo patrón exacto: llamar a
  `ejecutar_herramienta("modelo_dickey_fuller", ...)`, interpretar con
  `diagnostics.interpretar_resultado_adf`, y agregar la advertencia
  `ADF_NO_EJECUTABLE` si no pudo ejecutarse — literalmente el mismo bloque
  de código en tres archivos.
- `modelo_arimax.py` tenía, como funciones privadas propias,
  `_resolver_alineacion_exogenas` y `_validar_fechas_exogenas_futuras` —
  lógica genuinamente reutilizable (no depende de ARIMA en sí) que
  `modelo_sarimax` iba a necesitar igual.

Confirmé además, empíricamente, que `engine.ajustar_arima` (ya extendido en
fases 5 y 6 con `seasonal_order` y `exog` por separado) acepta **ambos
simultáneamente sin ningún conflicto**:

```python
resultado = engine.ajustar_arima(
    serie=y, p=1, d=0, q=0, con_constante=True, pasos_pronostico=3,
    nivel_confianza=0.95, seasonal_order=(1,1,0,12), exog=exog_df, exog_futuro=exog_futuro_df,
)
# -> parametros: drift, temperatura, ar.L1, ar.S.L12, sigma2 -- todo correcto
```

Esto significó que **no hacía falta ninguna migración de clase de
statsmodels ni ninguna rama nueva en el motor**: el trabajo de esta fase es
100% de consolidación de fachadas + una fachada nueva que combina lo que
`engine.py` ya sabía hacer.

### 2. Arquitectura consolidada

Se extrajo la lógica duplicada identificada arriba hacia el núcleo
compartido, **antes** de escribir `modelo_sarimax.py`, y se refactorizaron
las fachadas existentes para usarla (verificando con su propia suite después
de cada cambio):

- **`forecasting/seasonal.py`** (nuevo): validación de P/D/Q/s, límites
  pedagógicos de p/q para fachadas con estacionalidad, análisis de ciclos y
  clasificación de coherencia frecuencia-periodicidad. Extraído de
  `modelo_sarima.py`.
- **`diagnostics.evaluar_estacionariedad_regular`** (nuevo, en
  `diagnostics.py`): consolida "llamar ADF + interpretar + advertir si no
  ejecutable", antes triplicado en MA/SARIMA/ARIMAX.
- **`exogenous.resolver_alineacion_exogenas` /
  `exogenous.validar_fechas_exogenas_futuras`** (nuevas, en `exogenous.py`):
  extraídas de `modelo_arimax.py`.
- **`validation.calcular_minimo_observaciones_general` /
  `validation.advertencia_configuracion_compleja`** (nuevas, en
  `validation.py`): generalizan el patrón "pérdidas por diferenciación +
  parámetros + margen" y la advertencia de "pocos datos por parámetro" que
  antes existían solo como código local de una fachada.
- **`modelo_sarima.py`, `modelo_ma.py`, `modelo_arimax.py`** (refactorizados):
  ahora llaman a las funciones de arriba en vez de mantener copias propias.
  Verificado que el comportamiento observable no cambió: los 82 tests de MA,
  los 95 de SARIMA y los 101 de ARIMAX siguen pasando exactamente igual
  después del refactor.
- **`modelo_sarimax.py`** (nuevo): fachada que reutiliza todo lo anterior,
  agrega la clasificación automática del tipo de modelo (`_clasificar_tipo_modelo`,
  `_nombre_modelo` — la única lógica genuinamente nueva y específica de esta
  herramienta, ya que ninguna otra fachada necesita "adivinar" qué es) y
  arma la respuesta combinando piezas.

### 3. Clase o clases de statsmodels utilizadas

**Una sola: `statsmodels.tsa.arima.model.ARIMA`** (la misma desde fase 2),
vía `engine.ajustar_arima`, ahora con `seasonal_order` y `exog` pasados
simultáneamente cuando corresponde. No se usó `SARIMAX` directamente en
ningún punto, ni se creó un adaptador que alterne entre clases. Justificación:

- `ARIMA` es subclase de `SARIMAX` (confirmado en fase 5) y ya acepta ambos
  parámetros de forma nativa y simultánea (confirmado empíricamente en esta
  fase, ver sección 1): no hay ninguna configuración de SARIMAX que
  `engine.ajustar_arima` no pudiera representar ya.
- Migrar a instanciar `SARIMAX` directamente no habría cambiado ningún
  resultado (es la misma clase base) pero sí habría obligado a mantener dos
  rutas de construcción de modelo, duplicando exactamente lo que esta fase
  buscaba eliminar.
- Se verificó con pruebas de regresión cruzadas (sección 15/18) que
  `modelo_sarimax` sin estacionalidad ni exógenas devuelve **el mismo AIC y
  el mismo pronóstico** que `modelo_arima` para la misma configuración —
  evidencia directa de que es literalmente el mismo motor.

### 4. Archivos creados

```
apps/herramientas/forecasting/seasonal.py
apps/herramientas/tools/modelo_sarimax.py
apps/herramientas/test_modelo_sarimax.py
```

### 5. Archivos modificados

- `apps/herramientas/forecasting/validation.py` — se agregaron
  `calcular_minimo_observaciones_general` y `advertencia_configuracion_compleja`
  (funciones nuevas, no se tocó ninguna existente).
- `apps/herramientas/forecasting/diagnostics.py` — se agregó
  `evaluar_estacionariedad_regular` (nueva; el resto del módulo no cambió).
- `apps/herramientas/forecasting/exogenous.py` — se agregaron
  `resolver_alineacion_exogenas` y `validar_fechas_exogenas_futuras`
  (extraídas de `modelo_arimax.py`), más el import de `temporal` que esas
  funciones necesitan.
- `apps/herramientas/forecasting/exceptions.py` — se agregó
  `SeasonalPeriodRequiredError` (`PERIODICIDAD_ESTACIONAL_REQUERIDA`),
  subclase de `InvalidSeasonalPeriodError`.
- `apps/herramientas/tools/modelo_sarima.py` — se eliminaron las funciones
  privadas que ahora viven en `seasonal.py`/`diagnostics.py`/`validation.py`,
  reemplazadas por llamadas a esos módulos. Mismo comportamiento observable
  (95 tests propios sin cambios).
- `apps/herramientas/tools/modelo_ma.py` — `_evaluar_estacionariedad`
  reescrita para delegar en `diagnostics.evaluar_estacionariedad_regular`.
  Mismo comportamiento observable (82 tests propios sin cambios).
- `apps/herramientas/tools/modelo_arimax.py` — se eliminaron
  `_resolver_alineacion_exogenas`, `_validar_fechas_exogenas_futuras` y la
  llamada directa a ADF dentro de `_construir_estacionariedad`, reemplazadas
  por `exogenous.resolver_alineacion_exogenas`,
  `exogenous.validar_fechas_exogenas_futuras` y
  `diagnostics.evaluar_estacionariedad_regular`. Mismo comportamiento
  observable (101 tests propios sin cambios).

No se tocó `engine.py`, `metrics.py`, `evaluation.py`, `temporal.py`,
`schemas.py`, `tools.py` (loader) ni `modelo_arima.py`/`modelo_ar.py`.

### 6. Contrato de entrada SARIMAX

**Obligatorios:** `valores, p, d, q`. **Opcionales con default:** `P=0,
D=0, Q=0, s=null, variables_exogenas_historicas=null,
variables_exogenas_futuras=null, pasos_pronostico=1, con_constante=true,
nivel_confianza=0.95, evaluar_modelo=false`, más `fechas`,
`fechas_exogenas_historicas`, `fechas_exogenas_futuras`, `frecuencia`,
`cantidad_prueba`, `porcentaje_prueba`. **Condicionales:** `s` es
obligatorio si `P>0 or D>0 or Q>0` (si no, error
`PERIODICIDAD_ESTACIONAL_REQUERIDA`); `variables_exogenas_futuras` es
obligatorio si se proveyeron `variables_exogenas_historicas` (error
`EXOGENAS_FUTURAS_REQUERIDAS`, reutilizado de fase 6).

### 7. Clasificación de modelos

`_clasificar_tipo_modelo(p, d, q, es_estacional, tiene_exogenas)` (en
`modelo_sarimax.py`, lógica nueva y específica de esta fachada — ninguna
otra necesita "adivinar" su propio tipo):

```
tiene_exogenas and es_estacional         -> SARIMAX
tiene_exogenas (sin estacionalidad)      -> ARIMAX
es_estacional (sin exogenas)             -> SARIMA
d > 0 (sin estacionalidad ni exogenas)   -> ARIMA
p > 0 and q > 0                          -> ARMA
q > 0                                    -> MA
p > 0                                    -> AR
(0,0,0) sin nada                         -> ARIMA(0,0,0) (caso degenerado)
```

`es_estacional = P>0 or D>0 or Q>0`; `tiene_exogenas = bool(variables_exogenas_historicas)`.
Verificado con 7 pruebas dedicadas (una por tipo) más una que confirma que
`representacion_interna` (que siempre menciona la clase real de
statsmodels) es distinta de `modelo`/`tipo_modelo_detectado` — nunca se usa
"SARIMAX" como nombre pedagógico solo porque la clase interna se llame así.

### 8. Contrato de salida

Superconjunto de las claves de ARIMA/SARIMA/ARIMAX: `modelo`,
`tipo_modelo_detectado`, `representacion_interna`, `orden`,
`orden_estacional` (`{P,D,Q,s}`, con `s=null` si no hay estacionalidad),
`variables_exogenas` (`{utilizadas,nombres,cantidad}`), `n_observaciones`,
`n_ciclos_aproximados` (`null` si no hay estacionalidad), `coeficientes`,
`coeficientes_regulares`, `coeficientes_estacionales`,
`coeficientes_exogenos` (siempre dicts, `{}` cuando la categoría no
existe), `detalle_coeficientes`, `estacionariedad` (`{regular, estacional}`,
`estacional=null` sin componente estacional), `coherencia_estacional`
(`null` sin estacionalidad), `diagnostico_multicolinealidad` /
`diagnostico_fuga_informacion` (**`null`**, no `{}`, cuando no hay
exógenas — ver decisión en sección 11), `aic`, `bic`, `mse_residuos`,
`mse_residuos_entrenamiento`, `diagnostico_residuos`, `evaluacion`,
`informacion_temporal`, `pasos_pronostico`, `pronostico`,
`fechas_pronostico`, `intervalos_pronostico`, `nivel_confianza`,
`tendencia_statsmodels`, `descripcion_tendencia`, `informacion_ajuste`,
`explicacion_modelo` (`{descripcion, causalidad}`, con `causalidad=null`
sin exógenas), `advertencias`. Ejemplo real (SARIMAX completo, recortado):

```json
{
  "modelo": "SARIMAX(1,0,0)(1,1,0,12)",
  "tipo_modelo_detectado": "SARIMAX",
  "orden_estacional": {"P": 1, "D": 1, "Q": 0, "s": 12},
  "variables_exogenas": {"utilizadas": true, "nombres": ["temperatura"], "cantidad": 1},
  "coeficientes_regulares": {"ar.L1": -0.1514},
  "coeficientes_estacionales": {"ar.S.L12": -0.4777},
  "coeficientes_exogenos": {"temperatura": 1.3361},
  "n_ciclos_aproximados": 5.0,
  "advertencias": [{"codigo": "ASOCIACION_NO_IMPLICA_CAUSALIDAD", "severidad": "informacion"}]
}
```

**Decisión sobre `null` vs. `{}`** (sección 21 de la consigna, "no devolver
`null` y `{}` de forma inconsistente"): se distinguen dos tipos de campo.
Los diccionarios de **categorización de coeficientes existentes**
(`coeficientes_regulares/estacionales/exogenos`) siempre son dicts, `{}`
cuando esa categoría no tiene parámetros — porque `coeficientes` (el dict
completo) siempre existe y estos son subconjuntos filtrados de él. Los
campos de **diagnóstico que implican una tarea que se ejecutó o no**
(`diagnostico_multicolinealidad`, `diagnostico_fuga_informacion`,
`estacionariedad.estacional`, `coherencia_estacional`,
`explicacion_modelo.causalidad`) son `null` cuando esa tarea no corresponde
(no hay exógenas / no hay estacionalidad) — para no simular con un dict
vacío un análisis que nunca se ejecutó.

### 9. Validaciones

Todas reutilizadas, ninguna copiada:

- **Longitudes/columnas/tipos/finitud de exógenas:** `exogenous.py` sin
  cambios de comportamiento (fase 6).
- **Órdenes regulares:** `validation.validar_orden_arima` (fase 2) +
  `seasonal.validar_techo_ordenes_regulares` (extraída esta fase).
- **Órdenes/periodicidad estacional:** `seasonal.validar_ordenes_estacionales` /
  `validar_periodicidad` (extraídas esta fase).
- **Complejidad general:** `validation.calcular_minimo_observaciones_general`
  (nueva, generaliza `d + D*s + parametros + margen`) para el error duro
  (`MUESTRA_INSUFICIENTE`); `validation.advertencia_configuracion_compleja`
  (nueva) para la advertencia no bloqueante (`MODELO_DEMASIADO_COMPLEJO`),
  contando `p+q+P+Q+n_exogenas+constante` como parámetros estimados.
- **Fechas/frecuencia/ciclos/coherencia:** `temporal.py` (fase 3) +
  `seasonal.py` (esta fase), sin cambios de comportamiento.

### 10. Evaluación temporal

Funciona en las cuatro combinaciones (verificado con
`EvaluacionSARIMAXTests`, 11 tests): sin estacionalidad ni exógenas, con
estacionalidad, con exógenas, con ambas. El patrón es idéntico al de
SARIMA (fase 5) combinado con el de ARIMAX (fase 6), sin ninguna lógica
nueva de división: `evaluation.evaluar_holdout_temporal` sigue siendo
agnóstica; cuando hay exógenas, se pre-calcula el punto de corte con
`evaluation.determinar_tamano_prueba` (misma función pura, invocada una
vez más para alinear el DataFrame de exógenas, igual que en fase 6) y se
recortan `exog_entrenamiento`/`exog_prueba`; cuando hay estacionalidad, el
mínimo de entrenamiento exigido es `max(minimo_tecnico, 2*s)` (igual que en
fase 5). `evaluacion.tipo = "condicionada_a_exogenas_observadas"` se agrega
solo si hay exógenas; `PRUEBA_NO_CUBRE_CICLO_COMPLETO` solo si hay
estacionalidad y `n_prueba < s`.

### 11. Multicolinealidad

Sin cambios respecto de fase 6 (`exogenous.diagnosticar_multicolinealidad`,
mismos umbrales: `|correlación|>=0.95`, número de condición `>30`
advertencia / `>100` clasificación "alta", matriz sin rango completo →
`EXOGENAS_MATRIZ_DEGENERADA`). Se agregó la verificación explícita de
`clasificacion == "matriz_degenerada"` antes de ajustar (el bug que se
había encontrado y corregido en fase 6 para ARIMAX; en `modelo_sarimax.py`
se implementó correctamente desde el principio, verificado con un test
dedicado).

### 12. Fuga de información

Sin cambios respecto de fase 6 (`exogenous.diagnosticar_fuga_informacion`).
`explicacion_causalidad` y la advertencia `ASOCIACION_NO_IMPLICA_CAUSALIDAD`
solo se agregan cuando hay exógenas (ver sección 8 sobre `null` vs. dato
real).

### 13. Decisiones estadísticas

- **Tendencia:** `validation.resolver_tendencia(d, con_constante, D=D)`
  reutilizada sin cambios (fase 5); con `D=0` (sin estacionalidad) se
  comporta exactamente como en ARIMA/MA/ARIMAX.
- **Complejidad:** ver sección 9.
- **Ciclos:** `seasonal.analizar_ciclos` reutilizada sin cambios; solo se
  invoca si `es_estacional` (nunca se exige periodicidad ni se advierte
  sobre ciclos cuando `P=D=Q=0`, verificado con test dedicado).
- **Estacionariedad/invertibilidad:** `enforce_stationarity`/
  `enforce_invertibility` expuestos via `informacion_ajuste` (sin cambios,
  siempre `True`, heredado de `engine.py`); `estacionariedad.regular` (ADF
  sobre la serie objetivo) nunca se presenta como certificación de que la
  serie completa —con componentes regulares y estacionales combinados— sea
  estacionaria: el disclaimer de fase 5 sobre "el ADF regular no determina
  la diferenciación estacional" se conserva integro cuando hay
  estacionalidad.
- **`model_df`:** `p+q+P+Q` (nunca incluye variables exógenas), vía
  `diagnostics.construir_diagnostico_residuos(..., P=P, Q=Q)` sin cambios;
  verificado con test dedicado que confirma que agregar una exógena no
  cambia `model_df`.
- **Selección de lag:** reutiliza `diagnostics.seleccionar_lag_ljung_box`
  (fase 5) con `periodo_estacional=s` solo si `es_estacional`.
- **Coeficientes exógenos / causalidad:** mismo criterio de fase 6
  (clasificación por nombre de columna conocido, nunca interpretados como
  causales).

### 14. Reutilización

| Responsabilidad | Módulo | Estado en esta fase |
|---|---|---|
| Ajuste (`order`+`seasonal_order`+`exog` simultáneos) | `engine.ajustar_arima` | Sin cambios (ya lo soportaba) |
| Validación de serie/horizonte/confianza/orden regular | `validation.py` | Ampliado (funciones nuevas, nada modificado) |
| Validación/análisis estacional | `seasonal.py` | **Nuevo**, extraído de `modelo_sarima.py` |
| Validación/diagnóstico de exógenas | `exogenous.py` | Ampliado (2 funciones extraídas de `modelo_arimax.py`) |
| Métricas | `metrics.py` | Sin cambios |
| Evaluación temporal | `evaluation.py` | Sin cambios |
| Fechas/frecuencia | `temporal.py` | Sin cambios |
| Diagnóstico de residuos/Ljung-Box/parámetros/ADF | `diagnostics.py` | Ampliado (1 función nueva) |
| Serialización JSON | `tools.py: _to_json_safe` | Sin cambios |

### 15. Advertencias y errores

**Nuevos:** `PERIODICIDAD_ESTACIONAL_REQUERIDA` (`SeasonalPeriodRequiredError`),
`MODELO_DEMASIADO_COMPLEJO` (advertencia, vía
`validation.advertencia_configuracion_compleja`). **Reutilizados sin
cambios:** todos los de ARIMA/MA (fase 2-4), fechas/evaluación (fase 3),
estacionales (fase 5: `CICLOS_ESTACIONALES_INSUFICIENTES/LIMITADOS`,
`FRECUENCIA_Y_PERIODICIDAD_INUSUALES`, `ADF_NO_DETERMINA_DIFERENCIACION_ESTACIONAL`,
`PRUEBA_NO_CUBRE_CICLO_COMPLETO`), exógenas (fase 6: `EXOGENA_CONSTANTE`,
`EXOGENAS_DUPLICADAS`, `EXOGENAS_MATRIZ_DEGENERADA`,
`EXOGENAS_ALTAMENTE_CORRELACIONADAS`, `NUMERO_CONDICION_ALTO`,
`ESCALAS_EXOGENAS_MUY_DIFERENTES`, `POSIBLE_FUGA_INFORMACION`,
`EXOGENA_CORRELACION_CASI_PERFECTA_OBJETIVO`,
`EVALUACION_CON_EXOGENAS_OBSERVADAS`, `ASOCIACION_NO_IMPLICA_CAUSALIDAD`,
`ALINEACION_EXOGENAS_POR_POSICION`, `EXOGENAS_FUTURAS_REQUERIDAS`).
No se creó `CONFIGURACION_SARIMAX_INVALIDA` (los casos que cubriría ya los
representan `ORDEN_INVALIDA`/`ORDEN_ESTACIONAL_INVALIDO`/`PERIODICIDAD_INVALIDA`),
ni `AJUSTE_SARIMAX_FALLIDO`/`PRONOSTICO_SARIMAX_FALLIDO` (reutilizan
`ERROR_AJUSTE`/`ERROR_PRONOSTICO_GENERACION` de `engine.py`, igual que en
fases 4-6).

### 16. Pruebas agregadas

**`test_modelo_sarimax.py`** (109 tests): contrato (7), clasificación (8: una
por tipo AR/MA/ARMA/ARIMA/SARIMA/ARIMAX/SARIMAX + representación interna
separada del nombre), sin estacionalidad ni exógenas (7, incluyendo
regresión con `modelo_arima` y coherencia de campos vacíos/`null`), con
estacionalidad sin exógenas (5, incluyendo regresión con `modelo_sarima`),
sin estacionalidad con exógenas (5, incluyendo regresión con
`modelo_arimax`), SARIMAX completo con serie estacional+exógena+fechas (17:
ajuste, órdenes, coeficientes por categoría, pronóstico, intervalos, fechas
futuras, evaluación, MAE/RMSE/MAPE, diagnóstico, multicolinealidad,
serialización), componentes opcionales (7: sin `s`, error si falta `s` con
estacionalidad, sin/con exógenas, falta de futuras, `seasonal_order` vacío
válido, exógenas vacías tratadas como ausentes), complejidad (6),
tendencia (6), parámetros (7), Ljung-Box (7, incluyendo confirmación de que
las exógenas no se cuentan en `model_df`), evaluación (11), fechas (9) y
regresión global (6: las cinco fachadas anteriores siguen funcionando, las
seis conviven en `TOOL_REGISTRY`).

Un fallo detectado y corregido durante el desarrollo: una prueba propia
(`test_configuracion_incompatible_se_resuelve_sin_error`) esperaba
`tendencia_statsmodels == "t"` para `d=1,D=1`, pero `d+D=2` correctamente
resuelve a `"n"` (coincide con `test_diferenciacion_regular_y_estacional`,
que sí lo tenía bien) — error de la aserción del test, no del código de
producción; corregido y reverificado.

Series sintéticas construidas manualmente (AR/MA/tendencia/estacional/exógena,
combinables) con `numpy.random.default_rng(seed)` fijo, sin dependencia
nueva.

### 17. Comandos ejecutados

```
python manage.py test                                            → (linea base) Ran 497 tests — OK
python manage.py test apps.herramientas.test_modelo_sarima -v 1   → tras refactor a seasonal.py: Ran 95 tests — OK
python manage.py test apps.herramientas.test_modelo_ma -v 1       → tras refactor a diagnostics compartido: Ran 82 tests — OK
python manage.py test apps.herramientas.test_modelo_arimax -v 1   → tras refactor a exogenous/diagnostics compartidos: Ran 101 tests — OK
python manage.py test                                             → tras consolidacion, antes de SARIMAX: Ran 497 tests — OK
python manage.py test apps.herramientas.test_modelo_sarimax -v 2  → 1 fallo (aserción propia, ver seccion 16), corregido, luego Ran 109 tests — OK
python manage.py check                                             → 2 issues (warnings preexistentes de allauth, no relacionados)
python -m compileall -f apps/herramientas                           → todos los .py compilan sin errores, incluye seasonal.py y tools/modelo_sarimax.py
python manage.py test apps.herramientas -v 1                        → Ran 593 tests — OK
python manage.py test                                                → Ran 606 tests — OK (proyecto completo)
```

**Smoke tests manuales adicionales:** clasificación de las 7 configuraciones
(AR/MA/ARMA/ARIMA/SARIMA/ARIMAX/SARIMAX) sobre series sintéticas propias;
SARIMAX completo (estacional+exógena+fechas) con inspección de
coeficientes por categoría; comparación numérica cruzada
`modelo_ma`↔`modelo_sarimax(p=0,d=0,q=2)`,
`modelo_arima`↔`modelo_sarimax` (mismo AIC exacto y mismo pronóstico),
`modelo_sarima`↔`modelo_sarimax` sin exógenas (mismo pronóstico exacto).

### 18. Resultado completo de las pruebas

```
Antes de esta fase (linea base): 497 tests — 497 exitosas, 0 fallos, 0 errores, 0 omitidas — 19.50s
apps.herramientas.test_modelo_sarimax (aislado): 109 tests — 109 exitosas, 0 fallos, 0 errores, 0 omitidas — 8.46s
apps.herramientas (fases 1-7 completas): 593 tests — 593 exitosas, 0 fallos, 0 errores, 0 omitidas — 23.50s
proyecto completo (todas las apps): 606 tests — 606 exitosas, 0 fallos, 0 errores, 0 omitidas — 27.90s
```

### 19. Diferencias numéricas

**Ninguna esperada ni observada.** `modelo_sarimax` sin estacionalidad ni
exógenas usa exactamente `engine.ajustar_arima(seasonal_order=None,
exog=None)`, la misma llamada que `modelo_arima`; se verificó
empíricamente que el AIC coincide con precisión exacta (`131.223303` en
ambos, mismo test) y que los pronósticos son idénticos valor a valor. Lo
mismo se verificó para `modelo_ma`↔`modelo_sarimax(p=0,d=0,q=2)` (mismo
coeficiente `ma.L1`) y `modelo_sarima`↔`modelo_sarimax` sin exógenas
(mismos pronósticos). Esto es evidencia directa —no solo argumental— de que
no hay dos rutas de ajuste divergentes.

### 20. Limitaciones

- **Pocos ciclos:** igual que fase 5 (advertencia, no bloqueo, salvo que
  además viole el mínimo técnico).
- **Necesidad de exógenas futuras:** igual que fase 6, sin cambios.
- **Incertidumbre de las propias exógenas:** los intervalos no propagan la
  incertidumbre de haber pronosticado las exógenas futuras por fuera.
- **Multicolinealidad:** heurística de umbrales fijos, no un análisis
  formal (igual que fase 6).
- **Causalidad:** los coeficientes exógenos siguen siendo asociación
  predictiva, nunca causalidad.
- **Fuga de información:** controles básicos, no exhaustivos (igual que
  fase 6); ahora combinados con estacionalidad no cambian esta limitación.
- **Una única periodicidad:** como en SARIMA, no hay soporte para múltiple
  estacionalidad simultánea.
- **Complejidad del modelo:** `MODELO_DEMASIADO_COMPLEJO` es una heurística
  (observaciones por parámetro), no una prueba formal de identificabilidad;
  configuraciones cerca del límite pueden converger mal igual (las
  advertencias de convergencia de statsmodels seguirían apareciendo, no se
  silencian).
- **Consolidación parcial:** `modelo_arimax.py` conserva su propio
  `_minimo_tecnico` local (una instancia particular de la fórmula general
  ahora en `validation.calcular_minimo_observaciones_general`); no se
  retrofiteó para llamar a la función general, para no tocar más archivos
  de los estrictamente necesarios en una fase que explícitamente pedía "no
  realizar una refactorización cosmética masiva". Queda como candidato para
  la auditoría final (fase 8).

### 21. Preparación para la fase 8

El proyecto está preparado para la auditoría final:

- **Casos límite:** el patrón de pruebas ya cubre series vacías/cortas/
  constantes, NaN/Inf, fechas irregulares/duplicadas/desordenadas, y
  configuraciones degeneradas en las seis herramientas; una auditoría
  puede enfocarse en combinaciones cruzadas entre ellas (p. ej. SARIMAX con
  todas las condiciones límite simultáneas) más que en descubrir casos
  nuevos.
- **Prueba académica:** sigue sin existir en el repositorio una serie fija
  del caso académico de 24 meses (documentado desde fase 2); todas las
  fases usaron series sintéticas documentadas como tales.
- **Documentación:** este `informe.md` acumula el diagnóstico y las
  decisiones de las 7 fases; no hay documentación de usuario final (README)
  para las herramientas nuevas, que podría ser parte de la fase 8.
- **Revisión de compatibilidad:** las seis fachadas (`modelo_ar` no tocada
  en ninguna fase, `modelo_ma`, `modelo_arima`, `modelo_sarima`,
  `modelo_arimax`, `modelo_sarimax`) conviven en `TOOL_REGISTRY` y sus
  contratos públicos no cambiaron; queda pendiente de fase 8 decidir si
  `modelo_arimax._minimo_tecnico` se unifica con la función general (ver
  sección 20) y si conviene agregar renderers dedicados en `home.html` para
  las herramientas de fases 4-7 (siguen usando el fallback JSON genérico
  desde que se crearon).

No se realizó la auditoría final ni una refactorización estética general en
esta fase, tal como pedía la consigna.

---

## Fase 8 — Auditoría integral, robustez y documentación final

### 1. Resumen ejecutivo

La auditoría final encontró y corrigió tres defectos de alcance real:
`modelo_ar` todavía mantenía una ruta estadística independiente; la
serialización común permitía `NaN`/`Inf` nativos; y el diagnóstico residual
no declaraba ni controlaba residuos no finitos. También se eliminó
duplicación del serializador de parámetros en cinco fachadas y la fórmula
local de muestra mínima de ARIMAX.

El sistema final mantiene AR, MA, ARIMA, SARIMA, ARIMAX y SARIMAX,
registro dinámico, contratos históricos y un único motor statsmodels. La
suite final ejecutó **614 pruebas: 614 exitosas, 0 fallos, 0 errores, 0
omitidas, 27.745s**.

### 2. Estado inicial

`python manage.py test -v 1` encontró 606 pruebas y ejecutó 606/606
correctamente en 35.639s. `python manage.py check` devolvió código 0 con
dos advertencias preexistentes de configuración de django-allauth.

`python -m unittest` no es un runner válido por sí solo en este proyecto:
falló con 96 errores de inicialización porque no configura
`DJANGO_SETTINGS_MODULE` ni el registro de aplicaciones. El runner oficial
confirmado es `python manage.py test`; el fallo de unittest es preexistente
y no corresponde al motor de pronóstico.

### 3. Arquitectura final

- Fachadas: `apps/herramientas/tools/modelo_{ar,ma,arima,sarima,arimax,sarimax}.py`.
- Motor: `forecasting/engine.py`, una ruta basada en
  `statsmodels.tsa.arima.model.ARIMA`.
- Validación: `validation.py` + `seasonal.py`.
- Métricas/evaluación: `metrics.py` + `evaluation.py`.
- Fechas: `temporal.py`.
- Exógenas: `exogenous.py`.
- Diagnóstico: `diagnostics.py`.
- Esquemas/excepciones: `schemas.py` + `exceptions.py`.
- Serialización de parámetros: `serialization.py`.
- Serialización final JSON-safe y registro: `apps/herramientas/tools.py`.

Las fachadas adaptan nombres y contratos; no construyen una segunda clase de
statsmodels ni recalculan métricas, intervalos o Ljung-Box.

### 4. Árbol de archivos relevantes

```text
apps/herramientas/
├── forecasting/
│   ├── diagnostics.py
│   ├── engine.py
│   ├── evaluation.py
│   ├── exceptions.py
│   ├── exogenous.py
│   ├── metrics.py
│   ├── schemas.py
│   ├── seasonal.py
│   ├── serialization.py
│   ├── temporal.py
│   └── validation.py
├── tools/
│   ├── modelo_ar.py
│   ├── modelo_ma.py
│   ├── modelo_arima.py
│   ├── modelo_sarima.py
│   ├── modelo_arimax.py
│   └── modelo_sarimax.py
├── test_forecasting_auditoria.py
└── test_*.py
docs/
├── chatbot/consigna.txt
└── forecasting.md
README.md
```

No existe un directorio raíz `tests/`; el proyecto organiza las pruebas
dentro de las apps Django.

### 5. Archivos creados

- `forecasting/serialization.py`: serialización única del detalle de
  parámetros.
- `test_forecasting_auditoria.py`: robustez, AR compartido, JSON estricto,
  carga dinámica y volatilidad.
- `README.md`: entrada técnica y comandos válidos.
- `docs/forecasting.md`: arquitectura, contratos, ejemplos, diagnóstico,
  compatibilidad y limitaciones.

### 6. Archivos modificados

- `tools.py`: normalización de `NaN`/`Inf` a `null`.
- `forecasting/diagnostics.py`: finitud residual explícita y exclusión
  documentada de residuos no finitos.
- `tools/modelo_ar.py`: migrado al motor/validación/diagnóstico compartidos,
  preservando claves históricas.
- `tools/modelo_arima.py`, `modelo_ma.py`, `modelo_sarima.py`,
  `modelo_arimax.py`, `modelo_sarimax.py`: eliminación del serializador
  duplicado.
- `tools/modelo_arimax.py`: muestra mínima delegada a la fórmula general.
- `informe.md`: este cierre de fase.

### 7. Problemas encontrados

**Errores:** AR podía aceptar argumentos inválidos, lanzar errores crudos y
declarar “modelo válido” sólo por Ljung-Box; JSON-safe no neutralizaba
no-finitos; diagnóstico residual podía producir NaN.

**Limitaciones:** alta volatilidad no está modelada como varianza
condicional; una sola periodicidad; incertidumbre exógena no propagada.

**Duplicación:** serialización de parámetros repetida cinco veces y fórmula
de mínimo ARIMAX local.

**Inconsistencias:** AR no exponía intervalos, significancia, convergencia
ni `model_df`, a diferencia del resto del motor.

**Pruebas frágiles:** no se detectó una regresión activa; las comparaciones
exactas existentes se limitan a rutas que usan literalmente el mismo motor.

**Documentación faltante:** no existían README ni guía de usuario.

### 8. Correcciones realizadas

1. AR ahora configura `engine.ajustar_arima(p=p,d=0,q=0)` y usa validación
   de dominio.
2. Su Ljung-Box usa `model_df=p` e interpretación prudente.
3. AR conserva sus claves y agrega intervalos/detalle/advertencias.
4. `_to_json_safe` convierte cualquier float no finito a `None`.
5. El diagnóstico informa `residuos_finitos`,
   `residuos_no_finitos_excluidos` y cantidad original.
6. Ljung-Box recibe sólo residuos finitos; la exclusión genera advertencia
   alta y no queda oculta.
7. Las cinco fachadas comparten `serializar_parametro`.
8. ARIMAX usa `calcular_minimo_observaciones_general`.
9. Se agregaron pruebas globales `allow_nan=False`.
10. Se documentó el sistema y el runner oficial.

### 9. Contratos finales

- **AR:** obligatorios `valores,p`; opcionales horizonte/confianza; preserva
  `orden_p` y claves históricas.
- **MA:** obligatorios `valores,q`; fachada ARIMA(0,0,q).
- **ARIMA:** obligatorios `valores,p,d,q`; fechas/evaluación opcionales.
- **SARIMA:** agrega `P,D,Q,s`; sin exógenas.
- **ARIMAX:** exige exógenas históricas y futuras para pronosticar.
- **SARIMAX:** `valores,p,d,q` obligatorios; estacionalidad y exógenas
  opcionales/condicionales.

### 10. Compatibilidad

| Herramienta | Argumentos anteriores | Argumentos actuales | Claves preservadas | Claves nuevas | Incompatibilidad | Migración |
|---|---|---|---|---|---|---|
| AR | valores,p,horizonte | agrega confianza opcional | todas | detalle, intervalos, ajuste | ninguna | no |
| MA | sin cambios | sin cambios | todas | ya aditivas | ninguna | no |
| ARIMA | sin cambios | sin cambios | todas | ya aditivas | ninguna | no |
| SARIMA | sin cambios | sin cambios | todas | ya aditivas | ninguna | no |
| ARIMAX | sin cambios | sin cambios | todas | ya aditivas | ninguna | no |
| SARIMAX | fase 7 | sin cambios | todas | ninguna en fase 8 | ninguna | no |

`mse_residuos` permanece como alias de
`mse_residuos_entrenamiento`.

### 11. Decisiones estadísticas

- Tendencia central según `d+D`.
- ADF se interpreta como evidencia frente a raíz unitaria, nunca certeza.
- `enforce_stationarity`/`enforce_invertibility` son restricciones del
  estimador, no pruebas sobre la serie original.
- Diferenciación queda dentro de statsmodels; no se duplica manualmente.
- `model_df`: AR=`p`, MA=`q`, ARIMA/ARIMAX=`p+q`,
  SARIMA/SARIMAX=`p+q+P+Q`; exógenas excluidas.
- Lag: válido sólo si `lag>model_df` y `lag<n_residuos`; prefiere `s` si es
  viable.
- MAE/RMSE/MAPE se calculan únicamente sobre holdout.
- Intervalos provienen de `get_forecast().conf_int()`.
- Exógenas futuras nunca se inventan.
- Coeficientes exógenos son asociación, no causalidad.
- Multicolinealidad exacta bloquea; correlación alta no degenerada advierte.

### 12. Casos límite cubiertos

Vacío, muestras de 1/2/6 observaciones, constante/casi constante,
texto/booleanos/NaN/±Inf, ceros/negativos/cercanos a cero, escalas grandes,
outliers, cambios de nivel, tendencia, ruido y estacionalidad fuerte,
varianza cambiante, órdenes negativos/decimales/booleanos/excesivos,
periodicidad inválida/faltante, fechas inválidas/nulas/duplicadas/
desordenadas/incompatibles, frecuencias comunes y no inferibles, exógenas
vacías/no numéricas/no finitas/desalineadas/duplicadas/constantes/
degeneradas, multicolinealidad y fuga básica.

La fase agregó cinco patrones explícitos de alta volatilidad/outliers; todos
producen ajuste o error de dominio serializable, nunca traceback.

### 13. Caso académico

Se buscaron archivos estructurados, documentación, pruebas, código e
informes. **La serie académica original de 24 meses no existe en el
repositorio.** No se inventaron valores. Se conserva la regresión sintética
de 24 observaciones ya documentada desde fase 2, que verifica ajuste
ARIMA(0,1,0), pronóstico, intervalos, diagnóstico, compatibilidad y JSON,
pero no se presenta como el caso académico real.

### 14. Pruebas agregadas o modificadas

`test_forecasting_auditoria.py` agrega 8 pruebas:

- contrato y núcleo común de AR;
- equivalencia AR↔SARIMAX;
- invalidación backend de AR;
- residuos no finitos;
- normalización JSON de NaN/Inf;
- cinco series volátiles/outliers;
- registro sin colisiones;
- `json.dumps(..., allow_nan=False)` para las 12 herramientas.

Las 606 pruebas anteriores no se modificaron.

### 15. Comandos ejecutados

```text
python manage.py check                                      -> 0, 2 warnings allauth
python manage.py test -v 1                                  -> 0, línea base 606/606
python -m unittest                                          -> 1, 96 errores de bootstrap Django (preexistente/no oficial)
python -m compileall apps/herramientas                       -> 0
python manage.py test apps.herramientas.test_forecasting_auditoria -v 2 -> 0, 8/8
python manage.py test apps.herramientas -v 1                 -> 0, 601/601
python manage.py test <núcleo/diagnóstico> -v 0              -> 0, 57/57
python manage.py test <métricas> -v 0                        -> 0, 33/33
python manage.py test <evaluación> -v 0                      -> 0, 21/21
python manage.py test <fechas> -v 0                          -> 0, 29/29
python manage.py test <AR> -v 0                              -> 0, 3/3
python manage.py test <MA> -v 0                              -> 0, 82/82
python manage.py test <ARIMA> -v 0                           -> 0, 63/63
python manage.py test <SARIMA> -v 0                          -> 0, 95/95
python manage.py test <ARIMAX> -v 0                          -> 0, 101/101
python manage.py test <SARIMAX> -v 0                         -> 0, 109/109
python manage.py test <integración+chatbot> -v 0             -> 0, 10/10
python manage.py test -v 1                                  -> 0, final 614/614
python -m compileall -f apps/herramientas                    -> 0
```

### 16. Resultado final de pruebas

```text
cantidad total: 614
exitosas:       614
fallos:         0
errores:        0
omitidas:       0
duración:       27.745s
```

### 17. Diferencias numéricas

AR ahora usa exactamente el mismo motor que SARIMAX sin componentes
estacionales/exógenos; una prueba confirma AIC y pronóstico idénticos para
AR(1). No se observaron cambios numéricos en MA/ARIMA/SARIMA/ARIMAX/SARIMAX:
el refactor de serialización no altera el ajuste.

### 18. Documentación actualizada

- `README.md`: arquitectura breve, enlace y comandos.
- `docs/forecasting.md`: modelos, parámetros, fechas, exógenas, holdout,
  métricas, diagnóstico, ejemplos, compatibilidad, limitaciones y futuro.
- `informe.md`: cierre completo de fase 8.

### 19. Dependencias

No se agregó ni eliminó ninguna dependencia. Python 3.11 y las versiones
fijadas de numpy/pandas/statsmodels/Django siguen siendo suficientes. No se
detectó una dependencia nueva incorporada sólo para las pruebas de
forecasting.

### 20. Riesgos pendientes

Faltan los datos reales del caso académico; alta volatilidad requiere un
modelo de varianza si se desea inferencia específica; las exógenas futuras
son conocidas sin incertidumbre; multicolinealidad/fuga son heurísticas;
sólo hay una periodicidad; las muestras cercanas al mínimo pueden converger
mal; continúan dos warnings deprecados de allauth fuera de alcance.

### 21. Deuda técnica

Las fachadas todavía repiten parte del armado de respuestas y orquestación
de evaluación; extraerlo completamente produciría una abstracción grande y
menos pedagógica, por lo que no se hizo sin una necesidad funcional.
`python -m unittest` directo continúa no soportado: debe usarse el runner
Django. El frontend mantiene fallback JSON para herramientas sin renderer
dedicado.

### 22. Trabajo futuro recomendado

1. Incorporar la serie académica real como fixture versionado.
2. Rolling-origin y monitoreo de degradación.
3. Selección automática acotada/comparación homogénea de modelos.
4. Pruebas específicas de raíz estacional.
5. Escenarios e incertidumbre de exógenas.
6. Visualización de intervalos.
7. ARIMA-GARCH para varianza condicional.
8. Múltiples estacionalidades.

### 23. Veredicto final

**LISTO CON LIMITACIONES.**

La implementación está estable, compatible, documentada y pasa 614 pruebas.
La limitación que impide declarar un “listo” sin reservas es externa al
código: no está disponible la serie real del caso académico de 24 meses,
por lo que no puede existir una regresión auténtica sobre esos valores sin
inventarlos.
