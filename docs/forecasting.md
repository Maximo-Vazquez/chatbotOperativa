# Motor de pronóstico

## Arquitectura

Las fachadas públicas viven en `apps/herramientas/tools/` y exponen
`TOOL_DEFINITION`, `TOOL_META` y `TOOL_FUNCTION`. El cargador
`apps/herramientas/tools.py` las registra por descubrimiento de archivos.

El núcleo se divide por responsabilidad:

| Módulo | Responsabilidad |
|---|---|
| `forecasting/engine.py` | Construcción y ajuste con statsmodels, advertencias, parámetros, pronóstico e intervalos |
| `forecasting/validation.py` | Serie, órdenes regulares, muestra, horizonte, confianza, complejidad y tendencia |
| `forecasting/seasonal.py` | Órdenes estacionales, periodicidad, ciclos y coherencia con frecuencia |
| `forecasting/exogenous.py` | Exógenas históricas/futuras, alineación, multicolinealidad y fuga básica |
| `forecasting/metrics.py` | MAE, RMSE y MAPE |
| `forecasting/evaluation.py` | Holdout cronológico fuera de muestra |
| `forecasting/temporal.py` | Fechas, frecuencia, períodos faltantes y fechas futuras |
| `forecasting/diagnostics.py` | Residuos, ADF, Ljung-Box y clasificación de parámetros/advertencias |
| `forecasting/schemas.py` | Objetos normalizados del ajuste |
| `forecasting/serialization.py` | Estructura pública común de parámetros |
| `forecasting/exceptions.py` | Errores de dominio |

El motor instancia `statsmodels.tsa.arima.model.ARIMA`. Esa clase hereda de
SARIMAX y admite `order`, `seasonal_order` y `exog` simultáneamente. Mantener
una sola ruta evita diferencias accidentales entre fachadas.

## Modelos y parámetros

| Modelo | Configuración |
|---|---|
| AR(p) | `order=(p,0,0)` |
| MA(q) | `order=(0,0,q)` |
| ARMA(p,q) | `order=(p,0,q)` |
| ARIMA(p,d,q) | Orden regular, sin estacionalidad ni exógenas |
| SARIMA | `order=(p,d,q)` + `seasonal_order=(P,D,Q,s)` |
| ARIMAX | Orden regular + variables exógenas |
| SARIMAX | Orden regular + estacionalidad y exógenas opcionales |

- `p`: orden autorregresivo regular.
- `d`: diferenciación regular.
- `q`: orden de medias móviles regular.
- `P`, `D`, `Q`: órdenes estacionales.
- `s`: longitud del ciclo; es obligatorio si `P`, `D` o `Q` son positivos.
- `pasos_pronostico`: horizonte, entre 1 y 50.
- `con_constante`: solicita un término determinista compatible.
- `nivel_confianza`: nivel de los intervalos, entre 0,80 y 0,999.

La tendencia se resuelve centralmente: `d+D=0` permite constante (`c`),
`d+D=1` usa tendencia lineal (`t`) como drift del modelo integrado y
`d+D>=2` no admite término determinista (`n`). Si `con_constante=false`, se
usa `n`.

### MA(q) no es un promedio móvil

Un modelo MA(q) explica el valor actual mediante errores aleatorios pasados.
Un promedio móvil de observaciones es una técnica de suavizado. Comparten un
nombre histórico, pero no son equivalentes.

## Fechas y frecuencia

`fechas` es opcional y debe tener la misma longitud que `valores`. Se
recomienda ISO 8601 (`2026-07-01`). Las fechas deben ser interpretables,
únicas y estrictamente crecientes; el backend no reordena.

La frecuencia puede declararse (`diaria`, `semanal`, `mensual`,
`trimestral`, `anual`, `horaria` o alias compatibles de pandas) o inferirse.
Los períodos faltantes se informan, pero nunca se completan. Si la
frecuencia no puede inferirse, queda `null`, se emite una advertencia y no se
inventan fechas futuras. Las zonas horarias consistentes son preservadas por
pandas.

## Variables exógenas

El formato público es un diccionario por columnas:

```json
{
  "temperatura": [18.2, 19.1, 20.0],
  "promocion": [0, 1, 0]
}
```

Las columnas históricas deben coincidir en longitud con el objetivo. Para
pronosticar se requieren las mismas columnas futuras y exactamente
`pasos_pronostico` valores. No se extrapolan ni repiten valores.

Si se proporcionan fechas de exógenas, deben alinearse con el objetivo y con
las fechas futuras. Sin fechas se declara alineación posicional.

El diagnóstico informa columnas constantes o duplicadas, rango, correlación,
número de condición y diferencias de escala. Una matriz degenerada o
columnas exactamente duplicadas son errores; una correlación alta que aún
permita estimación genera advertencia.

Los controles de fuga detectan identidad o correlación casi perfecta con el
objetivo y nombres sospechosos. No pueden demostrar que una variable estuvo
disponible históricamente. **Cada variable exógena debe estar disponible en
el momento real del pronóstico.**

Un coeficiente significativo indica asociación predictiva condicionada al
modelo. No demuestra causalidad por sí solo.

## Evaluación temporal y métricas

Con `evaluar_modelo=true`, el último tramo se reserva como prueba. La
partición nunca es aleatoria:

1. se ajusta sólo con entrenamiento;
2. se pronostica el tramo final;
3. se calculan MAE, RMSE y MAPE sobre prueba;
4. se reajusta con todos los datos;
5. se produce el pronóstico futuro.

Las exógenas se cortan en la misma posición. La evaluación con exógenas es
`condicionada_a_exogenas_observadas`: no incorpora la incertidumbre de
pronosticarlas.

- MAE: error absoluto medio.
- RMSE: penaliza más los errores grandes.
- MAPE: porcentaje absoluto medio.

MAPE excluye objetivos exactamente cero e informa cuántos fueron excluidos.
Si todos son cero devuelve `null`. Valores cercanos a cero y valores
negativos generan advertencias de interpretación; nunca se devuelve
infinito.

`mse_residuos` es un alias compatible de
`mse_residuos_entrenamiento`. Ambos describen ajuste in-sample y no deben
interpretarse como precisión futura.

## Diagnóstico

Cada parámetro informa nombre, tipo, coeficiente, error estándar,
estadístico, p-valor, intervalo y significancia al 5 %. Un parámetro no
significativo no invalida automáticamente el modelo.

El diagnóstico residual descarta inicialmente `d + D*s` observaciones en
modelos estacionales y `d` en modelos no estacionales, para limitar el
transitorio de inicialización. Los residuos no finitos se cuentan, se
excluyen explícitamente del diagnóstico y generan una advertencia alta.

Ljung-Box usa:

- ARIMA/ARIMAX: `model_df=p+q`;
- MA: `model_df=q`;
- SARIMA/SARIMAX: `model_df=p+q+P+Q`.

Las exógenas no se incluyen en `model_df`: la prueba se usa para
autocorrelación residual asociada a los componentes AR/MA. El lag es mayor
que `model_df`, menor que la cantidad de residuos y, cuando es viable,
coincide con `s`. No detectar autocorrelación no certifica por sí solo la
validez total.

AIC y BIC provienen del ajuste real. Valores menores son preferibles sólo
al comparar modelos sobre la misma serie y la misma muestra; no son medidas
absolutas de precisión.

ADF contrasta raíz unitaria. La interpretación correcta es “se rechaza la
hipótesis de raíz unitaria” o “no existe evidencia suficiente para
rechazarla”. En SARIMA, ADF regular no determina automáticamente `D`.

`enforce_stationarity` y `enforce_invertibility` indican restricciones
impuestas por statsmodels sobre la parametrización; no demuestran que la
serie original sea estacionaria ni sustituyen un análisis manual de raíces.

## Ejemplos

Los fragmentos de salida son esquemáticos; `…` representa resultados que
deben obtenerse ejecutando la herramienta, no números inventados.

### 1. ARIMA simple

```json
{"valores": [10, 11, 13, 14, 16, 17], "p": 0, "d": 1, "q": 0, "pasos_pronostico": 2}
```

Salida relevante: `{"modelo":"ARIMA(0,1,0)", "pronostico":[…], "intervalos_pronostico":[…]}`.

### 2. MA(1)

```json
{"valores": [20, 21, 19, 22, 20, 23, 21, 22], "q": 1}
```

Salida relevante: `modelo="MA(1)"`, coeficientes MA y diagnóstico ACF/ADF.

### 3. SARIMA mensual

```json
{"valores": ["… al menos varios ciclos …"], "p":1, "d":0, "q":0, "P":1, "D":1, "Q":0, "s":12}
```

`s=12` representa el ciclo anual de datos mensuales. Pocos ciclos producen
una advertencia, no una garantía estadística.

### 4. ARIMAX

```json
{
  "valores": [100, 103, 105, 110],
  "p": 0, "d": 0, "q": 0,
  "variables_exogenas_historicas": {"temperatura":[18,19,20,21],"promocion":[0,0,1,1]},
  "variables_exogenas_futuras": {"temperatura":[22],"promocion":[0]}
}
```

Los coeficientes describen asociación predictiva, no causalidad.

### 5. SARIMAX

Use la estructura de ARIMAX junto con `P`, `D`, `Q` y `s`. La salida separa
coeficientes regulares, estacionales y exógenos.

### 6. Evaluación temporal

```json
{"valores":["…"], "p":1, "d":1, "q":0, "evaluar_modelo":true, "cantidad_prueba":4}
```

Consulte `evaluacion.metricas_prueba`; no la confunda con
`mse_residuos_entrenamiento`.

### 7. Fechas mensuales

```json
{"valores":[10,11,12], "fechas":["2026-01-01","2026-02-01","2026-03-01"], "frecuencia":"mensual", "p":0, "d":1, "q":0}
```

Con frecuencia válida, `fechas_pronostico` continúa la secuencia.

### 8. Falta de exógenas futuras

Una llamada ARIMAX/SARIMAX con exógenas históricas pero sin futuras devuelve
`codigo_error="EXOGENAS_FUTURAS_REQUERIDAS"`.

### 9. Pocos ciclos

Una configuración mensual `s=12` con menos de 24 observaciones devuelve
`CICLOS_ESTACIONALES_INSUFICIENTES`. El ajuste sólo continúa si supera
además el mínimo técnico.

### 10. MAPE con ceros

Para reales `[0,10,20]`, el cero se excluye del denominador y
`mape_detalle.observaciones_excluidas_por_cero` lo informa. Con `[0,0]`,
`mape=null`.

## Compatibilidad con la herramienta ARIMA anterior

Se preservan `valores`, `p`, `d`, `q`, `pasos_pronostico`,
`con_constante`, y las claves históricas `modelo`, `orden`,
`n_observaciones`, `coeficientes`, `aic`, `bic`, `mse_residuos`,
`media_residuos`, `varianza_residuos`, `ljung_box`, `pronostico`.

Se agregaron, de forma compatible, detalle estadístico, intervalos,
evaluación, fechas, confianza, información del ajuste y advertencias.
`mse_residuos` permanece como alias. No se requiere migración inmediata;
los consumidores pueden adoptar las claves nuevas progresivamente.
Pequeñas diferencias numéricas entre versiones de statsmodels son posibles:
no se recomienda fijar AIC, BIC o p-valores exactos.

## Tabla de contratos

| Herramienta | Obligatorios | Opcionales principales | Compatibilidad |
|---|---|---|---|
| AR | `valores,p` | horizonte, confianza | Claves históricas preservadas; campos comunes aditivos |
| MA | `valores,q` | horizonte, constante, confianza, fechas, evaluación | Sin cambios incompatibles |
| ARIMA | `valores,p,d,q` | horizonte, constante, confianza, fechas, evaluación | Alias históricos preservados |
| SARIMA | órdenes regulares y estacionales, `s` | fechas, evaluación | Sin exógenas; sin cambios incompatibles |
| ARIMAX | valores, exógenas históricas, órdenes | exógenas futuras condicionalmente obligatorias, fechas, evaluación | Sin cambios incompatibles |
| SARIMAX | `valores,p,d,q` | estacionalidad y exógenas opcionales | Fachada general aditiva |

## Limitaciones

- Muestras pequeñas y pocos ciclos reducen confiabilidad.
- Sólo se admite una periodicidad estacional.
- Los intervalos condicionados a exógenas no propagan la incertidumbre de
  esas variables.
- La multicolinealidad se diagnostica con heurísticas.
- La fuga de información no puede detectarse completamente.
- Asociación predictiva no implica causalidad.
- ARIMA y SARIMA modelan principalmente la media condicional. Cuando la
  varianza cambia dinámicamente, una extensión como ARIMA-GARCH puede ser
  más apropiada. Este proyecto no implementa GARCH.
- No existe en el repositorio una serie estructurada verificable del caso
  académico de 24 meses; las pruebas disponibles usan series sintéticas
  explícitamente identificadas como tales.

## Trabajo futuro

1. Rolling-origin y monitoreo de degradación predictiva.
2. Selección automática acotada y comparación de modelos.
3. Pruebas específicas de raíz estacional.
4. Escenarios e incertidumbre de exógenas.
5. Visualización de intervalos.
6. ARIMA-GARCH.
7. Múltiples estacionalidades.

## Pruebas

```powershell
python manage.py check
python manage.py test
python -m compileall apps/herramientas
```

El comando oficial es `python manage.py test`; `python -m unittest` sin el
bootstrap de Django no funciona en este repositorio.

