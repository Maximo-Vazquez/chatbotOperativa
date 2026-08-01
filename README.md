# Chatbot Operativa

Aplicación Django que integra un chatbot con herramientas de investigación
operativa y pronóstico de series temporales. Las herramientas se publican al
modelo de lenguaje mediante function calling y se descubren automáticamente
desde `apps/herramientas/tools/`.

## Pronóstico

El proyecto ofrece fachadas pedagógicas para AR, MA, ARIMA, SARIMA, ARIMAX y
SARIMAX. Comparten un núcleo basado en `statsmodels` para ajuste, intervalos,
diagnóstico residual y parámetros, además de módulos comunes para métricas,
holdout temporal, fechas, estacionalidad y variables exógenas.

La guía completa —contratos, ejemplos, interpretación estadística,
compatibilidad y limitaciones— está en
[docs/forecasting.md](docs/forecasting.md).

## Comandos de desarrollo

El runner oficial es el de Django:

```powershell
python manage.py check
python manage.py test
python -m compileall apps/herramientas
```

`python -m unittest` sin inicializar Django no es un comando válido para este
repositorio: no configura `DJANGO_SETTINGS_MODULE` ni el registro de
aplicaciones. Use `python manage.py test`.

## Configuración

Instale las dependencias de `requirements.txt` y configure las variables de
entorno que utiliza `main/settings.py`. No almacene credenciales nuevas en el
repositorio.

