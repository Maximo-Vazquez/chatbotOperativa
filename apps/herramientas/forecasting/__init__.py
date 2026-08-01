"""Nucleo estadistico compartido de pronostico de series temporales.

Separa la logica de ajuste/statsmodels (``engine``), las validaciones de
entrada (``validation``), el diagnostico posterior al ajuste
(``diagnostics``), las estructuras de datos internas (``schemas``) y las
excepciones de dominio (``exceptions``) de las herramientas publicas de
``apps/herramientas/tools/`` que exponen ``TOOL_DEFINITION``/``TOOL_META``/
``TOOL_FUNCTION`` al chatbot.

Este paquete no es una herramienta: el loader dinamico de ``tools.py`` solo
recorre ``apps/herramientas/tools/*.py``, por lo que este modulo nunca se
carga como tool ni requiere excluirse explicitamente.
"""
