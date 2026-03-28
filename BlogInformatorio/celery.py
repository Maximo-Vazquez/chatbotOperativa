"""Configuración de Celery para el proyecto BlogInformatorio."""
import os
from celery import Celery

# Establece el módulo de settings de Django para el programa 'celery'.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BlogInformatorio.settings")

app = Celery("BlogInformatorio")

# Usa la configuración de Django. El namespace 'CELERY' significa que todas
# las claves de configuración de Celery deben tener ese prefijo.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Carga automáticamente los módulos de tareas de todas las apps registradas.
app.autodiscover_tasks()