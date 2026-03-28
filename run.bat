@echo off
rem Activa el entorno virtual
call entorno\Scripts\activate

rem Ejecuta el servidor de Django
python manage.py runserver

rem Desactiva el entorno virtual cuando termines (opcional)
deactivate