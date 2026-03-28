# Usa una imagen base de Python
FROM python:3.11-slim

ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION}

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia el archivo requirements.txt al contenedor
COPY requirements.txt /app/

# Instala las dependencias del sistema necesarias para WeasyPrint y otras dependencias
RUN apt-get update && \
    apt-get install -y \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código de tu aplicación Django al contenedor
COPY . /app/

# Recopila los archivos estáticos (opcional para producción)
RUN python manage.py collectstatic --noinput

# Expone el puerto en el que Gunicorn se ejecutará
EXPOSE 8000

# Comando para ejecutar Gunicorn
CMD ["gunicorn", "BlogInformatorio.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3", "--timeout", "120"]
