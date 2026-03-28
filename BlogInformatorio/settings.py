from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "django-insecure-change-me")
DEBUG = os.environ.get("DJANGO_DEBUG", "true").strip().lower() in {"1", "true", "yes", "on"}

ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if host.strip()
]

CSRF_TRUSTED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "DJANGO_CSRF_TRUSTED_ORIGINS",
        "https://ia.indutienda.com",
    ).split(",")
    if origin.strip()
]

SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
USE_X_FORWARDED_HOST = True

CSRF_TRUSTED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",")
    if origin.strip()
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.sites",
    "apps.login",
    "apps.chatbot",
    "allauth",
    "allauth.account",
    "allauth.socialaccount",
    "allauth.socialaccount.providers.google",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "allauth.account.middleware.AccountMiddleware",
]

ROOT_URLCONF = "BlogInformatorio.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "apps.login.context_processors.google_profile_context",
                "apps.login.context_processors.app_version_context",
            ],
        },
    },
]

WSGI_APPLICATION = "BlogInformatorio.wsgi.application"

DB_ENGINE = os.environ.get("DB_ENGINE", "django.db.backends.sqlite3").strip()
if DB_ENGINE == "django.db.backends.sqlite3":
    DATABASES = {
        "default": {
            "ENGINE": DB_ENGINE,
            "NAME": os.environ.get("DB_NAME", str(BASE_DIR / "db.sqlite3")),
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": DB_ENGINE,
            "NAME": os.environ.get("DB_NAME", "chatbot"),
            "USER": os.environ.get("DB_USER", "postgres"),
            "PASSWORD": os.environ.get("DB_PASSWORD", "max135"),
            "HOST": os.environ.get("DB_HOST", "localhost"),
            "PORT": os.environ.get("DB_PORT", "5432"),
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "es-ar"
TIME_ZONE = "America/Argentina/Buenos_Aires"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"] if (BASE_DIR / "static").exists() else []

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

SITE_ID = int(os.environ.get("SITE_ID", "1") or 1)

AUTHENTICATION_BACKENDS = [
    "allauth.account.auth_backends.AuthenticationBackend",
    "django.contrib.auth.backends.ModelBackend",
]

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_CLIENT_KEY = os.environ.get("GOOGLE_CLIENT_KEY", "")

SOCIALACCOUNT_PROVIDERS = {
    "google": {
        "APP": {
            "client_id": GOOGLE_CLIENT_ID,
            "secret": GOOGLE_CLIENT_SECRET,
            "key": GOOGLE_CLIENT_KEY,
        },
        "SCOPE": ["profile", "email"],
        "AUTH_PARAMS": {"access_type": "online", "prompt": "consent"},
        "OAUTH_PKCE_ENABLED": True,
    }
}

SOCIALACCOUNT_ADAPTER = "apps.login.adapters.MySocialAccountAdapter"
SOCIALACCOUNT_AUTO_SIGNUP = True
SOCIALACCOUNT_QUERY_EMAIL = True
SOCIALACCOUNT_LOGIN_ON_GET = True
ACCOUNT_EMAIL_VERIFICATION = "none"
ACCOUNT_EMAIL_REQUIRED = True
ACCOUNT_USERNAME_REQUIRED = True

LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/login_registro/"
ACCOUNT_LOGOUT_REDIRECT_URL = "/login_registro/"

APP_VERSION = os.environ.get("APP_VERSION", "dev-local").strip() or "dev-local"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
CHATBOT_MODEL = os.environ.get("CHATBOT_MODEL", "deepseek-chat").strip() or "deepseek-chat"
CHATBOT_SYSTEM_PROMPT = os.environ.get(
    "CHATBOT_SYSTEM_PROMPT",
    (
        "Sos un asistente experto en pronóstico con procesos estocásticos y series temporales, "
        "orientado a estudiantes y profesionales que están aprendiendo estos temas. "
        "Tu rol incluye:\n"
        "1. CONCEPTOS: Explicar claramente definiciones como serie temporal, ruido blanco, tendencia, "
        "estacionalidad, ciclo, componente irregular, proceso estocástico, cadena de Markov, etc.\n"
        "2. CLASIFICACIÓN: Ayudar a determinar si una serie es estacionaria o no estacionaria. "
        "Guiar en pruebas como Dickey-Fuller (ADF), KPSS, Phillips-Perron. "
        "Explicar conceptos de raíz unitaria, integración I(d), cointegración.\n"
        "3. ANÁLISIS EXPLORATORIO: Orientar en el análisis visual y estadístico: "
        "gráficos de la serie, histogramas, boxplots por período, descomposición (aditiva/multiplicativa).\n"
        "4. ACF/PACF: Guiar la interpretación de la función de autocorrelación (ACF) y "
        "autocorrelación parcial (PACF). Explicar qué patrones indican AR, MA, ARMA, ARIMA, SARIMA. "
        "Ayudar a identificar los órdenes p, d, q.\n"
        "5. DETECCIÓN DE PATRONES: Ayudar a identificar tendencia, estacionalidad, "
        "quiebres estructurales, valores atípicos (outliers), heterocedasticidad.\n"
        "6. PREPARACIÓN PARA MODELADO: Guiar en transformaciones (log, diferenciación, Box-Cox), "
        "tratamiento de valores faltantes, normalización, y definición de hipótesis antes de elegir modelo.\n"
        "7. FLUJO GUIADO: Cuando el usuario lo pida, conducirlo paso a paso por el flujo completo "
        "de análisis: exploración → estacionariedad → ACF/PACF → selección de modelo → validación.\n\n"
        "Respondé siempre en español. Sé claro, didáctico y concreto. "
        "Cuando des fórmulas, usá notación matemática legible. "
        "Cuando des código, preferí Python con pandas, statsmodels o pmdarima. "
        "Si el usuario comparte datos o resultados, interpretálos en contexto."
    ),
).strip()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
