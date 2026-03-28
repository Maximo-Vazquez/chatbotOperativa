# MF Software - Plataforma de Venta de Software como Servicio

<div align="center">

![Django](https://img.shields.io/badge/Django-092E20?style=flat&logo=django&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![MercadoPago](https://img.shields.io/badge/MercadoPago-00B1EA?style=flat&logo=mercadopago&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=flat&logo=kubernetes&logoColor=white)

**Plataforma multi-tenant para la venta y gestión de software como servicio (SaaS)**

</div>

---

## 🎯 ¿Qué es este proyecto?

Una plataforma web completa que permite:
- **Vender software como servicio** con planes de suscripción
- **Gestionar múltiples tenants** (cada cliente tiene su propio entorno aislado)
- **Procesar pagos** con Mercado Pago (únicos y recurrentes)
- **Administrar suscripciones, facturación y renovaciones**

---

## 🚀 Inicio Rápido

```bash
# 1. Clonar e instalar dependencias
git clone <repo-url>
cd TiendaSoftware
pip install -r requirements.txt

# 2. Configurar variables de entorno
cp .env.example .env  # Editar con tus credenciales

# 3. Ejecutar migraciones y servidor
python manage.py migrate
python manage.py runserver
```

Acceder a `http://localhost:8000`

---

## 📁 Estructura del Proyecto

```
TiendaSoftware/
├── BlogInformatorio/     # Configuración principal de Django
├── apps/                 # Módulos de la aplicación
│   ├── administracion/   # Usuarios, productos, pedidos, facturas
│   ├── chatbot/          # Asistente con OpenAI
│   ├── login/            # Autenticación (incluyendo Google OAuth)
│   ├── metodos/          # Pasarelas de pago (MercadoPago)
│   ├── referidos/        # Sistema de afiliados
│   └── softwareVentas/   # Catálogo y proceso de compra
├── k8s/                  # Manifiestos de Kubernetes
├── .github/workflows/    # CI/CD con GitHub Actions
└── Documentacion/        # Guías técnicas
```

---

## 📚 Documentación

> **Antes de trabajar en este proyecto, lee la documentación relevante:**

| Tema | Descripción | Enlace |
|------|-------------|--------|
| 💳 **Pagos y Suscripciones** | Integración con Mercado Pago, cuentas de prueba, webhooks | [Ver guía](./Documentacion/docs/pagos-suscripcion-mercadopago.md) |
| 🚀 **Despliegue** | CI/CD, Docker, Kubernetes, NAS QNAP | [Ver guía](./Documentacion/docs/despliegue.md) |
| 🏗️ **Arquitectura** | Multi-tenancy, APIs externas, módulos | [Ver guía](./Documentacion/docs/arquitectura.md) |

---

## ⚙️ APIs Externas Utilizadas

| API | Propósito | Configuración |
|-----|-----------|---------------|
| **MercadoPago** | Pagos únicos y suscripciones | `ACCESS_TOKEN_MP`, `ACCESS_TOKEN_MP_SUSCRIPCION` |
| **OpenAI** | Chatbot de asistencia | `OPENAI_API_KEY` |
| **GoDaddy** | Verificación de dominios | `GODADDY_API_KEY`, `GODADDY_API_SECRET` |
| **Google OAuth** | Login con Google | Configurar en `django-allauth` |

---

## 🔄 CI/CD (GitHub Actions)

El proyecto usa workflows para automatizar build y despliegue:

### 1. Construir Imagen Docker
**Trigger:** Crear un tag `v*` (ej: `v1.0.5`)

```bash
git tag v1.0.5
git push origin v1.0.5
```
→ Construye y sube la imagen a `ghcr.io`

### 2. Desplegar a NAS QNAP
**Trigger:** Manual desde GitHub (workflow_dispatch)

### 3. Deploy Manual a VPS
**Trigger:** Manual desde GitHub (workflow_dispatch)

- Copia manifiestos `k8s/*.yaml` al VPS.
- Usa el secret S3 ya existente en el cluster (`s3-credentials-saas`).
- Actualiza imagen del deployment y ejecuta rollout.

**Secrets adicionales requeridos para VPS/S3:**

| Secret | Descripcion |
|--------|-------------|
| `VPS_HOST` | Host/IP del VPS |
| `VPS_PORT` | Puerto SSH del VPS |
| `VPS_USER` | Usuario SSH |
| `VPS_SSH_KEY` | Clave privada SSH para deploy |

→ Copia los manifiestos K8s al NAS y ejecuta `kubectl apply`

📖 [Guía detallada de despliegue](./.github/workflows/GUIA_DESPLIEGUE_IMAGENES.md)

---

## 🧩 Módulos Principales

| Módulo | Función |
|--------|---------|
| **administracion** | Modelos centrales: `Usuario`, `Software`, `PedidoSuscripcion`, `Pago`, `Factura` |
| **softwareVentas** | Catálogo de productos, planes, validación de dominios, checkout |
| **metodos/mercadoPago** | Creación de preferencias, suscripciones, webhooks |
| **login** | Registro, autenticación, grupos de usuarios |
| **chatbot** | Asistente conversacional con OpenAI API |
| **referidos** | Sistema de afiliados y comisiones |

---

## 🏪 Funcionalidades del Software

<details>
<summary><b>Ver todas las funcionalidades</b></summary>

### 💼 Gestión de Marca
- Logotipo, colores, contactos, redes sociales
- SEO y ubicación en mapa

### 👩‍💻 Equipo y Control
- Roles predefinidos y permisos personalizados
- Usuarios internos con acceso seguro

### 👕 Catálogo
- Productos con variantes (talles, colores, fotos)
- Categorías y descuentos inteligentes

### 🛒 Experiencia de Compra
- Filtros avanzados, búsqueda rápida
- Carrito persistente, descuentos automáticos

### 📦 Pedidos e Inventario
- Seguimiento de estados, notificaciones
- Control de stock, alertas automáticas

### 💳 Pagos
- Mercado Pago, tarjeta, efectivo
- Suscripciones recurrentes

### 📊 Reportes
- Ingresos, costos, rentabilidad
- Comparativos mensuales

</details>

---

## 📝 Variables de Entorno Requeridas

```env
# Base de datos
DATABASE_URL=postgres://...

# Mercado Pago
ACCESS_TOKEN_MP=TEST-...
ACCESS_TOKEN_MP_SUSCRIPCION=APP_USR-...
PUBLIC_KEY_MP=TEST-...

# OpenAI
OPENAI_API_KEY=sk-...

# GoDaddy
GODADDY_API_KEY=...
GODADDY_API_SECRET=...

# Media en S3/MinIO (SaaS)
USE_S3_MEDIA=true
AWS_S3_ENDPOINT_URL=https://s3.indutienda.com
AWS_STORAGE_BUCKET_NAME=media
AWS_S3_REGION_NAME=us-east-1
AWS_S3_SIGNATURE_VERSION=s3v4
AWS_S3_ADDRESSING_STYLE=path
AWS_QUERYSTRING_AUTH=true
AWS_QUERYSTRING_EXPIRE=3600
AWS_S3_FILE_OVERWRITE=false
AWS_MEDIA_LOCATION=saas
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# Producción
ES_PRODUCCION=True/False
SITE_URL=https://tu-dominio.com
```

---

## 📄 Licencia

Proyecto privado © MF Software

---

<div align="center">

**¿Preguntas?** Revisa la [documentación](./Documentacion/docs/) o el historial de commits.

</div>
