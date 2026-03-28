# API de Gestion de Tenants

Documentacion tecnica de los endpoints para que sistemas externos gestionen tiendas (tenants) en la plataforma.

---

## Autenticacion

Todos los endpoints requieren autenticacion mediante API Key.

| Header | Valor |
|--------|-------|
| `X-API-KEY` | `{TENANT_MANAGEMENT_API_KEY}` |
| `Content-Type` | `application/json` |

> [!CAUTION]
> La API Key debe configurarse en `settings.py` como `TENANT_MANAGEMENT_API_KEY`. Sin esta clave, todas las peticiones retornan `401 Unauthorized`.

---

## Resumen de Endpoints

| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| `POST` | `/api/tenants/` | Crear tenant con dominio, plan y usuario admin opcional |
| `GET` | `/api/tenants/list/` | Listar todos los tenants |
| `GET` | `/api/tenants/{esquema}/info/` | Consultar estado completo |
| `DELETE` | `/api/tenants/{esquema}/` | Eliminar tenant |
| `POST` | `/api/tenants/{esquema}/pause/` | Pausar/reanudar tenant |
| `PATCH` | `/api/tenants/{esquema}/plan/` | Cambiar plan de suscripcion |
| `PATCH` | `/api/tenants/{esquema}/renovar/` | Renovar suscripcion (fecha fin) |

---

## 1. Listar Tenants

Devuelve todos los tenants registrados con su estado de suscripcion.

```http
GET /api/tenants/list/
```

### Ejemplo Response (200 OK)

```json
{
  "tenants": [
    {
      "id": 24,
      "nombre": "Indumentaria",
      "esquema": "indu",
      "dominio": "indu.mycloudnas.com",
      "dominio_id": 24,
      "pagado_hasta": "2026-06-01",
      "en_prueba": false,
      "pausado": false,
      "creado_en": "2026-02-03",
      "suscripcion": {
        "plan": "pro",
        "plan_etiqueta": "Pro",
        "fecha_fin": "2026-06-01T00:00:00-03:00",
        "activo": true,
        "vigente": true,
        "suscripcion_id_externa": "sub_abc123",
        "caracteristicas": {
          "pedidos": true,
          "productos": true,
          "inventario": true
        }
      }
    }
  ]
}
```

---

## 2. Crear Tenant

Crea una nueva tienda con esquema de base de datos aislado, dominio principal, plan y usuario administrador opcional.

```http
POST /api/tenants/
```

### Request Body

| Campo | Tipo | Requerido | Descripcion |
|-------|------|-----------|-------------|
| `nombre` | string | si | Nombre comercial de la tienda |
| `esquema` | string | si | Nombre del schema PostgreSQL (unico, sin espacios) |
| `dominio` | string | si | Subdominio o dominio completo |
| `pagado_hasta` | string | no* | Fecha `YYYY-MM-DD` |
| `en_prueba` | boolean | no | `true` por defecto |
| `plan` | string | no | `demo`, `inicio`, `plus`, `pro` (default: `inicio`) |
| `usuario_admin` | object | no | Datos del administrador inicial |
| `usuario_admin.email` | string | si* | Email (tambien se usa como username) |
| `usuario_admin.password` | string | si* | Contrasena inicial |
| `usuario_admin.nombre` | string | no | Nombre del usuario |

*Requerido solo si se envia `usuario_admin`.

**\* `pagado_hasta` es obligatorio cuando `en_prueba=true`.**

> [!IMPORTANT]
> Si se envia `usuario_admin`, el usuario se crea con `is_staff=True` y `is_superuser=False`.

### Ejemplo Request

```json
{
  "nombre": "Nueva Boutique",
  "esquema": "nueva_boutique",
  "dominio": "boutique.mitiendaropa.com",
  "usuario_admin": {
    "email": "admin@boutique.com",
    "password": "ClaveSegura2026!",
    "nombre": "Juan Perez"
  },
  "plan": "plus",
  "pagado_hasta": "2026-06-01"
}
```

### Ejemplo Request (alta en prueba)

```json
{
  "nombre": "Tienda Trial",
  "esquema": "tienda_trial",
  "dominio": "trial.mitiendaropa.com",
  "en_prueba": true,
  "plan": "plus",
  "pagado_hasta": "2026-03-15"
}
```

### Ejemplo Request (alta sin prueba)

```json
{
  "nombre": "Tienda Estable",
  "esquema": "tienda_estable",
  "dominio": "estable.mitiendaropa.com",
  "en_prueba": false,
  "plan": "plus"
}
```

### Ejemplo Response (201 Created)

```json
{
  "id": 6,
  "nombre": "Nueva Boutique",
  "esquema": "nueva_boutique",
  "dominio": "boutique.mitiendaropa.com",
  "dominio_id": 6,
  "pagado_hasta": "2026-06-01",
  "en_prueba": false,
  "pausado": false,
  "creado_en": "2026-02-03",
  "plan": "plus",
  "usuario_admin": {
    "id": 1,
    "email": "admin@boutique.com",
    "creado": true
  }
}
```

`usuario_admin` solo aparece en la respuesta si se envio en el request.

> [!NOTE]
> El plan se aplica exactamente segun el valor recibido en `plan` (no se fuerza automaticamente un plan especifico por estar en prueba).

### Errores posibles

| Codigo | Detalle |
|--------|---------|
| `400` | Campos requeridos faltantes, JSON invalido, formato invalido o `en_prueba=true` sin `pagado_hasta` |
| `401` | API Key invalida o ausente |
| `409` | Esquema o dominio ya existe |

### Ejemplo Error (400)

```json
{
  "detalle": "El campo 'pagado_hasta' es obligatorio cuando 'en_prueba' es true."
}
```

---

## 3. Consultar Tenant

Obtiene informacion completa de un tenant incluyendo su suscripcion.

```http
GET /api/tenants/{esquema}/info/
```

### Path Parameters

| Parametro | Descripcion |
|-----------|-------------|
| `esquema` | Nombre del schema del tenant |

### Ejemplo Response (200 OK)

```json
{
  "id": 5,
  "nombre": "Tienda Ejemplo SA",
  "esquema": "tienda_ejemplo",
  "dominio": "ejemplo.mitiendaropa.com",
  "dominio_id": 5,
  "pagado_hasta": "2026-12-31",
  "en_prueba": false,
  "pausado": false,
  "creado_en": "2026-01-15",
  "suscripcion": {
    "plan": "pro",
    "plan_etiqueta": "Pro",
    "fecha_fin": "2026-12-31T00:00:00-03:00",
    "activo": true,
    "vigente": true
  }
}
```

---

## 4. Pausar/Reanudar Tenant

Bloquea o desbloquea el acceso a una tienda.

```http
POST /api/tenants/{esquema}/pause/
```

### Request Body

| Campo | Tipo | Descripcion |
|-------|------|-------------|
| `pausado` | boolean | `true` para pausar, `false` para reanudar (default: `true`) |

### Ejemplo Request

```json
{
  "pausado": true
}
```

### Ejemplo Response (200 OK)

Devuelve la estructura base del tenant (`id`, `nombre`, `esquema`, `dominio`, `dominio_id`, `pagado_hasta`, `en_prueba`, `pausado`, `creado_en`).

---

## 5. Cambiar Plan de Suscripcion

Actualiza el nivel de plan de un tenant.

```http
PATCH /api/tenants/{esquema}/plan/
```

### Request Body

| Campo | Tipo | Descripcion |
|-------|------|-------------|
| `plan` | string | Nuevo plan: `demo`, `inicio`, `plus`, `pro` |

Notas:
- Tambien se aceptan alias como `trial/prueba` -> `demo`, `basica` -> `inicio`, `avanzado` -> `plus`, `profesional/empresarial` -> `pro`.

### Ejemplo Request

```json
{
  "plan": "pro"
}
```

### Ejemplo Response (200 OK)

```json
{
  "esquema": "tienda_ejemplo",
  "plan_anterior": "plus",
  "plan_nuevo": "pro",
  "actualizado": true
}
```

---

## 6. Renovar Suscripcion

Extiende la fecha de vencimiento de un tenant.

```http
PATCH /api/tenants/{esquema}/renovar/
```

### Request Body

| Campo | Tipo | Descripcion |
|-------|------|-------------|
| `fecha_fin` | string | Nueva fecha: `YYYY-MM-DD` o `YYYY-MM-DDTHH:MM:SS` |

### Ejemplo Request

```json
{
  "fecha_fin": "2027-02-03"
}
```

### Ejemplo Response (200 OK)

```json
{
  "esquema": "tienda_ejemplo",
  "fecha_fin_anterior": "2026-02-03T00:00:00-03:00",
  "fecha_fin_nueva": "2027-02-03T00:00:00-03:00",
  "renovado": true
}
```

---

## 7. Eliminar Tenant

Elimina permanentemente una tienda y todos sus datos.

```http
DELETE /api/tenants/{esquema}/
```

### Ejemplo Response (200 OK)

```json
{
  "detalle": "Inquilino eliminado correctamente.",
  "esquema": "tienda_ejemplo"
}
```

> [!CAUTION]
> Esta operacion es irreversible.

---

## Codigos de Error Comunes

| Codigo | Significado | Causa |
|--------|-------------|-------|
| `400` | Bad Request | JSON mal formado o campos invalidos |
| `401` | Unauthorized | API Key faltante o incorrecta |
| `404` | Not Found | Tenant no existe |
| `405` | Method Not Allowed | Metodo HTTP no permitido |
| `409` | Conflict | Esquema/dominio duplicado |
| `500` | Server Error | Error interno del servidor (principalmente en plan/renovar) |

---

## Notas de Implementacion

- Base URL: debe apuntar a un dominio que resuelva al esquema `public`.
- Todas las vistas fuerzan ejecucion en esquema `public` y cambian al schema tenant solo cuando necesitan leer/escribir su suscripcion.
- Timeout recomendado: 30 segundos (la creacion de esquema puede demorar).
- Idempotencia: `POST /api/tenants/` no es idempotente.
