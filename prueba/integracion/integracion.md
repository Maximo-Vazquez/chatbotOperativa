# Integración con API de Tenants y Suscripciones

Este documento describe los endpoints HTTP disponibles para que sistemas externos
administren tenants y suscripciones dentro de la plataforma. Incluye detalles de
autenticación, estructuras de request/response y reglas de negocio relevantes.

> **Nota:** Todas las rutas aquí descriptas se publican desde el dominio/base URL
> del proyecto Django y deben consumirse con HTTPS en entornos productivos.

## Autenticación

Los endpoints están protegidos con una API key configurable mediante la variable
de entorno `TENANT_MANAGEMENT_API_KEY` (valor por defecto: `32323232-3232-4323-8432-323432343234`).
Cada request debe incluir el encabezado HTTP:

```
X-API-KEY: <valor_configurado>
```

Si la API key falta o no coincide se devolverá `401 Unauthorized` con el cuerpo:

```json
{"detalle": "API key inválida o ausente."}
```

## Endpoints de Tenants

Base: `/api/tenants/`

### Crear tenant

- **Método:** `POST /api/tenants/`
- **Body (JSON):**
  ```json
  {
    "nombre": "Nombre comercial visible",
    "esquema": "identificador_esquema",
    "dominio": "tenant.ejemplo.com",
    "pagado_hasta": "2024-12-31",
    "en_prueba": true
  }
  ```
  - `nombre`, `esquema` y `dominio` son obligatorios.
  - `pagado_hasta` es opcional y debe respetar el formato `YYYY-MM-DD`.
  - `en_prueba` es booleano (por defecto `true`).
- **Response 201 (JSON):**
  ```json
  {
    "id": 5,
    "nombre": "Nombre comercial visible",
    "esquema": "identificador_esquema",
    "dominio": "tenant.ejemplo.com",
    "dominio_id": 12,
    "pagado_hasta": "2024-12-31",
    "en_prueba": true,
    "pausado": false,
    "creado_en": "2024-01-10T15:23:01.512345"
  }
  ```
- **Errores comunes:**
  - `400` si faltan campos obligatorios, si `pagado_hasta` no es una fecha válida o si viola constraints únicos (`schema`, `domain`).

### Eliminar tenant

- **Método:** `DELETE /api/tenants/<esquema>/`
- **Response 200 (JSON):**
  ```json
  {"detalle": "Inquilino eliminado correctamente.", "esquema": "identificador_esquema"}
  ```
- **Errores comunes:** `404` si el esquema no existe.

### Pausar/Reanudar tenant

- **Método:** `POST /api/tenants/<esquema>/pause/`
- **Body (JSON):**
  ```json
  {"pausado": true}
  ```
  - `pausado` es opcional, por defecto se asume `true` (pausar). Enviar `false` para reanudar.
- **Response 200 (JSON):** retorna el mismo esquema de la creación del tenant con el estado actualizado.
- **Errores comunes:** `404` si el esquema no existe.

## Endpoints de Suscripción

Base: `/api/suscripciones/`

Cada operación se ejecuta dentro del esquema público y, al consultar o modificar
una suscripción, el sistema cambia temporalmente al esquema del tenant solicitado.

### Listar todas las suscripciones

- **Método:** `GET /api/suscripciones/`
- **Response 200 (JSON):**
  ```json
  {
    "suscripciones": [
      {
        "esquema": "tienda1",
        "inquilino": "Tienda 1",
        "plan": "inicio",
        "plan_etiqueta": "Inicio",
        "activo": true,
        "vigente": true,
        "fecha_fin": "2024-12-31T23:59:59-03:00",
        "suscripcion_id_externa": "sub_123",
        "caracteristicas": {
          "pedidos": true,
          "productos": false,
          "categorias": false,
          "promociones": false,
          "boletines": false,
          "inventario": false,
          "punto_venta": false,
          "contabilidad": false,
          "usuarios": false,
          "ajustes": false
        }
      }
    ]
  }
  ```

### Consultar detalle de una suscripción

- **Método:** `GET /api/suscripciones/<esquema>/`
- **Response 200 (JSON):** objeto idéntico a cada ítem del listado anterior.
- **Errores comunes:** `404` si el tenant no existe.

### Actualizar una suscripción

- **Método:** `PATCH /api/suscripciones/<esquema>/`
- **Body (JSON):** se pueden enviar uno o más de los siguientes campos:
  ```json
  {
    "plan": "plus",
    "activo": true,
    "fecha_fin": "2024-12-31T23:59:59-03:00",
    "suscripcion_id_externa": "sub_456"
  }
  ```
  - `plan` acepta: `inicio`, `plus`, `pro` (sin mayúsculas). También se aceptan alias:
    `basica` → `inicio`, `avanzado` → `plus`, `profesional`/`empresarial` → `pro`.
  - `fecha_fin` debe venir en formato ISO-8601. Se admite fecha/hora sin zona, que se asume en la zona del servidor.
  - `suscripcion_id_externa` puede ser `null`/cadena vacía para limpiar el valor.
- **Response 200 (JSON):** mismo esquema que el detalle/listado.
- **Errores comunes:**
  - `400` si no se envían campos válidos o si `plan`/`fecha_fin` tienen formato incorrecto.
  - `404` si el tenant no existe.

### Regla de vigencia y características

Según el modelo `Suscripcion`, cada tenant tiene un único registro por esquema.
El campo `vigente` (en las respuestas) se calcula como verdadero solo cuando:

- `activo` es `true`, y
- `fecha_fin` está vacía o es posterior a la hora actual.

La clave `caracteristicas` devuelve un mapa de banderas para habilitar módulos.
El plan mínimo requerido por característica se resume así:

| Característica    | Plan mínimo |
| ----------------- | ----------- |
| pedidos           | Inicio      |
| productos         | Plus        |
| categorias        | Plus        |
| promociones       | Plus        |
| boletines         | Plus        |
| usuarios          | Plus        |
| inventario        | Pro         |
| punto_venta       | Pro         |
| contabilidad      | Pro         |
| ajustes           | Pro         |

## Webhook de eventos de suscripción (opcional)

Además de los endpoints anteriores, existe `POST /webhook/suscripcion/` para que
el proveedor de pagos notifique altas, bajas o cambios. El payload aceptado es el
mismo que los campos descritos en la sección de actualización, con un campo
adicional `evento` que puede ser `suscripcion.creada`, `suscripcion.actualizada`,
`suscripcion.cancelada` o `suscripcion.expirada`.

## Ejemplo de flujo

1. El sistema externo crea el tenant vía `POST /api/tenants/`.
2. Opcionalmente actualiza la suscripción con el plan contratado.
3. Ante cancelaciones, invoca `PATCH /api/suscripciones/<esquema>/` para marcar
   `activo=false` o ajustar `fecha_fin`.
4. Si el tenant debe bloquearse por falta de pago, se usa `POST /api/tenants/<esquema>/pause/`.

Con esta integración, el equipo externo puede administrar de forma centralizada
el ciclo de vida de cada tenant y mantener sincronizado el estado de sus planes.
