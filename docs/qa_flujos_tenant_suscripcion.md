# Guia de Pruebas End-to-End
## Integracion Tenant API + Gestion de Suscripciones

Fecha: 2026-02-13

Este documento describe los flujos que hay que ejecutar para validar que la integracion nueva de tenants y la gestion de suscripciones funciona correctamente.

---

## 1. Precondiciones

1. Tener configuradas estas variables en `settings` / entorno:
   - `TENANT_MANAGEMENT_API_BASE_URL`
   - `TENANT_MANAGEMENT_API_KEY`
   - `TENANT_MANAGEMENT_TIMEOUT`
2. Tener el servidor Django levantado.
3. Tener al menos:
   - 1 usuario cliente
   - 1 software con promociones activas
   - promocion anual de 12 meses por plan (`inicio`, `plus`, `pro`) si se quiere probar todo el flujo.
4. Ejecutar chequeos basicos:

```bash
python manage.py check
python manage.py makemigrations --check
```

Resultado esperado: sin errores y "No changes detected".

---

## 2. Datos de prueba recomendados

Preparar 3 suscripciones para el mismo usuario:

1. `SUSC_A`: activa, faltan > 365 dias (ej. 400 dias).
2. `SUSC_B`: activa, faltan < 365 dias (ej. 300 dias).
3. `SUSC_C`: activa, faltan aprox 730 dias (para validar tope de 2 anios).

Y con `tenant_schema` cargado (para pausar/reanudar y cambio de plan remoto).

---

## 3. Flujo de creacion de tenant (usuario_admin automatico)

Objetivo: validar que al crear tenant se envia `usuario_admin` automatico.

### Pasos

1. Crear una nueva suscripcion/pedido que termine en alta de tenant.
2. Completar el flujo hasta que se ejecute sincronizacion con gestor externo.
3. Revisar en la API de tenants (o logs del backend externo) el payload recibido en `POST /api/tenants/`.

### Esperado

1. Se envia `nombre`, `esquema`, `dominio`, `plan`, `pagado_hasta`.
2. Se envia `usuario_admin` con:
   - `nombre`: primera palabra de `get_full_name()` o fallback `username`
   - `email`: email del usuario (o fallback tecnico)
   - `password`: generada aleatoria segura
3. La password NO se persiste en campos locales de tenant.

---

## 4. Flujo pausar / reanudar desde panel usuario

Ruta UI: `panel/software/administrar/<suscripcion_id>/`

### Caso 4.1 Pausar

1. Entrar a administrar suscripcion con tenant activo.
2. Click en `Pausar`.
3. Confirmar.

Esperado:

1. Se hace POST a `panel/software/administrar/<id>/pausa/`.
2. Se llama API externa `POST /api/tenants/{esquema}/pause/` con `pausado=true`.
3. Se actualiza `Suscripcion.tenant_pausado=True`.
4. Mensaje de exito visible.

### Caso 4.2 Reanudar

1. Repetir sobre una suscripcion pausada.

Esperado:

1. Se envia `pausado=false`.
2. Se actualiza `tenant_pausado=False`.
3. Mensaje de exito visible.

### Caso 4.3 Sin tenant_schema

1. Probar con suscripcion sin `tenant_schema`.

Esperado:

1. No intenta llamada remota.
2. Muestra error controlado.

---

## 5. Flujo renovar suscripcion (regla 2 anios)

Ruta UI: `panel/software/renovar/<suscripcion_id>/`

### Caso 5.1 Bloqueado por tiempo restante >= 365

Usar `SUSC_A`.

Esperado:

1. La UI informa que aun no se puede renovar.
2. No se crea pedido.

### Caso 5.2 Permitido con < 365 dias

Usar `SUSC_B`.

1. Entrar a renovar.
2. Continuar con renovacion.

Esperado:

1. Solo se renueva en bloque de 12 meses.
2. Se crea `PedidoSuscripcion` con:
   - `accion='extenderSuscripcion'`
   - `meses_a_pagar=12`
   - `estado_pago='pendiente'`
3. Redirige a `pago_pedido`.

### Caso 5.3 Bloqueado por tope 2 anios

Usar `SUSC_C`.

Esperado:

1. Muestra mensaje de tope superado.
2. No crea pedido.

### Caso 5.4 Aplicacion tras pago aprobado

1. Aprobar pago del pedido de renovacion (mercado pago/test).
2. Esperar procesamiento.

Esperado:

1. Se extiende `fecha_fin` local.
2. Se sincroniza remoto via `PATCH /api/tenants/{esquema}/renovar/`.

---

## 6. Flujo cambio de plan (upgrade/downgrade)

Ruta UI: `panel/software/administrar/<suscripcion_id>/` -> `Actualizar Plan`.

### Caso 6.1 Upgrade (inicio->plus o plus->pro)

1. Seleccionar un plan superior.
2. Confirmar.

Esperado:

1. Se calcula prorrateo por dias restantes.
2. Se crea pedido por diferencia (`estado_pago='pendiente'`).
3. NO cambia `suscripcion.nivel` inmediatamente.
4. Redirige al flujo de pago.

### Caso 6.2 Upgrade aplicado al pagar

1. Completar pago del pedido de upgrade.

Esperado:

1. Se actualiza plan activo local.
2. Se sincroniza remoto con `PATCH /api/tenants/{esquema}/plan/`.

### Caso 6.3 Downgrade (pro->plus o plus->inicio)

1. Seleccionar plan inferior.
2. Confirmar.

Esperado:

1. No cambia plan inmediato.
2. Guarda:
   - `nivel_programado=<nuevo plan>`
   - `fecha_aplicar_nivel_programado=fecha_fin actual`
3. Mensaje de cambio programado.

### Caso 6.4 Cancelar downgrade programado

1. Click en `Cancelar cambio programado`.

Esperado:

1. `nivel_programado=NULL`
2. `fecha_aplicar_nivel_programado=NULL`

---

## 7. Aplicacion automatica de plan programado

Objetivo: validar que al llegar la fecha se aplica automaticamente.

### Forma A (manual con fecha en pasado)

1. En DB, setear para una suscripcion:
   - `nivel_programado='inicio'`
   - `fecha_aplicar_nivel_programado` en pasado
2. Abrir una de estas pantallas:
   - listado de suscripciones
   - administrar suscripcion
   - renovar suscripcion

Esperado:

1. Se mueve `nivel_programado` a `nivel`.
2. Se limpian campos programados.
3. Se intenta sincronizar remoto `PATCH /plan/`.

### Forma B (durante procesamiento de pedidos)

1. Ejecutar flujo de pago/procesamiento sobre esa suscripcion.

Esperado:

1. Se aplica tambien durante `procesar_pedido`.

---

## 8. Pruebas de compatibilidad (admin/integracion)

### Caso 8.1 Panel admin de suscripcion

Ruta: `administracion/suscripciones/<id>/`

1. Probar extender.
2. Probar actualizar plan.

Esperado:

1. Usan metodos nuevos (renovar/plan de tenants API).
2. Sin errores de endpoints viejos (`/api/suscripciones/...`).

### Caso 8.2 Estado de despliegue/pago

1. Ejecutar despliegue de pago.

Esperado:

1. Consulta estado remoto via `GET /api/tenants/{esquema}/info/`.
2. Lee `suscripcion.activo` / `suscripcion.vigente` correctamente.

---

## 9. Consultas utiles de verificacion en DB

```python
# python manage.py shell
from apps.administracion.models import Suscripcion, PedidoSuscripcion

# Ver flags de tenant y planes
Suscripcion.objects.values(
    'id', 'nivel', 'nivel_programado', 'fecha_aplicar_nivel_programado',
    'tenant_schema', 'tenant_pausado', 'fecha_fin'
)

# Ultimos pedidos de renovacion/cambio
PedidoSuscripcion.objects.order_by('-id').values(
    'id', 'suscripcion_id', 'accion', 'nivel', 'meses_a_pagar',
    'monto_a_pagar', 'estado_pago', 'detalle'
)[:20]
```

---

## 10. Criterios de aceptacion final

Se considera OK cuando:

1. No hay llamadas a endpoints viejos `/api/suscripciones/...`.
2. Pausa/reanudar funciona desde panel usuario y persiste `tenant_pausado`.
3. Renovacion respeta:
   - `< 365` dias
   - `+12` meses
   - tope `2` anios
4. Upgrade crea pedido prorrateado y aplica tras pago.
5. Downgrade queda programado y se aplica al vencer.
6. Creacion tenant envia siempre `usuario_admin` automatico.
7. `python manage.py check` sin errores.

---

## 11. Problemas comunes

1. Error 401 en API externa:
   - revisar `TENANT_MANAGEMENT_API_KEY`.
2. Timeout en sincronizacion:
   - subir `TENANT_MANAGEMENT_TIMEOUT`.
3. No aparece renovacion anual:
   - verificar promocion `duracion_meses=12` y `disponible=True` para ese plan.
4. Plan programado no se aplica:
   - revisar `tenant_schema` y conectividad API externa.
