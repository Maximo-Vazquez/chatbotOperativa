# Maquina De Estados De Suscripciones

## 1. Objetivo
Centralizar el cambio de `Suscripcion.estado` para evitar estados incoherentes entre:
- vistas
- tareas programadas
- procesamiento de pagos/webhooks

La implementacion vive en `apps/administracion/services/subscription_state_machine.py`.

## 2. Estados Canonicos
La maquina maneja 4 estados:
- `PreparadoSoftware`
- `activa`
- `caducada`
- `cancelada`

Fuente: `apps/administracion/services/subscription_state_machine.py:9`.

## 3. Normalizacion De Estado
La funcion `normalizar_estado_suscripcion()` convierte entradas "sucias" a un estado canonico:
- `None` o vacio -> `PreparadoSoftware`
- alias `preparandosoftware` -> `PreparadoSoftware`
- valores desconocidos -> `PreparadoSoftware` (fallback seguro)

Fuente: `apps/administracion/services/subscription_state_machine.py:43`.

## 4. Reglas De Transicion
Matriz actual (`_TRANSICIONES_PERMITIDAS`):

- desde `PreparadoSoftware`: puede ir a `PreparadoSoftware`, `activa`, `cancelada`, `caducada`
- desde `activa`: puede ir a `activa`, `cancelada`, `caducada`
- desde `caducada`: puede ir a `caducada`, `activa`, `cancelada`
- desde `cancelada`: puede ir a `cancelada`, `activa`

Esto bloquea saltos invalidos, por ejemplo:
- `cancelada -> caducada` (no permitido)

Fuente: `apps/administracion/services/subscription_state_machine.py:29`.

## 5. API Del Servicio
### `puede_transicionar_estado(actual, nuevo)`
Devuelve `True/False` segun la matriz.

### `transicionar_suscripcion(suscripcion, estado_nuevo, motivo="", extra_update_fields=None, force=False)`
Funcion principal para persistir transiciones.

Comportamiento:
- normaliza estado actual y destino
- valida transicion (si `force=False`)
- si es invalida, lanza `ErrorTransicionEstado`
- actualiza `estado` y guarda con `update_fields`
- registra log estructurado:
  - `subscription_state_transition`
  - `suscripcion_id`, `from`, `to`, `motivo`

Parametro importante:
- `extra_update_fields`: permite guardar otros campos en el mismo `save()` sin perder atomicidad de datos.
- `force=True`: ignora la validacion de la matriz (usar solo en operaciones administrativas controladas).

Fuente: `apps/administracion/services/subscription_state_machine.py:61`.

### `aplicar_expiracion_por_fecha(suscripcion, referencia=None, motivo="expiracion_automatica")`
Regla de expiracion:
- si `fecha_fin` no existe: no hace nada
- si `fecha_fin >= referencia`: no hace nada
- si ya esta `caducada` o `cancelada`: no hace nada
- si vencio y estaba activa/preparada: transiciona a `caducada`

Fuente: `apps/administracion/services/subscription_state_machine.py:95`.

## 6. Donde Se Usa Hoy
## 6.1 Tarea de expiracion (Celery)
`expirar_suscripciones()` ahora llama:
- `aplicar_expiracion_por_fecha(...)`

Antes se hacia `suscripcion.estado = "caducada"` directo.

Fuente: `apps/administracion/tasks.py:27`.

## 6.2 Flujo de creacion/extension por pago
En `utils/CrearSucripcion.py`:
- al crear suscripcion, se inicializa en `ESTADO_ACTIVA`
- en extension con pago, se evalua `estado_objetivo`
- si corresponde y la transicion es valida, se aplica con `transicionar_suscripcion(...)`

Fuente: `utils/CrearSucripcion.py:788` y `utils/CrearSucripcion.py:829`.

## 6.3 Admin de pedidos/suscripciones
En administracion:
- activacion manual usa `transicionar_suscripcion(..., ESTADO_ACTIVA, force=True)`
- extension manual usa `transicionar_suscripcion(..., ESTADO_ACTIVA, force=True)`

Fuente: `apps/administracion/views.py:1689` y `apps/administracion/views.py:2574`.

## 6.4 Panel de usuario (lectura coherente)
Al abrir paneles se aplica expiracion lazy para evitar mostrar activas vencidas:
- listado de suscripciones
- listado de pedidos
- detalle de pedido
- administrar suscripcion

Fuente: `apps/panelUsuarioConf/views.py:171`, `apps/panelUsuarioConf/views.py:369`, `apps/panelUsuarioConf/views.py:455`, `apps/panelUsuarioConf/views.py:508`.

## 7. Diferencia Clave: Estado Comercial vs Tenant Pausado
Pausar tenant **no cambia** `Suscripcion.estado`.

Se guarda en:
- `Suscripcion.tenant_pausado = True/False`

Interpretacion:
- `estado` responde "vigencia comercial/contractual"
- `tenant_pausado` responde "servicio operativo pausado/reanudado"

Esto evita mezclar:
- estado de facturacion/suscripcion
- estado tecnico de ejecucion del tenant

## 8. Escenarios Reales
## 8.1 Suscripcion vence por fecha
1. tarea o panel detecta `fecha_fin < now`
2. `aplicar_expiracion_por_fecha()` transiciona a `caducada`
3. queda auditado por log

## 8.2 Pago de renovacion mensual MP acreditado
1. webhook crea/actualiza pedido y pago
2. extension calcula `estado_objetivo` (normalmente `activa`)
3. aplica transicion si es valida

## 8.3 Suscripcion cancelada que vuelve a pagarse
Con matriz actual, `cancelada -> activa` esta permitido.
Eso habilita reactivacion al acreditar pago sin "parches" manuales.

## 9. Tests
Hay pruebas unitarias base en:
- `apps/administracion/tests.py`

Cobertura actual:
- normalizacion de alias
- transicion valida `cancelada -> activa`
- transicion invalida `cancelada -> caducada` (lanza error)
- expiracion por fecha (`activa -> caducada`)

## 10. Relacion Con El Fix De `desconodico`
Se corrigio typo de estado de pago en `PedidoSuscripcion`:
- modelo: `apps/administracion/models.py:461`
- migracion de datos + schema: `apps/administracion/migrations/0056_fix_pedidosuscripcion_estado_pago_typo.py`

Esto no pertenece a la maquina de estados de suscripcion, pero mejora consistencia global de estados.

## 11. Limites Actuales Y Proximo Paso
La maquina ya gobierna transiciones clave, pero todavia hay logica de ciclo/plan repartida fuera de ella (por ejemplo cambios programados de `modo_cobro` y `nivel` en `apps/administracion/tasks.py`).

Siguiente evolucion recomendada:
1. modelar tambien estados operativos (ej. "pausada") si queres que sea parte del estado comercial
2. mover transiciones de plan/modo a un servicio unico de dominio
3. agregar auditoria persistente (tabla historial de transiciones) ademas del log
4. ampliar tests de integracion con webhooks y tareas
