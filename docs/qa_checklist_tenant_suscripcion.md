# Checklist QA - Tenant API y Suscripciones

Fecha: 2026-02-13
Responsable QA: __________________
Ambiente: __________________
Build/Commit: __________________

## Leyenda
- Estado: `OK` / `FAIL` / `N/A`
- Evidencia: URL, screenshot, log, id de pedido/suscripcion, etc.

---

## 1. Preflight Tecnico

| ID | Verificacion | Estado | Evidencia | Observaciones |
|---|---|---|---|---|
| PF-01 | `python manage.py check` sin errores |  |  |  |
| PF-02 | `python manage.py makemigrations --check` sin cambios pendientes |  |  |  |
| PF-03 | `TENANT_MANAGEMENT_API_BASE_URL` configurado |  |  |  |
| PF-04 | `TENANT_MANAGEMENT_API_KEY` configurado |  |  |  |
| PF-05 | Existe promo anual (`duracion_meses=12`) para planes `inicio/plus/pro` |  |  |  |

---

## 2. Creacion Tenant (usuario_admin automatico)

| ID | Caso | Estado | Evidencia | Observaciones |
|---|---|---|---|---|
| CT-01 | Al crear tenant, request incluye `nombre/esquema/dominio/plan/pagado_hasta` |  |  |  |
| CT-02 | Request incluye `usuario_admin.email` |  |  |  |
| CT-03 | Request incluye `usuario_admin.nombre` (primera palabra de nombre o username) |  |  |  |
| CT-04 | Request incluye `usuario_admin.password` aleatoria |  |  |  |
| CT-05 | Password admin tenant NO se persiste en modelo local |  |  |  |

---

## 3. Pausar/Reanudar desde Panel Usuario

Ruta: `panel/software/administrar/<suscripcion_id>/`

| ID | Caso | Estado | Evidencia | Observaciones |
|---|---|---|---|---|
| PR-01 | Boton `Pausar` visible para suscripcion activa |  |  |  |
| PR-02 | Al pausar: POST interno correcto + llamada `POST /api/tenants/{esquema}/pause/` con `pausado=true` |  |  |  |
| PR-03 | Se actualiza `tenant_pausado=True` en DB |  |  |  |
| PR-04 | Mensaje de exito de pausa visible en UI |  |  |  |
| PR-05 | Al reanudar: llamada remota con `pausado=false` |  |  |  |
| PR-06 | Se actualiza `tenant_pausado=False` en DB |  |  |  |
| PR-07 | Si falta `tenant_schema`, se muestra error controlado y no rompe UI |  |  |  |

---

## 4. Renovacion (regla <365 dias, +12 meses, tope 2 anios)

Ruta: `panel/software/renovar/<suscripcion_id>/`

| ID | Caso | Estado | Evidencia | Observaciones |
|---|---|---|---|---|
| RN-01 | Si faltan >=365 dias: bloquea renovacion |  |  |  |
| RN-02 | Si faltan 364 dias: permite renovacion |  |  |  |
| RN-03 | Solo permite bloque de 12 meses (no periodos variables) |  |  |  |
| RN-04 | Si al renovar supera 2 anios vigentes: bloquea renovacion |  |  |  |
| RN-05 | Monto se calcula con promo anual de 12 meses del plan aplicable |  |  |  |
| RN-06 | Crea `PedidoSuscripcion` con `accion='extenderSuscripcion'`, `meses_a_pagar=12`, `estado_pago='pendiente'` |  |  |  |
| RN-07 | Redirige a flujo de pago (`pago_pedido`) |  |  |  |
| RN-08 | Tras pago aprobado, se extiende `fecha_fin` local |  |  |  |
| RN-09 | Tras pago aprobado, sincroniza remoto via `PATCH /api/tenants/{esquema}/renovar/` |  |  |  |

---

## 5. Cambio de Plan - Upgrade

Ruta: administrar suscripcion -> actualizar plan

| ID | Caso | Estado | Evidencia | Observaciones |
|---|---|---|---|---|
| UP-01 | Upgrade (ej. inicio->plus) crea pedido por diferencia prorrateada |  |  |  |
| UP-02 | El pedido queda en `estado_pago='pendiente'` |  |  |  |
| UP-03 | NO cambia `suscripcion.nivel` antes del pago |  |  |  |
| UP-04 | Tras pago aprobado, se aplica plan nuevo localmente |  |  |  |
| UP-05 | Tras pago aprobado, sincroniza remoto por `PATCH /api/tenants/{esquema}/plan/` |  |  |  |

---

## 6. Cambio de Plan - Downgrade Programado

| ID | Caso | Estado | Evidencia | Observaciones |
|---|---|---|---|---|
| DW-01 | Downgrade (ej. pro->plus) NO aplica inmediato |  |  |  |
| DW-02 | Se setea `nivel_programado` con plan destino |  |  |  |
| DW-03 | Se setea `fecha_aplicar_nivel_programado=fecha_fin actual` |  |  |  |
| DW-04 | UI muestra opcion de cancelar cambio programado |  |  |  |
| DW-05 | Al cancelar: `nivel_programado=NULL`, `fecha_aplicar_nivel_programado=NULL` |  |  |  |

---

## 7. Aplicacion automatica de plan programado

| ID | Caso | Estado | Evidencia | Observaciones |
|---|---|---|---|---|
| AP-01 | Con fecha programada en pasado, al entrar a panel se aplica plan programado |  |  |  |
| AP-02 | Se limpia programacion (`nivel_programado`, `fecha_aplicar_nivel_programado`) |  |  |  |
| AP-03 | Se intenta sincronizar remoto por endpoint de plan |  |  |  |
| AP-04 | Tambien aplica durante procesamiento de pedidos/pagos |  |  |  |

---

## 8. Compatibilidad Admin / Integracion

| ID | Caso | Estado | Evidencia | Observaciones |
|---|---|---|---|---|
| AD-01 | Panel admin suscripcion: extender funciona con API nueva |  |  |  |
| AD-02 | Panel admin suscripcion: cambiar plan funciona con API nueva |  |  |  |
| AD-03 | Pantalla despliegue/estado consulta `GET /api/tenants/{esquema}/info/` |  |  |  |
| AD-04 | No hay errores por rutas viejas `/api/suscripciones/...` |  |  |  |

---

## 9. Validacion de Datos en DB (marcar tras ejecutar)

| ID | Verificacion DB | Estado | Evidencia | Observaciones |
|---|---|---|---|---|
| DB-01 | `Suscripcion.tenant_pausado` refleja estado remoto |  |  |  |
| DB-02 | `Suscripcion.nivel` refleja plan activo real |  |  |  |
| DB-03 | `Suscripcion.nivel_programado` solo existe para downgrades pendientes |  |  |  |
| DB-04 | `PedidoSuscripcion` de upgrade/renovacion con montos y estado correctos |  |  |  |

---

## 10. Cierre QA

| Item | Resultado |
|---|---|
| Total casos OK |  |
| Total casos FAIL |  |
| Total casos N/A |  |
| Riesgo residual |  |
| Recomendacion de pase a produccion |  |

Firma QA: __________________
Fecha cierre: __________________
