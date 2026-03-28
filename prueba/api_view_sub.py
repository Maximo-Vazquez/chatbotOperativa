"""Endpoints para gestionar la suscripción de cada tenant."""

from __future__ import annotations

from typing import Any, Dict, List

from django.http import HttpRequest, JsonResponse
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from django.utils.timezone import make_aware
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from django_tenants.utils import schema_context

from apps.tenants.api_utils import (
    cargar_json,
    error_json,
    forzar_esquema_publico,
    requiere_api_key,
)
from apps.tenants.models import Client

from .models import Suscripcion


def _analizar_fecha_hora(valor: str | None):
    """Convierte texto ISO-8601 en un datetime consciente de zona horaria."""

    if not valor:
        return None
    fecha_procesada = parse_datetime(valor)
    if fecha_procesada is None:
        raise ValueError("El campo 'fecha_fin' debe tener formato ISO-8601.")
    if timezone.is_naive(fecha_procesada):
        fecha_procesada = make_aware(fecha_procesada, timezone.get_current_timezone())
    return fecha_procesada


def _obtener_dominios_tenant(inquilino: Client) -> List[str]:
    """Devuelve los dominios asociados al tenant ordenados alfabéticamente."""

    dominios: List[str] = []
    for atributo in ("domains", "domain_set"):
        related_manager = getattr(inquilino, atributo, None)
        if related_manager is None:
            continue
        try:
            candidatos = related_manager.all()
        except Exception:  # pragma: no cover - manager inesperado
            continue
        dominios = [dom.domain.strip() for dom in candidatos if getattr(dom, "domain", "").strip()]
        break

    if not dominios:
        posible_dominio = (
            getattr(inquilino, "domain", None)
            or getattr(inquilino, "domain_url", None)
            or getattr(inquilino, "dominio", None)
        )
        if posible_dominio:
            dominio_limpio = str(posible_dominio).strip()
            if dominio_limpio:
                dominios = [dominio_limpio]

    return sorted(dict.fromkeys(dominios))


def _serializar_suscripcion(suscripcion: Suscripcion, inquilino: Client) -> Dict[str, Any]:
    """Prepara un diccionario con datos del plan y sus características activas."""

    caracteristicas = suscripcion.caracteristicas_activas()
    dominios = _obtener_dominios_tenant(inquilino)
    return {
        "esquema": inquilino.schema_name,
        "inquilino": inquilino.name,
        "plan": suscripcion.tipo,
        "plan_etiqueta": suscripcion.get_tipo_display(),
        "activo": suscripcion.activo,
        "vigente": suscripcion.esta_vigente(),
        "fecha_fin": suscripcion.fecha_fin.isoformat() if suscripcion.fecha_fin else None,
        "suscripcion_id_externa": suscripcion.suscripcion_id_externa,
        "caracteristicas": caracteristicas,
        "dominios": dominios,
        "dominio": dominios[0] if dominios else None,
    }


def _serializar_todas_las_suscripciones() -> List[Dict[str, Any]]:
    """Genera una lista con el estado de suscripción de cada tenant registrado."""

    suscripciones: List[Dict[str, Any]] = []
    for inquilino in Client.objects.all().order_by("name"):
        with schema_context(inquilino.schema_name):
            suscripcion = Suscripcion.load()
            suscripciones.append(_serializar_suscripcion(suscripcion, inquilino))
    return suscripciones
def _obtener_inquilino(nombre_esquema: str) -> Client | None:
    """Busca el tenant asociado al esquema solicitado."""

    try:
        return Client.objects.get(schema_name=nombre_esquema)
    except Client.DoesNotExist:
        return None


@csrf_exempt
@require_http_methods(["GET"])
@requiere_api_key
@forzar_esquema_publico
def listar_suscripciones(request: HttpRequest) -> JsonResponse:
    """Devuelve el estado de todas las suscripciones disponibles."""

    datos = _serializar_todas_las_suscripciones()
    return JsonResponse({"suscripciones": datos})


@csrf_exempt
@require_http_methods(["GET", "PATCH"])
@requiere_api_key
@forzar_esquema_publico
def detalle_suscripcion(request: HttpRequest, nombre_esquema: str) -> JsonResponse:
    """Permite consultar o actualizar el plan asignado a un tenant."""

    inquilino = _obtener_inquilino(nombre_esquema)
    if inquilino is None:
        return error_json("Inquilino no encontrado.", status=404)

    if request.method == "GET":
        with schema_context(nombre_esquema):
            suscripcion = Suscripcion.load()
            return JsonResponse(_serializar_suscripcion(suscripcion, inquilino))

    datos = cargar_json(request)
    if isinstance(datos, JsonResponse):
        return datos

    actualizaciones: list[str] = []

    with schema_context(nombre_esquema):
        suscripcion = Suscripcion.load()

        if "plan" in datos:
            plan_normalizado = Suscripcion.normalizar_plan(datos.get("plan"))
            if not plan_normalizado:
                return error_json("Valor de 'plan' inválido. Use inicio, plus o pro.")
            suscripcion.tipo = plan_normalizado
            actualizaciones.append("tipo")

        if "activo" in datos:
            suscripcion.activo = bool(datos.get("activo"))
            actualizaciones.append("activo")

        if "fecha_fin" in datos:
            try:
                suscripcion.fecha_fin = _analizar_fecha_hora(datos.get("fecha_fin"))
            except ValueError as exc:
                return error_json(str(exc))
            actualizaciones.append("fecha_fin")

        if "suscripcion_id_externa" in datos:
            suscripcion.suscripcion_id_externa = datos.get("suscripcion_id_externa") or None
            actualizaciones.append("suscripcion_id_externa")

        if not actualizaciones:
            return error_json("No se enviaron campos para actualizar.", status=400)

        suscripcion.save(update_fields=list(dict.fromkeys(actualizaciones)))

        return JsonResponse(_serializar_suscripcion(suscripcion, inquilino))

