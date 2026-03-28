"""Cliente HTTP para administrar suscripciones de tenants vía los endpoints públicos."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


_UNSET = object()


class SubscriptionAPIClientError(RuntimeError):
    """Error generado cuando el servidor devuelve un resultado inesperado."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class Subscription:
    """Representa el estado de suscripción de un tenant."""

    schema: str
    tenant: str
    plan: str
    plan_label: str
    active: bool
    valid: bool
    end_date: Optional[_dt.datetime]
    external_id: Optional[str]
    features: Dict[str, bool]
    domains: List[str]

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Subscription":
        """Construye una instancia a partir de los datos devueltos por la API."""

        raw_end_date = data.get("fecha_fin")
        parsed_end_date: Optional[_dt.datetime]
        if raw_end_date:
            try:
                parsed_end_date = _dt.datetime.fromisoformat(raw_end_date)
            except ValueError:
                parsed_end_date = None
        else:
            parsed_end_date = None

        raw_domains = data.get("dominios") or data.get("domains") or []
        domains: List[str]
        if isinstance(raw_domains, (list, tuple, set)):
            domains = [str(item).strip() for item in raw_domains if str(item).strip()]
        elif isinstance(raw_domains, str):
            domain = raw_domains.strip()
            domains = [domain] if domain else []
        else:
            domains = []

        primary_domain = data.get("dominio") or data.get("domain") or data.get("tenant_domain")
        if primary_domain:
            dominio_normalizado = str(primary_domain).strip()
            if dominio_normalizado:
                domains = [dominio_normalizado] + [d for d in domains if d != dominio_normalizado]

        return cls(
            schema=data.get("esquema", ""),
            tenant=data.get("inquilino", ""),
            plan=data.get("plan", ""),
            plan_label=data.get("plan_etiqueta", ""),
            active=bool(data.get("activo", False)),
            valid=bool(data.get("vigente", False)),
            end_date=parsed_end_date,
            external_id=data.get("suscripcion_id_externa"),
            features=data.get("caracteristicas", {}),
            domains=domains,
        )

    def as_dict(self) -> Dict[str, Any]:
        """Devuelve la representación serializada útil para depuración."""

        return {
            "schema": self.schema,
            "tenant": self.tenant,
            "plan": self.plan,
            "plan_label": self.plan_label,
            "active": self.active,
            "valid": self.valid,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "external_id": self.external_id,
            "features": self.features,
            "domains": self.domains,
        }


class SubscriptionAPIClient:
    """Pequeño cliente REST para interactuar con los endpoints de suscripciones."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: int = 10,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not base_url:
            raise ValueError("'base_url' es obligatorio")
        if not api_key:
            raise ValueError("'api_key' es obligatorio")

        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._session = session or requests.Session()

    # ------------------------------------------------------------------
    # Utilidades internas
    def _compose_url(self, *parts: str) -> str:
        segmentos = [self._base_url] + [segment.strip("/") for segment in parts if segment]
        return "/".join(segmentos) + "/"

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> Dict[str, Any]:
        headers = kwargs.pop("headers", {})
        headers.setdefault("X-API-KEY", self._api_key)
        headers.setdefault("Accept", "application/json")
        if "json" in kwargs:
            headers.setdefault("Content-Type", "application/json")
        response = self._session.request(
            method,
            endpoint,
            timeout=self._timeout,
            headers=headers,
            **kwargs,
        )
        if response.status_code >= 400:
            try:
                payload = response.json()
                detail = payload.get("detalle") or payload
            except ValueError:
                detail = response.text or "Error desconocido"
            raise SubscriptionAPIClientError(str(detail), status_code=response.status_code)
        try:
            return response.json()
        except ValueError as exc:
            raise SubscriptionAPIClientError("Respuesta JSON inválida del servidor") from exc

    # ------------------------------------------------------------------
    # Operaciones públicas
    def list_subscriptions(self) -> List[Subscription]:
        """Obtiene la lista completa de suscripciones disponibles."""

        url = self._compose_url("api", "suscripciones")
        payload = self._request("GET", url)
        data = payload.get("suscripciones", []) if isinstance(payload, dict) else []
        return [Subscription.from_json(item) for item in data]

    def get_subscription(self, schema_name: str) -> Subscription:
        """Devuelve la suscripción asociada al esquema indicado."""

        if not schema_name:
            raise ValueError("'schema_name' es obligatorio")
        url = self._compose_url("api", "suscripciones", schema_name)
        payload = self._request("GET", url)
        return Subscription.from_json(payload)

    def update_subscription(
        self,
        schema_name: str,
        *,
        plan: Optional[str] | object = _UNSET,
        active: Optional[bool] | object = _UNSET,
        end_date: Optional[str] | object = _UNSET,
        external_id: Optional[str] | object = _UNSET,
    ) -> Subscription:
        """Envía un PATCH para modificar la suscripción del tenant indicado."""

        if not schema_name:
            raise ValueError("'schema_name' es obligatorio")

        body: Dict[str, Any] = {}
        if plan is not _UNSET:
            body["plan"] = plan
        if active is not _UNSET:
            body["activo"] = active
        if end_date is not _UNSET:
            body["fecha_fin"] = end_date
        if external_id is not _UNSET:
            body["suscripcion_id_externa"] = external_id

        if not body:
            raise ValueError("Debe indicarse al menos un campo para actualizar")

        url = self._compose_url("api", "suscripciones", schema_name)
        payload = self._request("PATCH", url, json=body)
        return Subscription.from_json(payload)

    # ------------------------------------------------------------------
    # Gestión de recursos
    def close(self) -> None:
        """Cierra la sesión HTTP subyacente."""

        self._session.close()

    def __enter__(self) -> "SubscriptionAPIClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def describe_subscription(subscription: Subscription) -> str:
    """Genera un resumen legible de la suscripción."""

    end_date = subscription.end_date.isoformat() if subscription.end_date else "Sin fecha"
    feature_summary = ", ".join(
        sorted(f for f, enabled in subscription.features.items() if enabled)
    )
    if not feature_summary:
        feature_summary = "Sin características habilitadas"

    estado = "Vigente" if subscription.valid else "Bloqueada"
    dominios = ", ".join(subscription.domains) if subscription.domains else "—"
    return (
        f"Tenant: {subscription.tenant} ({subscription.schema})\n"
        f"Plan: {subscription.plan_label} [{subscription.plan}]\n"
        f"Activo: {'Sí' if subscription.active else 'No'}\n"
        f"Estado: {estado}\n"
        f"Vence: {end_date}\n"
        f"ID externa: {subscription.external_id or '—'}\n"
        f"Dominios: {dominios}\n"
        f"Características habilitadas: {feature_summary}"
    )


__all__ = [
    "SubscriptionAPIClient",
    "SubscriptionAPIClientError",
    "Subscription",
    "describe_subscription",
]
