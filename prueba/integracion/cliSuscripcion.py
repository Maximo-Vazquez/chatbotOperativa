"""Interfaz de línea de comandos para gestionar suscripciones de tenants."""

from __future__ import annotations

import sys
from typing import List, Optional

# ==== CONFIGURACIÓN FIJA (EDITÁ ACÁ) =========================================
BASE_URL = "http://127.0.0.1:8001"      # <-- Cambiá por tu URL
API_KEY  = "32323232-3232-4323-8432-323432343234"  # <-- Pegá tu API key
TIMEOUT  = 10                           # segundos
# =============================================================================

# Import relativo (ejecutar con: python -m utils.subscription_cli)
from .suscripcion_api_client import (
    Subscription,
    SubscriptionAPIClient,
    SubscriptionAPIClientError,
    describe_subscription,
)


def _print_title(title: str) -> None:
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))


def _prompt(text: str, *, default: Optional[str] = None) -> str:
    mensaje = f"{text}"
    if default is not None:
        mensaje += f" [{default}]"
    mensaje += ": "
    respuesta = input(mensaje).strip()
    if not respuesta and default is not None:
        return default
    return respuesta


def _prompt_yes_no(text: str, default: bool) -> bool:
    default_text = "s" if default else "n"
    respuesta = _prompt(f"{text} (s/n)", default=default_text).lower()
    return respuesta in {"s", "si", "sí", "y", "yes"}


def _seleccionar_tenant(subscriptions: List[Subscription]) -> Optional[Subscription]:
    if not subscriptions:
        print("No hay tenants registrados.")
        return None

    print("\nTenants disponibles:")
    for idx, subscription in enumerate(subscriptions, start=1):
        estado = "Vigente" if subscription.valid else "Bloqueada"
        dominios = ", ".join(subscription.domains) if subscription.domains else "—"
        print(
            f"  {idx}. {subscription.tenant} ({subscription.schema}) - {subscription.plan_label} [{estado}]"
            f" | Dominios: {dominios}"
        )

    while True:
        eleccion = _prompt("Seleccione un tenant por número (o presione Enter para cancelar)")
        if not eleccion:
            return None
        try:
            indice = int(eleccion)
        except ValueError:
            print("Ingrese un número válido.")
            continue
        if 1 <= indice <= len(subscriptions):
            return subscriptions[indice - 1]
        print("Opción fuera de rango. Intente nuevamente.")


def _mostrar_detalle(client: SubscriptionAPIClient, schema: str) -> None:
    try:
        subscription = client.get_subscription(schema)
    except SubscriptionAPIClientError as error:
        print(f"Error al obtener la suscripción: {error}")
        return

    _print_title(f"Suscripción de {subscription.tenant}")
    print(describe_subscription(subscription))


def _cambiar_plan(client: SubscriptionAPIClient, schema: str) -> None:
    nuevo_plan = _prompt("Ingrese el nuevo plan (inicio, plus, pro)").lower()
    if not nuevo_plan:
        print("No se ingresó ningún plan. Operación cancelada.")
        return
    try:
        subscription = client.update_subscription(schema, plan=nuevo_plan)
    except (SubscriptionAPIClientError, ValueError) as error:
        print(f"No fue posible actualizar el plan: {error}")
        return
    print("Plan actualizado correctamente.")
    print(describe_subscription(subscription))


def _actualizar_estado_activo(client: SubscriptionAPIClient, schema: str) -> None:
    activar = _prompt_yes_no("¿Desea marcar la suscripción como activa?", default=True)
    try:
        subscription = client.update_subscription(schema, active=activar)
    except SubscriptionAPIClientError as error:
        print(f"No fue posible actualizar el estado: {error}")
        return
    print("Estado de actividad actualizado.")
    print(describe_subscription(subscription))


def _actualizar_fecha_fin(client: SubscriptionAPIClient, schema: str) -> None:
    print("Ingrese una fecha en formato ISO-8601 (ej: 2024-12-31T23:59:59) o deje vacío para limpiar.")
    fecha = _prompt("Fecha de finalización", default="")
    if not fecha:
        fecha = None
    try:
        subscription = client.update_subscription(schema, end_date=fecha)
    except SubscriptionAPIClientError as error:
        print(f"No fue posible actualizar la fecha: {error}")
        return
    print("Fecha de finalización actualizada.")
    print(describe_subscription(subscription))


def _actualizar_id_externa(client: SubscriptionAPIClient, schema: str) -> None:
    print("Ingrese el identificador externo (vacío para eliminarlo).")
    nuevo_id = _prompt("ID externo", default="")
    if nuevo_id == "":
        nuevo_id = None
    try:
        subscription = client.update_subscription(schema, external_id=nuevo_id)
    except SubscriptionAPIClientError as error:
        print(f"No fue posible actualizar el identificador: {error}")
        return
    print("Identificador externo actualizado.")
    print(describe_subscription(subscription))


def _mostrar_menu() -> None:
    print(
        "\nAcciones disponibles:\n"
        "  1. Listar suscripciones\n"
        "  2. Ver detalle de un tenant\n"
        "  3. Cambiar plan de un tenant\n"
        "  4. Cambiar estado activo/inactivo\n"
        "  5. Actualizar fecha de finalización\n"
        "  6. Actualizar ID externo\n"
        "  7. Salir"
    )


def run_cli(client: SubscriptionAPIClient) -> None:
    suscripciones_cache: List[Subscription] = []

    while True:
        _mostrar_menu()
        opcion = _prompt("Seleccione una opción", default="1")

        if opcion == "1":
            try:
                suscripciones_cache = client.list_subscriptions()
            except SubscriptionAPIClientError as error:
                print(f"No fue posible obtener las suscripciones: {error}")
                continue
            _print_title("Suscripciones registradas")
            if not suscripciones_cache:
                print("No hay tenants registrados.")
            else:
                for subscription in suscripciones_cache:
                    estado = "Vigente" if subscription.valid else "Bloqueada"
                    dominios = ", ".join(subscription.domains) if subscription.domains else "—"
                    print(
                        f"- {subscription.tenant} ({subscription.schema}) => {subscription.plan_label}"
                        f" | Activo: {'Sí' if subscription.active else 'No'} | {estado} | Dominios: {dominios}"
                    )
            continue

        if opcion == "7":
            print("Hasta luego.")
            break

        if not suscripciones_cache:
            try:
                suscripciones_cache = client.list_subscriptions()
            except SubscriptionAPIClientError as error:
                print(f"No fue posible obtener las suscripciones: {error}")
                continue

        seleccionado = _seleccionar_tenant(suscripciones_cache)
        if not seleccionado:
            continue
        schema = seleccionado.schema

        if opcion == "2":
            _mostrar_detalle(client, schema)
        elif opcion == "3":
            _cambiar_plan(client, schema)
        elif opcion == "4":
            _actualizar_estado_activo(client, schema)
        elif opcion == "5":
            _actualizar_fecha_fin(client, schema)
        elif opcion == "6":
            _actualizar_id_externa(client, schema)
        else:
            print("Opción no reconocida.")


def main() -> int:
    # Usamos directamente las constantes definidas arriba
    try:
        with SubscriptionAPIClient(BASE_URL, API_KEY, timeout=TIMEOUT) as client:
            run_cli(client)
    except ValueError as error:
        print(f"Error de configuración: {error}")
        return 1
    except SubscriptionAPIClientError as error:
        print(f"Error del servidor: {error}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupción por parte del usuario.")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
