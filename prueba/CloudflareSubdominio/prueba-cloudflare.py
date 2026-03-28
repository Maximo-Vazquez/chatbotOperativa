#!/usr/bin/env python3
"""
Cliente Cloudflare (OO) para gestionar DNS records (crear / actualizar / eliminar / consultar)
mediante Cloudflare API v4.

Requisitos:
  pip install requests

Notas:
  - "Crear" NO actualiza: si existe, avisa y termina.
  - "Actualizar" es acción separada.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import requests


# =========================
# CONSTANTES (EDITA AQUÍ)
# =========================
TOKEN_API_CLOUDFLARE = "NXE57XJKAv57Ncj2bcwfIz7w1H9y7wAPJKH5NS8a"   # Recomendado: export CF_API_TOKEN y leer desde env
NOMBRE_ZONA = "bibliotecadvschaco.com"

# Defaults sugeridos para el menú (podés editarlos o cargar en runtime)
DEFAULT_SUBDOMINIO = "mftienda"              # "" o "@" => raíz
DEFAULT_TIPO_REGISTRO = "A"                  # A / CNAME / TXT / etc.
DEFAULT_CONTENIDO = "181.91.114.255"         # IP si A; hostname si CNAME

DEFAULT_TTL = 1                              # 1 = Auto
DEFAULT_PROXIED = True                       # True = nube naranja, False = solo DNS

DEFAULT_NOMBRE_COMPLETO_FORZADO: Optional[str] = None  # ej "api.tudominio.com"
# =========================
# FIN CONSTANTES
# =========================


class ErrorAPICloudflare(RuntimeError):
    """Error lógico/procesamiento sobre respuesta Cloudflare."""
    pass


@dataclass(frozen=True)
class ResultadoOperacionDNS:
    accion: str  # "creado" | "actualizado" | "eliminado" | "existente" | "no_encontrado"
    registro: Optional[Dict[str, Any]] = None
    mensaje: Optional[str] = None


class ClienteCloudflare:
    """
    Cliente OO para Cloudflare v4 API (DNS records).
    Encapsula:
      - sesión HTTP
      - headers
      - manejo de errores
      - cache de id_zona
      - CRUD DNS records
    """

    def __init__(self, token_api: str, nombre_zona: str, url_base: str = "https://api.cloudflare.com/client/v4"):
        if not token_api or token_api.startswith("TU_TOKEN") or token_api.startswith("PON_AQUI"):
            raise ValueError("token_api inválido. Configurá un token real.")
        if not nombre_zona or "." not in nombre_zona:
            raise ValueError("nombre_zona inválido. Debe ser un dominio, por ejemplo: midominio.com")

        self.token_api = token_api.strip()
        self.nombre_zona = nombre_zona.strip().lower()
        self.url_base = url_base.rstrip("/")

        self._id_zona_cache: Optional[str] = None

        self._sesion = requests.Session()
        self._sesion.headers.update({
            "Authorization": f"Bearer {self.token_api}",
            "Content-Type": "application/json",
        })

    def _solicitud(
        self,
        metodo: str,
        ruta: str,
        *,
        parametros: Optional[Dict[str, Any]] = None,
        cuerpo_json: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        url = f"{self.url_base}{ruta}"
        respuesta = self._sesion.request(
            metodo,
            url,
            params=parametros,
            json=cuerpo_json,
            timeout=timeout,
        )

        try:
            datos = respuesta.json()
        except Exception as e:
            raise ErrorAPICloudflare(
                f"Respuesta no-JSON (HTTP {respuesta.status_code}): {respuesta.text}"
            ) from e

        if not datos.get("success", False):
            errores = datos.get("errors", [])
            mensaje = "; ".join([f"{err.get('code')}: {err.get('message')}" for err in errores]) or "Error desconocido"
            raise ErrorAPICloudflare(
                f"Error Cloudflare API (HTTP {respuesta.status_code}): {mensaje}"
            )

        return datos

    def obtener_id_zona(self) -> str:
        if self._id_zona_cache:
            return self._id_zona_cache

        datos = self._solicitud("GET", "/zones", parametros={"name": self.nombre_zona})
        resultado = datos.get("result", [])

        if not resultado:
            raise ErrorAPICloudflare(f"No se encontró la zona '{self.nombre_zona}' con el token provisto.")

        self._id_zona_cache = resultado[0]["id"]
        return self._id_zona_cache

    def construir_fqdn(self, subdominio: str, nombre_completo_forzado: Optional[str] = None) -> str:
        if nombre_completo_forzado:
            return nombre_completo_forzado.strip().lower()

        sub = (subdominio or "").strip().lower()
        if sub in ("", "@"):
            return self.nombre_zona
        return f"{sub}.{self.nombre_zona}"

    def buscar_registro_dns(self, *, tipo_registro: str, nombre_fqdn: str) -> Optional[Dict[str, Any]]:
        id_zona = self.obtener_id_zona()
        datos = self._solicitud(
            "GET",
            f"/zones/{id_zona}/dns_records",
            parametros={"type": tipo_registro.upper(), "name": nombre_fqdn},
        )
        resultado = datos.get("result", [])
        return resultado[0] if resultado else None

    def listar_registros_dns(self, *, tipo_registro: Optional[str] = None, nombre_fqdn: Optional[str] = None, per_page: int = 50) -> List[Dict[str, Any]]:
        id_zona = self.obtener_id_zona()
        params: Dict[str, Any] = {"per_page": min(max(int(per_page), 5), 100)}
        if tipo_registro:
            params["type"] = tipo_registro.upper()
        if nombre_fqdn:
            params["name"] = nombre_fqdn

        datos = self._solicitud("GET", f"/zones/{id_zona}/dns_records", parametros=params)
        return datos.get("result", [])

    def crear_registro_dns(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        id_zona = self.obtener_id_zona()
        datos = self._solicitud("POST", f"/zones/{id_zona}/dns_records", cuerpo_json=payload)
        return datos["result"]

    def actualizar_registro_dns(self, id_registro: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        id_zona = self.obtener_id_zona()
        datos = self._solicitud("PUT", f"/zones/{id_zona}/dns_records/{id_registro}", cuerpo_json=payload)
        return datos["result"]

    def eliminar_registro_dns(self, id_registro: str) -> Dict[str, Any]:
        id_zona = self.obtener_id_zona()
        datos = self._solicitud("DELETE", f"/zones/{id_zona}/dns_records/{id_registro}")
        return datos.get("result", {})

    # -------------------------
    # Operaciones de negocio
    # -------------------------

    def crear_si_no_existe(
        self,
        *,
        tipo_registro: str,
        subdominio: str,
        contenido: str,
        ttl: int = 1,
        proxied: bool = False,
        nombre_completo_forzado: Optional[str] = None,
    ) -> ResultadoOperacionDNS:
        """
        Crea SOLO si no existe. Si existe, NO actualiza.
        """
        nombre_fqdn = self.construir_fqdn(subdominio, nombre_completo_forzado)
        tipo = tipo_registro.upper()

        existente = self.buscar_registro_dns(tipo_registro=tipo, nombre_fqdn=nombre_fqdn)
        if existente:
            return ResultadoOperacionDNS(
                accion="existente",
                registro=existente,
                mensaje=f"Ya existe {tipo} {nombre_fqdn}. No se realizaron cambios.",
            )

        payload: Dict[str, Any] = {
            "type": tipo,
            "name": nombre_fqdn,
            "content": contenido,
            "ttl": int(ttl),
            "proxied": bool(proxied),
        }
        creado = self.crear_registro_dns(payload)
        return ResultadoOperacionDNS(accion="creado", registro=creado)

    def actualizar_existente(
        self,
        *,
        tipo_registro: str,
        subdominio: str,
        contenido: str,
        ttl: int = 1,
        proxied: bool = False,
        nombre_completo_forzado: Optional[str] = None,
    ) -> ResultadoOperacionDNS:
        """
        Actualiza SOLO si existe. Si no existe, avisa.
        """
        nombre_fqdn = self.construir_fqdn(subdominio, nombre_completo_forzado)
        tipo = tipo_registro.upper()

        existente = self.buscar_registro_dns(tipo_registro=tipo, nombre_fqdn=nombre_fqdn)
        if not existente:
            return ResultadoOperacionDNS(
                accion="no_encontrado",
                registro=None,
                mensaje=f"No existe {tipo} {nombre_fqdn}. No se realizó actualización.",
            )

        payload: Dict[str, Any] = {
            "type": tipo,
            "name": nombre_fqdn,
            "content": contenido,
            "ttl": int(ttl),
            "proxied": bool(proxied),
        }
        actualizado = self.actualizar_registro_dns(existente["id"], payload)
        return ResultadoOperacionDNS(accion="actualizado", registro=actualizado)

    def eliminar_existente(
        self,
        *,
        tipo_registro: str,
        subdominio: str,
        nombre_completo_forzado: Optional[str] = None,
    ) -> ResultadoOperacionDNS:
        """
        Elimina SOLO si existe.
        """
        nombre_fqdn = self.construir_fqdn(subdominio, nombre_completo_forzado)
        tipo = tipo_registro.upper()

        existente = self.buscar_registro_dns(tipo_registro=tipo, nombre_fqdn=nombre_fqdn)
        if not existente:
            return ResultadoOperacionDNS(
                accion="no_encontrado",
                mensaje=f"No existe {tipo} {nombre_fqdn}. No se eliminó nada.",
            )

        self.eliminar_registro_dns(existente["id"])
        return ResultadoOperacionDNS(accion="eliminado", registro=existente)

    def consultar(
        self,
        *,
        tipo_registro: str,
        subdominio: str,
        nombre_completo_forzado: Optional[str] = None,
    ) -> ResultadoOperacionDNS:
        nombre_fqdn = self.construir_fqdn(subdominio, nombre_completo_forzado)
        tipo = tipo_registro.upper()
        existente = self.buscar_registro_dns(tipo_registro=tipo, nombre_fqdn=nombre_fqdn)
        if not existente:
            return ResultadoOperacionDNS(accion="no_encontrado", mensaje=f"No existe {tipo} {nombre_fqdn}.")
        return ResultadoOperacionDNS(accion="existente", registro=existente)


# -------------------------
# UI / Menú CLI
# -------------------------

def _prompt(texto: str, default: Optional[str] = None) -> str:
    if default is None:
        return input(f"{texto}: ").strip()
    val = input(f"{texto} [{default}]: ").strip()
    return val if val else str(default)


def _prompt_bool(texto: str, default: bool) -> bool:
    d = "s" if default else "n"
    val = input(f"{texto} (s/n) [{d}]: ").strip().lower()
    if not val:
        return default
    return val in ("s", "si", "sí", "y", "yes", "true", "1")


def _prompt_int(texto: str, default: int) -> int:
    val = input(f"{texto} [{default}]: ").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        print("Valor inválido, se usa el default.")
        return default


def _armar_parametros_basicos() -> Tuple[str, str, str, int, bool, Optional[str]]:
    tipo = _prompt("Tipo de registro (A/CNAME/TXT/etc.)", DEFAULT_TIPO_REGISTRO).upper()
    sub = _prompt('Subdominio ("" o "@" para raíz)', DEFAULT_SUBDOMINIO)
    nombre_forzado = _prompt("Nombre completo forzado (FQDN) (vacío para no usar)", DEFAULT_NOMBRE_COMPLETO_FORZADO or "")
    nombre_forzado = nombre_forzado.strip() or None

    contenido = _prompt("Contenido (IP si A, hostname si CNAME, etc.)", DEFAULT_CONTENIDO)
    ttl = _prompt_int("TTL (1=Auto)", DEFAULT_TTL)
    proxied = _prompt_bool("Proxied (nube naranja)", DEFAULT_PROXIED)

    return tipo, sub, contenido, ttl, proxied, nombre_forzado


def _imprimir_registro(reg: Dict[str, Any]) -> None:
    print(f"  - ID: {reg.get('id')}")
    print(f"  - Tipo: {reg.get('type')}")
    print(f"  - Nombre: {reg.get('name')}")
    print(f"  - Content: {reg.get('content')}")
    print(f"  - TTL: {reg.get('ttl')}")
    if "proxied" in reg:
        print(f"  - Proxied: {reg.get('proxied')}")


def main() -> int:
    try:
        cliente = ClienteCloudflare(
            token_api=TOKEN_API_CLOUDFLARE,
            nombre_zona=NOMBRE_ZONA,
        )
    except Exception as e:
        print(f"FALLO inicializando cliente: {e}")
        return 1

    while True:
        print("\n=== Cloudflare DNS Manager ===")
        print(f"Zona: {NOMBRE_ZONA}")
        print("1) Crear registro (NO actualiza si existe)")
        print("2) Actualizar registro (solo si existe)")
        print("3) Eliminar registro (solo si existe)")
        print("4) Consultar registro (tipo+nombre)")
        print("5) Listar registros (filtro opcional)")
        print("0) Salir")

        opcion = input("Elegí una opción: ").strip()

        try:
            if opcion == "1":
                tipo, sub, contenido, ttl, proxied, nombre_forzado = _armar_parametros_basicos()
                res = cliente.crear_si_no_existe(
                    tipo_registro=tipo,
                    subdominio=sub,
                    contenido=contenido,
                    ttl=ttl,
                    proxied=proxied,
                    nombre_completo_forzado=nombre_forzado,
                )
                if res.mensaje:
                    print(res.mensaje)
                print(f"Resultado: {res.accion.upper()}")
                if res.registro:
                    _imprimir_registro(res.registro)

            elif opcion == "2":
                tipo, sub, contenido, ttl, proxied, nombre_forzado = _armar_parametros_basicos()
                res = cliente.actualizar_existente(
                    tipo_registro=tipo,
                    subdominio=sub,
                    contenido=contenido,
                    ttl=ttl,
                    proxied=proxied,
                    nombre_completo_forzado=nombre_forzado,
                )
                if res.mensaje:
                    print(res.mensaje)
                print(f"Resultado: {res.accion.upper()}")
                if res.registro:
                    _imprimir_registro(res.registro)

            elif opcion == "3":
                tipo = _prompt("Tipo de registro (A/CNAME/TXT/etc.)", DEFAULT_TIPO_REGISTRO).upper()
                sub = _prompt('Subdominio ("" o "@" para raíz)', DEFAULT_SUBDOMINIO)
                nombre_forzado = _prompt("Nombre completo forzado (FQDN) (vacío para no usar)", DEFAULT_NOMBRE_COMPLETO_FORZADO or "")
                nombre_forzado = nombre_forzado.strip() or None

                res = cliente.eliminar_existente(
                    tipo_registro=tipo,
                    subdominio=sub,
                    nombre_completo_forzado=nombre_forzado,
                )
                if res.mensaje:
                    print(res.mensaje)
                print(f"Resultado: {res.accion.upper()}")
                if res.registro:
                    print("Se eliminó el siguiente registro:")
                    _imprimir_registro(res.registro)

            elif opcion == "4":
                tipo = _prompt("Tipo de registro (A/CNAME/TXT/etc.)", DEFAULT_TIPO_REGISTRO).upper()
                sub = _prompt('Subdominio ("" o "@" para raíz)', DEFAULT_SUBDOMINIO)
                nombre_forzado = _prompt("Nombre completo forzado (FQDN) (vacío para no usar)", DEFAULT_NOMBRE_COMPLETO_FORZADO or "")
                nombre_forzado = nombre_forzado.strip() or None

                res = cliente.consultar(
                    tipo_registro=tipo,
                    subdominio=sub,
                    nombre_completo_forzado=nombre_forzado,
                )
                if res.mensaje:
                    print(res.mensaje)
                print(f"Resultado: {res.accion.upper()}")
                if res.registro:
                    _imprimir_registro(res.registro)

            elif opcion == "5":
                tipo = _prompt("Filtrar por tipo (vacío = sin filtro)", "").upper().strip() or None
                nombre = _prompt("Filtrar por nombre FQDN exacto (vacío = sin filtro)", "").strip().lower() or None
                regs = cliente.listar_registros_dns(tipo_registro=tipo, nombre_fqdn=nombre, per_page=50)
                print(f"Se encontraron {len(regs)} registros:")
                for r in regs:
                    print(f"- {r.get('type')} {r.get('name')} -> {r.get('content')} (ttl={r.get('ttl')}, proxied={r.get('proxied', 'n/a')})")

            elif opcion == "0":
                print("Saliendo.")
                return 0

            else:
                print("Opción inválida.")

        except (ValueError, ErrorAPICloudflare) as e:
            print(f"FALLO: {e}")
        except requests.RequestException as e:
            print(f"FALLO de red: {e}")

    # unreachable
    # return 0


if __name__ == "__main__":
    raise SystemExit(main())
