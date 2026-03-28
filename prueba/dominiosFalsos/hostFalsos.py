#!/usr/bin/env python3
import sys
import os
import platform
from textwrap import dedent

# ⚙️ EDITÁ SOLO ESTA PARTE
DOMAINS = [
    "max.local",
    "max2.local",
    "max3.local",
    "max4.local",
]

IP = "127.0.0.1"

MARKER_START = "# >>> DJANGO TENANTS DOMAINS (AUTO-GENERATED) >>>"
MARKER_END = "# <<< DJANGO TENANTS DOMAINS (AUTO-GENERATED) <<<"


def get_hosts_path():
    system = platform.system().lower()
    if "windows" in system:
        return r"C:\Windows\System32\drivers\etc\hosts"
    # Linux / Mac / otros Unix
    return "/etc/hosts"


def build_block():
    if not DOMAINS:
        return ""

    lines = [MARKER_START]
    for domain in DOMAINS:
        lines.append(f"{IP}\t{domain}")
    lines.append(MARKER_END)
    return "\n".join(lines) + "\n"


def remove_old_block(content: str) -> str:
    """
    Elimina el bloque anterior entre MARKER_START y MARKER_END (si existe).
    """
    start_idx = content.find(MARKER_START)
    if start_idx == -1:
        # No hay bloque previo
        return content

    end_idx = content.find(MARKER_END, start_idx)
    if end_idx == -1:
        # Marcador de inicio sin fin: por las dudas, no tocamos nada
        return content

    # Incluir el salto de línea después del marcador final
    end_idx = content.find("\n", end_idx)
    if end_idx == -1:
        # Marcador final hasta el final del archivo
        new_content = content[:start_idx]
    else:
        new_content = content[:start_idx] + content[end_idx + 1 :]

    # Limpieza de líneas en blanco duplicadas
    while "\n\n\n" in new_content:
        new_content = new_content.replace("\n\n\n", "\n\n")

    return new_content.rstrip() + "\n"


def apply_hosts():
    hosts_path = get_hosts_path()
    print(f"Usando archivo hosts: {hosts_path}")

    if not os.path.exists(hosts_path):
        print("⚠️ No se encontró el archivo hosts.")
        sys.exit(1)

    with open(hosts_path, "r", encoding="utf-8", errors="ignore") as f:
        original = f.read()

    without_block = remove_old_block(original)
    new_block = build_block()

    if not new_block.strip():
        # Si no hay dominios, solo dejamos el archivo sin el bloque
        final_content = without_block
        action = "Se eliminó el bloque de tenants (no hay dominios definidos)."
    else:
        final_content = without_block.rstrip() + "\n\n" + new_block
        action = "Se actualizó/agregó el bloque de tenants."

    with open(hosts_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(final_content)

    print("✅ Listo.")
    print(action)
    if DOMAINS:
        print("Dominios actuales:")
        for d in DOMAINS:
            print(f"  - {d}")


def show_help():
    print(
        dedent(
            f"""
            Uso:
              python {os.path.basename(__file__)} apply
                Aplica cambios al archivo hosts usando la lista DOMAINS.

            Pasos típicos:
              1) Editá la lista DOMAINS en el script.
              2) Ejecutá como administrador:
                 - Linux/Mac: sudo python {os.path.basename(__file__)} apply
                 - Windows: abrir consola como admin y ejecutar:
                     python {os.path.basename(__file__)} apply
            """
        )
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_help()
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "apply":
        apply_hosts()
    else:
        show_help()
