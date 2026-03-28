import subprocess
import random
import os

# Ruta al ejecutable de MiniUPnPc
upnpc_path = r"D:\Users\MAXIMO\Documents\descargas\upnpc-exe-win32-20220515\upnp.exe"

# Dirección IP local (cámbiala según tu configuración)
local_ip = "192.168.0.11"  # Reemplaza esto con tu dirección IP local


random_port = 8000

# Comando para abrir el puerto
open_port_command = [upnpc_path, "-a", local_ip, str(random_port), str(random_port), "TCP"]

try:
    # Ejecutar el comando para abrir el puerto
    subprocess.run(open_port_command, check=True)
    print(f"Puerto {random_port} abierto en el router.")

    # Verificar el mapeo
    check_port_command = [upnpc_path, "-l"]
    result = subprocess.run(check_port_command, capture_output=True, text=True, check=True)
    
    print("Lista de mapeos de puertos:")
    print(result.stdout)
    
except subprocess.CalledProcessError as e:
    print("Error al abrir el puerto:")
    print(e.stderr)
