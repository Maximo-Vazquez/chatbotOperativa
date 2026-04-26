from django.http import JsonResponse

from .tools import TOOL_META

SESION_INVITADO = "es_invitado"


def list_tools(request):
    """Devuelve la lista de herramientas disponibles para el frontend.
    Accesible para usuarios autenticados o invitados con sesión activa."""
    es_invitado = bool(request.session.get(SESION_INVITADO))
    if not (request.user.is_authenticated or es_invitado):
        return JsonResponse({"tools": {}}, status=401)
    return JsonResponse({"tools": TOOL_META})
