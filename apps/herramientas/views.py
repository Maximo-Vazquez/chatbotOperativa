from django.contrib.auth.decorators import login_required
from django.http import JsonResponse

from .tools import TOOL_META


@login_required
def list_tools(request):
    """Devuelve la lista de herramientas disponibles para el frontend."""
    return JsonResponse({"tools": TOOL_META})
