from urllib.parse import quote

from django.contrib import messages
from django.contrib.auth import logout
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme

SESION_INVITADO = "es_invitado"


def _resolve_next_url(request, explicit: str | None = None) -> str:
    raw_next = (explicit or request.POST.get("next") or request.GET.get("next") or "").strip()
    if not raw_next:
        return ""
    if url_has_allowed_host_and_scheme(
        raw_next,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        return raw_next
    return ""


def _redirect_with_next(view_name: str, next_url: str):
    if not next_url:
        return redirect(view_name)
    return redirect(f"{reverse(view_name)}?next={quote(next_url, safe='')}")


def login_view(request):
    next_url = _resolve_next_url(request)
    if request.method == "POST":
        messages.error(request, "El inicio con usuario y contraseña está deshabilitado. Usá Google o invitado.")
        return _redirect_with_next("login", next_url)

    return render(
        request,
        "login_registro.html",
        {
            "active_tab": "login",
            "next_url": next_url,
        },
    )


def login_invitado_view(request):
    if request.method != "POST":
        return _redirect_with_next("login", _resolve_next_url(request))

    request.session[SESION_INVITADO] = True
    request.session["chat_provider"] = "deepseek"
    request.session["chat_model"] = "deepseek-v4-flash"
    request.session.modified = True
    return redirect(_resolve_next_url(request) or "chat_home")


def registro_view(request):
    messages.info(request, "El registro local está deshabilitado. Podés entrar con Google o como invitado.")
    return _redirect_with_next("login", _resolve_next_url(request))


def cerrar_sesion_view(request):
    logout(request)
    request.session.flush()
    return redirect("login")
