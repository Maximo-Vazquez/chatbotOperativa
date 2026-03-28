import random
from datetime import timedelta
from urllib.parse import quote

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, get_user_model, login, logout
from django.core.mail import send_mail
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme

REGISTRATION_SESSION_KEY = "registro_pendiente_email"
REGISTRATION_CODE_TTL_MINUTES = 3
REGISTRATION_RESEND_COOLDOWN_SECONDS = 60


def _send_registration_code_email(*, email: str, code: str) -> None:
    from_email = (
        getattr(settings, "DEFAULT_FROM_EMAIL", "")
        or getattr(settings, "EMAIL_HOST_USER", "")
        or None
    )
    send_mail(
        "Codigo de verificacion - Registro MF Software",
        (
            f"Tu codigo de verificacion es: {code}. "
            f"Vence en {REGISTRATION_CODE_TTL_MINUTES} minutos."
        ),
        from_email,
        [email],
        fail_silently=False,
    )


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


def _pending_registration(request):
    data = request.session.get(REGISTRATION_SESSION_KEY)
    return data if isinstance(data, dict) else None


def _clear_pending_registration(request):
    request.session.pop(REGISTRATION_SESSION_KEY, None)


def _parse_iso_datetime(raw: str | None):
    if not raw:
        return None
    try:
        dt = timezone.datetime.fromisoformat(raw)
        if timezone.is_naive(dt):
            dt = timezone.make_aware(dt)
        return dt
    except Exception:
        return None


def _registration_issued_at(data: dict):
    return _parse_iso_datetime(data.get("issued_at") or data.get("timestamp"))


def _registration_last_sent_at(data: dict):
    return _parse_iso_datetime(data.get("last_sent_at") or data.get("timestamp"))


def _registration_expired(data: dict) -> bool:
    issued_at = _registration_issued_at(data)
    if not issued_at:
        return True
    return timezone.now() - issued_at > timedelta(minutes=REGISTRATION_CODE_TTL_MINUTES)


def _seconds_until_expiration(data: dict) -> int:
    issued_at = _registration_issued_at(data)
    if not issued_at:
        return 0
    expire_at = issued_at + timedelta(minutes=REGISTRATION_CODE_TTL_MINUTES)
    return max(0, int((expire_at - timezone.now()).total_seconds()))


def _seconds_until_resend(data: dict) -> int:
    sent_at = _registration_last_sent_at(data)
    if not sent_at:
        return 0
    ready_at = sent_at + timedelta(seconds=REGISTRATION_RESEND_COOLDOWN_SECONDS)
    return max(0, int((ready_at - timezone.now()).total_seconds()))


def _registration_context(request, *, next_url: str = ""):
    pending = _pending_registration(request)
    if pending and _registration_expired(pending):
        _clear_pending_registration(request)
        pending = None
    if not next_url:
        next_url = _resolve_next_url(request)
    expires_in_seconds = _seconds_until_expiration(pending) if pending else 0
    resend_in_seconds = _seconds_until_resend(pending) if pending else 0
    return {
        "active_tab": "register",
        "show_code_form": bool(pending),
        "pending_email": (pending or {}).get("email", ""),
        "code_expires_in_seconds": expires_in_seconds,
        "resend_available_in_seconds": resend_in_seconds,
        "resend_cooldown_seconds": REGISTRATION_RESEND_COOLDOWN_SECONDS,
        "next_url": next_url,
    }


def login_view(request):
    next_url = _resolve_next_url(request)

    if request.method == "POST":
        email = (request.POST.get("gmail") or "").strip().lower()
        password = request.POST.get("contrasena") or ""

        user = authenticate(request, username=email, password=password)
        if user is None:
            user_model = get_user_model()
            existing_user = user_model.objects.filter(email__iexact=email).first()
            if existing_user:
                user = authenticate(
                    request,
                    username=existing_user.get_username(),
                    password=password,
                )

        if user is not None:
            login(request, user)
            messages.success(request, "Inicio de sesion exitoso.")
            return redirect(next_url or "chat_home")

        messages.error(request, "Credenciales invalidas.")
        return render(
            request,
            "login_registro.html",
            {
                "active_tab": "login",
                "next_url": next_url,
            },
        )

    context = {
        "active_tab": request.GET.get("tab", "login"),
        "next_url": next_url,
    }
    if _pending_registration(request):
        context.update(_registration_context(request, next_url=next_url))
    return render(request, "login_registro.html", context)


def registro_view(request):
    next_url = _resolve_next_url(request)

    if request.method != "POST":
        if (request.GET.get("reset_registration") or "").strip() == "1":
            _clear_pending_registration(request)
            messages.info(request, "Puedes volver a completar tus datos de registro.")
            return render(
                request,
                "login_registro.html",
                {"active_tab": "register", "next_url": next_url},
            )

        pending = _pending_registration(request)
        if pending and _registration_expired(pending):
            _clear_pending_registration(request)
            messages.error(request, "El codigo expiro. Solicita uno nuevo.")
            pending = None

        context = (
            _registration_context(request, next_url=next_url)
            if pending
            else {"active_tab": "register", "next_url": next_url}
        )
        return render(request, "login_registro.html", context)

    action = (request.POST.get("action") or "send_code").strip().lower()

    if action == "send_code":
        email = (request.POST.get("gmail") or "").strip().lower()
        username = (request.POST.get("usuario") or "").strip()
        password = request.POST.get("contrasena") or ""
        password_confirm = request.POST.get("conf_contrasena") or ""

        if not email or not username or not password:
            messages.error(request, "Completa correo, usuario y contrasena.")
            return render(
                request,
                "login_registro.html",
                {"active_tab": "register", "next_url": next_url},
            )

        if password != password_confirm:
            messages.error(request, "Las contrasenas no coinciden.")
            return render(
                request,
                "login_registro.html",
                {"active_tab": "register", "next_url": next_url},
            )

        user_model = get_user_model()
        if user_model.objects.filter(username__iexact=username).exists():
            messages.error(request, "El nombre de usuario ya esta registrado.")
            return render(
                request,
                "login_registro.html",
                {"active_tab": "register", "next_url": next_url},
            )

        if user_model.objects.filter(email__iexact=email).exists():
            messages.error(request, "El correo electronico ya esta registrado.")
            return render(
                request,
                "login_registro.html",
                {"active_tab": "register", "next_url": next_url},
            )

        code = f"{random.randint(0, 999999):06d}"
        request.session[REGISTRATION_SESSION_KEY] = {
            "email": email,
            "username": username,
            "password": password,
            "code": code,
            "issued_at": timezone.now().isoformat(),
            "last_sent_at": timezone.now().isoformat(),
        }
        request.session.modified = True

        try:
            _send_registration_code_email(email=email, code=code)
        except Exception:
            _clear_pending_registration(request)
            messages.error(request, "No se pudo enviar el codigo. Intenta nuevamente.")
            return render(
                request,
                "login_registro.html",
                {"active_tab": "register", "next_url": next_url},
            )

        messages.success(request, "Te enviamos un codigo de 6 digitos a tu correo.")
        return _redirect_with_next("registro_local", next_url)

    if action == "resend_code":
        pending = _pending_registration(request)
        if not pending:
            messages.error(request, "Primero completa el formulario para solicitar un codigo.")
            return _redirect_with_next("registro_local", next_url)

        if _registration_expired(pending):
            _clear_pending_registration(request)
            messages.error(request, "El codigo expiro. Vuelve a completar el registro.")
            return _redirect_with_next("registro_local", next_url)

        resend_in = _seconds_until_resend(pending)
        if resend_in > 0:
            messages.warning(request, f"Espera {resend_in} segundos para reenviar el codigo.")
            return _redirect_with_next("registro_local", next_url)

        code = f"{random.randint(0, 999999):06d}"
        pending["code"] = code
        pending["issued_at"] = timezone.now().isoformat()
        pending["last_sent_at"] = timezone.now().isoformat()
        request.session[REGISTRATION_SESSION_KEY] = pending
        request.session.modified = True

        try:
            _send_registration_code_email(email=pending.get("email", ""), code=code)
        except Exception:
            messages.error(request, "No se pudo reenviar el codigo. Intenta nuevamente.")
            return _redirect_with_next("registro_local", next_url)

        messages.success(request, "Te reenviamos un nuevo codigo de verificacion.")
        return _redirect_with_next("registro_local", next_url)

    if action == "reset_registration":
        _clear_pending_registration(request)
        messages.info(request, "Registro reiniciado. Puedes ingresar tus datos nuevamente.")
        return _redirect_with_next("registro_local", next_url)

    if action == "verify_code":
        entered_code = "".join(
            ch for ch in (request.POST.get("codigo_verificacion") or "").strip() if ch.isdigit()
        )
        pending = _pending_registration(request)

        if not pending:
            messages.error(request, "Primero solicita un codigo de verificacion.")
            return _redirect_with_next("registro_local", next_url)

        if _registration_expired(pending):
            _clear_pending_registration(request)
            messages.error(request, "El codigo expiro. Solicita uno nuevo.")
            return _redirect_with_next("registro_local", next_url)

        if entered_code != pending.get("code"):
            messages.error(request, "El codigo ingresado no es correcto.")
            return _redirect_with_next("registro_local", next_url)

        email = pending.get("email", "").lower()
        username = pending.get("username", "")
        password = pending.get("password", "")

        user_model = get_user_model()
        if user_model.objects.filter(email__iexact=email).exists():
            _clear_pending_registration(request)
            messages.error(request, "Ese correo ya esta registrado. Inicia sesion.")
            return _redirect_with_next("login", next_url)

        if user_model.objects.filter(username__iexact=username).exists():
            _clear_pending_registration(request)
            messages.error(request, "Ese nombre de usuario ya esta en uso.")
            return _redirect_with_next("registro_local", next_url)

        new_user = user_model.objects.create_user(
            username=username,
            email=email,
            password=password,
        )
        _clear_pending_registration(request)
        login(request, new_user, backend="django.contrib.auth.backends.ModelBackend")
        messages.success(request, "Registro verificado correctamente.")
        return redirect(next_url or "chat_home")

    messages.error(request, "Accion de registro no valida.")
    return _redirect_with_next("registro_local", next_url)


def cerrar_sesion_view(request):
    logout(request)
    return redirect("login")