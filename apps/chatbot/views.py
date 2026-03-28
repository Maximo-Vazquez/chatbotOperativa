import json

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from openai import OpenAI

MAX_HISTORY_ITEMS = 12

PROVIDERS = {
    "openai": {
        "label": "OpenAI",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "base_url": None,
        "key_setting": "OPENAI_API_KEY",
    },
    "deepseek": {
        "label": "DeepSeek",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "base_url": settings.DEEPSEEK_BASE_URL,
        "key_setting": "DEEPSEEK_API_KEY",
    },
}


def _clean_history(raw_history):
    if not isinstance(raw_history, list):
        return []
    cleaned = []
    for item in raw_history:
        if not isinstance(item, dict):
            continue
        user_message = (item.get("user") or "").strip()
        assistant_message = (item.get("assistant") or "").strip()
        if user_message and assistant_message:
            cleaned.append({"user": user_message, "assistant": assistant_message})
    return cleaned[-MAX_HISTORY_ITEMS:]


def _get_chat_config(request):
    """
    Devuelve (provider_key, model, api_key) según la sesión del usuario.
    La sesión puede sobreescribir el proveedor, modelo y clave API.
    """
    provider_key = request.session.get("chat_provider", "openai")
    if provider_key not in PROVIDERS:
        provider_key = "openai"

    provider = PROVIDERS[provider_key]
    default_model = provider["models"][0]
    model = request.session.get("chat_model", default_model)
    if model not in provider["models"]:
        model = default_model

    # Clave: primero la guardada en sesión, luego la de settings
    api_key = request.session.get(f"api_key_{provider_key}", "").strip()
    if not api_key:
        api_key = getattr(settings, provider["key_setting"], "").strip()

    return provider_key, provider, model, api_key


@login_required
def chat_home(request):
    history = _clean_history(request.session.get("chat_history", []))
    provider_key, provider, model, api_key = _get_chat_config(request)

    return render(request, "chatbot/home.html", {
        "chat_history": history,
        "chatbot_model": model,
        "current_provider": provider_key,
        "current_model": model,
        "providers": {k: {"label": v["label"], "models": v["models"]} for k, v in PROVIDERS.items()},
        "has_api_key": bool(api_key),
    })


@login_required
@require_POST
def chat_api(request):
    provider_key, provider, model, api_key = _get_chat_config(request)

    if not api_key:
        return JsonResponse(
            {"error": f"Falta la API key para {provider['label']}. Configurala en el panel."},
            status=500,
        )

    payload = {}
    if (request.content_type or "").startswith("application/json"):
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return JsonResponse({"error": "JSON inválido."}, status=400)
    else:
        payload = request.POST

    user_message = (payload.get("message") or "").strip()
    if not user_message:
        return JsonResponse({"error": "El mensaje no puede estar vacío."}, status=400)

    history = _clean_history(request.session.get("chat_history", []))
    messages = [{"role": "system", "content": settings.CHATBOT_SYSTEM_PROMPT}]
    for item in history:
        messages.append({"role": "user", "content": item["user"]})
        messages.append({"role": "assistant", "content": item["assistant"]})
    messages.append({"role": "user", "content": user_message})

    try:
        client_kwargs = {"api_key": api_key}
        if provider["base_url"]:
            client_kwargs["base_url"] = provider["base_url"]

        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.4,
        )
        assistant_message = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        return JsonResponse({"error": f"Error al consultar {provider['label']}: {exc}"}, status=502)

    if not assistant_message:
        assistant_message = "No pude generar respuesta. Intentá nuevamente."

    history.append({"user": user_message, "assistant": assistant_message})
    request.session["chat_history"] = history[-MAX_HISTORY_ITEMS:]

    return JsonResponse({"response": assistant_message})


@login_required
@require_POST
def reset_chat(request):
    request.session["chat_history"] = []
    return JsonResponse({"ok": True})


@login_required
@require_POST
def save_config(request):
    """Guarda proveedor, modelo y API key en la sesión del usuario."""
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "JSON inválido."}, status=400)

    provider_key = (payload.get("provider") or "").strip()
    model = (payload.get("model") or "").strip()
    api_key = (payload.get("api_key") or "").strip()

    if provider_key not in PROVIDERS:
        return JsonResponse({"error": "Proveedor inválido."}, status=400)

    provider = PROVIDERS[provider_key]
    if model and model not in provider["models"]:
        return JsonResponse({"error": "Modelo inválido para ese proveedor."}, status=400)

    request.session["chat_provider"] = provider_key
    request.session["chat_model"] = model or provider["models"][0]

    if api_key:
        request.session[f"api_key_{provider_key}"] = api_key

    # Limpiar historial al cambiar proveedor/modelo
    request.session["chat_history"] = []

    return JsonResponse({
        "ok": True,
        "provider": provider_key,
        "model": request.session["chat_model"],
        "has_api_key": bool(
            request.session.get(f"api_key_{provider_key}") or
            getattr(settings, provider["key_setting"], "")
        ),
    })
