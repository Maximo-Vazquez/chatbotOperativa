import json

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from openai import OpenAI

from .models import UserAPIKey
from apps.herramientas.tools import TOOL_DEFINITIONS, ejecutar_herramienta
from apps.herramientas.models import ToolCall

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


def _get_user_api_key(user, provider_key):
    try:
        return UserAPIKey.objects.get(user=user, provider=provider_key).api_key.strip()
    except UserAPIKey.DoesNotExist:
        return ""


def _get_chat_config(request):
    provider_key = request.session.get("chat_provider", "deepseek")
    if provider_key not in PROVIDERS:
        provider_key = "deepseek"

    provider = PROVIDERS[provider_key]
    default_model = provider["models"][0]
    model = request.session.get("chat_model", default_model)
    if model not in provider["models"]:
        model = default_model

    api_key = _get_user_api_key(request.user, provider_key)
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


def _model_supports_tools(provider_key, model):
    """DeepSeek Reasoner no soporta tool calling."""
    if provider_key == "deepseek" and model == "deepseek-reasoner":
        return False
    return True


@csrf_exempt
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

    # Herramienta sugerida desde el frontend (opcional)
    suggested_tool = (payload.get("suggested_tool") or "").strip()

    history = _clean_history(request.session.get("chat_history", []))
    messages = [{"role": "system", "content": settings.CHATBOT_SYSTEM_PROMPT}]
    for item in history:
        messages.append({"role": "user", "content": item["user"]})
        messages.append({"role": "assistant", "content": item["assistant"]})

    # Si hay herramienta sugerida, agregar hint en el mensaje del usuario
    final_user_message = user_message
    if suggested_tool:
        final_user_message = (
            f"{user_message}\n\n"
            f"[Sugerencia del usuario: por favor usá la herramienta '{suggested_tool}' para responder.]"
        )

    messages.append({"role": "user", "content": final_user_message})

    try:
        client_kwargs = {"api_key": api_key}
        if provider["base_url"]:
            client_kwargs["base_url"] = provider["base_url"]

        client = OpenAI(**client_kwargs)
        supports_tools = _model_supports_tools(provider_key, model)

        call_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": 0.4,
        }
        if supports_tools:
            call_kwargs["tools"] = TOOL_DEFINITIONS
            call_kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**call_kwargs)
        choice = response.choices[0]

        tool_result_data = None  # datos para el frontend si se usó herramienta

        # ── Ciclo tool use ──
        if supports_tools and choice.finish_reason == "tool_calls":
            tool_calls = choice.message.tool_calls

            # Agregar el mensaje del asistente con las tool_calls
            messages.append(choice.message)

            tool_results_for_history = []
            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                result = ejecutar_herramienta(fn_name, fn_args)

                # Guardar en DB
                ToolCall.objects.create(
                    user=request.user,
                    tool_name=fn_name,
                    input_data=fn_args,
                    output_data=result,
                )

                # Acumular para la respuesta al frontend
                tool_result_data = {
                    "tool_name": fn_name,
                    "input": fn_args,
                    "output": result,
                }
                tool_results_for_history.append(tool_result_data)

                # Agregar resultado de la herramienta a los mensajes
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

            # Segunda llamada: el modelo interpreta los resultados
            response2 = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.4,
            )
            assistant_message = (response2.choices[0].message.content or "").strip()
        else:
            assistant_message = (choice.message.content or "").strip()

    except Exception as exc:
        return JsonResponse({"error": f"Error al consultar {provider['label']}: {exc}"}, status=502)

    if not assistant_message:
        assistant_message = "No pude generar respuesta. Intentá nuevamente."

    history.append({"user": user_message, "assistant": assistant_message})
    request.session["chat_history"] = history[-MAX_HISTORY_ITEMS:]

    response_payload = {"response": assistant_message}
    if tool_result_data:
        response_payload["tool_result"] = tool_result_data

    return JsonResponse(response_payload)


@csrf_exempt
@login_required
@require_POST
def reset_chat(request):
    request.session["chat_history"] = []
    return JsonResponse({"ok": True})


@csrf_exempt
@login_required
@require_POST
def save_config(request):
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
        UserAPIKey.objects.update_or_create(
            user=request.user,
            provider=provider_key,
            defaults={"api_key": api_key},
        )

    request.session["chat_history"] = []

    has_key = bool(
        _get_user_api_key(request.user, provider_key) or
        getattr(settings, provider["key_setting"], "")
    )

    return JsonResponse({
        "ok": True,
        "provider": provider_key,
        "model": request.session["chat_model"],
        "has_api_key": has_key,
    })
