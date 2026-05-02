import json
import re

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST
from openai import OpenAI

from .models import UserAPIKey
from .constants import (
    MODEL_DEEPSEEK_FLASH,
    MODEL_DEEPSEEK_PRO,
    get_context_window,
    get_model_params,
)
from apps.herramientas.tools import TOOL_DEFINITIONS, ejecutar_herramienta
from apps.herramientas.models import ToolCall

SESION_INVITADO = "es_invitado"

PROVIDERS = {
    "deepseek": {
        "label": "DeepSeek",
        "models": [MODEL_DEEPSEEK_FLASH, MODEL_DEEPSEEK_PRO],
        "base_url": settings.DEEPSEEK_BASE_URL,
        "key_setting": "DEEPSEEK_API_KEY",
    },
}


def _es_solicitud_invitada(request):
    return bool(request.session.get(SESION_INVITADO)) and not request.user.is_authenticated


def _tiene_acceso_chat(request):
    return request.user.is_authenticated or _es_solicitud_invitada(request)


def requiere_acceso_chat(view_func):
    def wrapper(request, *args, **kwargs):
        if not _tiene_acceso_chat(request):
            return redirect("login")
        return view_func(request, *args, **kwargs)

    return wrapper


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
    return cleaned


def _get_user_api_key(user, provider_key):
    if not user.is_authenticated:
        return ""
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

    if _es_solicitud_invitada(request):
        api_key = settings.CLAVE_API_DEEPSEEK_INVITADO
    else:
        api_key = _get_user_api_key(request.user, provider_key)
    if not api_key:
        api_key = getattr(settings, provider["key_setting"], "").strip()

    return provider_key, provider, model, api_key


@requiere_acceso_chat
@ensure_csrf_cookie
def chat_home(request):
    history = _clean_history(request.session.get("chat_history", []))
    provider_key, provider, model, api_key = _get_chat_config(request)

    return render(request, "chatbot/home.html", {
        "chat_history": history,
        "chat_usage": request.session.get("chat_last_usage"),
        "chatbot_model": model,
        "current_provider": provider_key,
        "current_model": model,
        "providers": {k: {"label": v["label"], "models": v["models"]} for k, v in PROVIDERS.items()},
        "has_api_key": bool(api_key),
        "es_invitado": _es_solicitud_invitada(request),
        "nombre_usuario": "Invitado" if _es_solicitud_invitada(request) else request.user.username,
    })


def _model_supports_tools(provider_key, model):
    """Ambos modelos DeepSeek V4 soportan tool calling.
    El alias legacy 'deepseek-reasoner' (mapea a v4-pro thinking) tampoco lo bloquea
    porque la API V4 lo expone en ambos modos. Mantenemos el helper por si en el
    futuro alguno lo deshabilita."""
    return True


# Bloques DSML que algunos modelos DeepSeek emiten como texto en lugar de
# usar el campo nativo `tool_calls`. Los detectamos para parsearlos manualmente.
_DSML_BLOCK_RE = re.compile(
    r"<\s*[│|｜][^>]*?DSML[^>]*?[│|｜]\s*tool_calls\s*>(.*?)<\s*/\s*[│|｜][^>]*?DSML[^>]*?[│|｜]\s*tool_calls\s*>",
    re.DOTALL,
)
_DSML_INVOKE_RE = re.compile(
    r"<\s*[│|｜][^>]*?DSML[^>]*?[│|｜]\s*invoke\s+name\s*=\s*\"([^\"]+)\"\s*>(.*?)<\s*/\s*[│|｜][^>]*?DSML[^>]*?[│|｜]\s*invoke\s*>",
    re.DOTALL,
)
_DSML_PARAM_RE = re.compile(
    r"<\s*[│|｜][^>]*?DSML[^>]*?[│|｜]\s*parameter\s+name\s*=\s*\"([^\"]+)\"(?:\s+string\s*=\s*\"(true|false)\")?\s*>(.*?)<\s*/\s*[│|｜][^>]*?DSML[^>]*?[│|｜]\s*parameter\s*>",
    re.DOTALL,
)


def _parse_dsml_tool_calls(text: str):
    """Si el modelo emitió tool_calls como texto DSML, devuelve una lista
    de (nombre_funcion, dict_argumentos). Si no encuentra, devuelve []."""
    if not text or "DSML" not in text:
        return []
    calls = []
    for m_invoke in _DSML_INVOKE_RE.finditer(text):
        fn_name = m_invoke.group(1)
        body = m_invoke.group(2)
        args = {}
        for m_param in _DSML_PARAM_RE.finditer(body):
            pname = m_param.group(1)
            is_string_flag = m_param.group(2)
            raw = m_param.group(3).strip()
            if is_string_flag == "false":
                try:
                    args[pname] = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    args[pname] = raw
            else:
                args[pname] = raw
        calls.append((fn_name, args))
    return calls


def _strip_dsml_blocks(text: str) -> str:
    """Quita cualquier bloque DSML residual del texto visible al usuario."""
    if not text:
        return text
    cleaned = _DSML_BLOCK_RE.sub("", text)
    cleaned = re.sub(r"<\s*/?\s*[│|｜][^>]*?DSML[^>]*?[│|｜][^>]*?>", "", cleaned)
    return cleaned.strip()


def _completion_kwargs(model, messages, params, include_tools=False):
    call_kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": params["max_tokens"],
        "extra_body": {
            "thinking": {"type": "enabled" if params["thinking"] else "disabled"},
        },
    }
    if params["thinking"]:
        call_kwargs["extra_body"]["reasoning_effort"] = "high"
    elif params["temperature"] is not None:
        call_kwargs["temperature"] = params["temperature"]

    if include_tools:
        call_kwargs["tools"] = TOOL_DEFINITIONS
        call_kwargs["tool_choice"] = "auto"

    return call_kwargs


@requiere_acceso_chat
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
    system_prompt = settings.CHATBOT_SYSTEM_PROMPT + (
        "\n\nIMPORTANTE: cuando necesites usar una herramienta, usá EXCLUSIVAMENTE "
        "el mecanismo nativo de function calling (tool_calls). Nunca escribas bloques "
        "<DSML>, <invoke>, <parameter> ni etiquetas similares en el contenido del mensaje. "
        "Si no vas a usar una herramienta, respondé directamente en español."
    )
    messages = [{"role": "system", "content": system_prompt}]
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
        params = get_model_params(model)

        call_kwargs = _completion_kwargs(model, messages, params, include_tools=supports_tools)

        response = client.chat.completions.create(**call_kwargs)
        choice = response.choices[0]
        last_response = response  # se actualiza si hay segunda llamada

        tool_result_data = None  # ultima herramienta (compatibilidad frontend)
        tool_results_all = []  # todas las herramientas usadas en la respuesta

        # ── Ciclo tool use: hasta MAX_TOOL_ITERS pasadas encadenadas ──
        MAX_TOOL_ITERS = 9
        iteraciones = 0
        while (
            supports_tools
            and choice.finish_reason == "tool_calls"
            and iteraciones < MAX_TOOL_ITERS
        ):
            iteraciones += 1
            tool_calls = choice.message.tool_calls

            # Agregar el mensaje del asistente con las tool_calls
            messages.append(choice.message)

            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                result = ejecutar_herramienta(fn_name, fn_args)

                # Guardar en DB
                if request.user.is_authenticated:
                    ToolCall.objects.create(
                        user=request.user,
                        tool_name=fn_name,
                        input_data=fn_args,
                        output_data=result,
                    )

                tool_result_data = {
                    "tool_name": fn_name,
                    "input": fn_args,
                    "output": result,
                }
                tool_results_all.append(tool_result_data)

                # Agregar resultado de la herramienta a los mensajes
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

            # Siguiente llamada: el modelo puede pedir más herramientas o cerrar.
            # En la última iteración permitida, deshabilitamos tools para forzar respuesta final.
            permitir_mas_tools = iteraciones < MAX_TOOL_ITERS
            response_iter = client.chat.completions.create(
                **_completion_kwargs(model, messages, params, include_tools=permitir_mas_tools)
            )
            last_response = response_iter
            choice = response_iter.choices[0]

        if supports_tools and tool_results_all:
            assistant_message = _strip_dsml_blocks((choice.message.content or "").strip())
        elif supports_tools and choice.finish_reason == "tool_calls":
            # Salimos por tope sin haber procesado: forzamos mensaje informativo
            assistant_message = (choice.message.content or "").strip()
        else:
            assistant_message = (choice.message.content or "").strip()

            # Fallback: el modelo emitió tool_calls como texto DSML
            dsml_calls = _parse_dsml_tool_calls(assistant_message) if supports_tools else []
            if dsml_calls:
                for fn_name, fn_args in dsml_calls:
                    result = ejecutar_herramienta(fn_name, fn_args)
                    if request.user.is_authenticated:
                        ToolCall.objects.create(
                            user=request.user,
                            tool_name=fn_name,
                            input_data=fn_args,
                            output_data=result,
                        )
                    tool_result_data = {
                        "tool_name": fn_name,
                        "input": fn_args,
                        "output": result,
                    }
                    messages.append({
                        "role": "assistant",
                        "content": (
                            f"He invocado la herramienta {fn_name}. "
                            f"Resultado: {json.dumps(result, ensure_ascii=False)}"
                        ),
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            "Interpretá el resultado anterior y respondeme en español, "
                            "sin volver a llamar herramientas y sin incluir ningún bloque DSML."
                        ),
                    })

                response2 = client.chat.completions.create(**_completion_kwargs(model, messages, params))
                last_response = response2
                assistant_message = (response2.choices[0].message.content or "").strip()

            assistant_message = _strip_dsml_blocks(assistant_message)

    except Exception as exc:
        if _is_context_length_error(exc):
            context_window = get_context_window(model)
            return JsonResponse(
                {
                    "error": (
                        "El chat es demasiado largo y superó el contexto del modelo. "
                        "Iniciá un nuevo chat para continuar."
                    ),
                    "context_overflow": True,
                    "usage": {
                        "context_window": context_window,
                        "prompt_tokens": context_window,
                        "context_used_percent": 100,
                        "context_remaining_percent": 0,
                        "context_remaining_tokens": 0,
                    },
                },
                status=413,
            )
        return JsonResponse({"error": f"Error al consultar {provider['label']}: {exc}"}, status=502)

    if not assistant_message:
        assistant_message = "No pude generar respuesta. Intentá nuevamente."

    history.append({"user": user_message, "assistant": assistant_message})
    request.session["chat_history"] = history

    response_payload = {"response": assistant_message}
    if tool_result_data:
        response_payload["tool_result"] = tool_result_data
    if tool_results_all:
        response_payload["tool_results"] = tool_results_all

    usage_payload = _build_usage_payload(last_response, model)
    response_payload["usage"] = usage_payload
    request.session["chat_last_usage"] = usage_payload

    return JsonResponse(response_payload)


def _is_context_length_error(exc) -> bool:
    """Detecta el 400 que devuelve DeepSeek (formato OpenAI) cuando el prompt
    supera la ventana de contexto del modelo."""
    msg = str(exc).lower()
    if "context_length_exceeded" in msg or "context length" in msg:
        return True
    if "maximum context" in msg and "tokens" in msg:
        return True
    code = getattr(exc, "code", None) or ""
    if isinstance(code, str) and code.lower() == "context_length_exceeded":
        return True
    return False


def _build_usage_payload(api_response, model):
    """Construye el objeto `usage` que consume el frontend para el indicador
    de ventana de contexto. Usa `prompt_tokens` como referencia del contexto
    enviado; el cache hit/miss es solo informativo (costo/latencia)."""
    context_window = get_context_window(model)
    usage = getattr(api_response, "usage", None)
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "context_window": context_window,
            "context_used_percent": 0,
            "context_remaining_percent": 100,
            "context_remaining_tokens": context_window,
        }

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or 0
    cache_hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
    cache_miss = getattr(usage, "prompt_cache_miss_tokens", 0) or 0

    used = min(prompt_tokens, context_window)
    used_pct = (used / context_window * 100) if context_window else 0
    remaining = max(context_window - used, 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_cache_hit_tokens": cache_hit,
        "prompt_cache_miss_tokens": cache_miss,
        "context_window": context_window,
        "context_used_percent": round(used_pct, 2),
        "context_remaining_percent": round(100 - used_pct, 2),
        "context_remaining_tokens": remaining,
    }


@requiere_acceso_chat
@require_POST
def reset_chat(request):
    request.session["chat_history"] = []
    request.session["chat_last_usage"] = None
    return JsonResponse({"ok": True})


@requiere_acceso_chat
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

    prev_provider = request.session.get("chat_provider")
    history_cleared = False
    request.session["chat_provider"] = provider_key
    request.session["chat_model"] = model or provider["models"][0]

    if api_key and request.user.is_authenticated:
        UserAPIKey.objects.update_or_create(
            user=request.user,
            provider=provider_key,
            defaults={"api_key": api_key},
        )

    # Solo limpiar el historial cuando cambia el proveedor o se pisa la API key.
    # Cambiar de modelo dentro del mismo proveedor (ej. flash ↔ pro vía toggle
    # de razonamiento) preserva la conversación.
    if api_key or (prev_provider and prev_provider != provider_key):
        request.session["chat_history"] = []
        request.session["chat_last_usage"] = None
        history_cleared = True

    has_key = bool(settings.CLAVE_API_DEEPSEEK_INVITADO) if _es_solicitud_invitada(request) else bool(
        _get_user_api_key(request.user, provider_key) or
        getattr(settings, provider["key_setting"], "")
    )

    return JsonResponse({
        "ok": True,
        "provider": provider_key,
        "model": request.session["chat_model"],
        "has_api_key": has_key,
        "history_cleared": history_cleared,
    })
