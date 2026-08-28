from django.test import TestCase
from django.urls import reverse
from django.contrib.auth import get_user_model
from unittest.mock import patch

from .constants import MODEL_DEEPSEEK_FLASH, MODEL_DEEPSEEK_PRO
from .views import _clean_history, _completion_kwargs


class ChatbotViewsTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            username="tester",
            email="tester@example.com",
            password="StrongPass123!",
        )

    def test_home_requires_login(self):
        response = self.client.get(reverse("chat_home"))
        self.assertEqual(response.status_code, 302)

    def test_home_includes_throttled_markdown_rendering_for_streamed_deltas(self):
        self.client.force_login(self.user)

        response = self.client.get(reverse("chat_home"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "function scheduleMarkdownRender()")
        self.assertContains(response, "textDiv.innerHTML = renderMarkdownPreview(assistantText)")
        self.assertContains(response, "function showStreamingStatus(message)")
        self.assertContains(response, "showStreamingStatus(data.message || 'Procesando…')")
        self.assertContains(response, "textDiv.after(groupEl)")
        self.assertContains(response, "if (event === 'delta') {\n        statusEl.remove();\n        startText();")
        self.assertNotContains(response, "function showStreamingStatus(message) {\n      if (started) return;")
        self.assertContains(response, "statusEl.remove();")

    def test_tool_card_copy_all_button_binds_json_without_inline_html_interpolation(self):
        self.client.force_login(self.user)

        response = self.client.get(reverse("chat_home"))

        self.assertEqual(response.status_code, 200)
        # El botón no debe llevar el JSON serializado interpolado en un atributo HTML/onclick.
        self.assertNotContains(response, "onclick=\"tcCopyAll(this,")
        # El valor original (jsonAll) se asocia al handler desde JS, vía closure, sin pasar por HTML.
        self.assertContains(
            response,
            "copyAllBtn.addEventListener('click', () => tcCopyAll(copyAllBtn, jsonAll));",
        )
        self.assertContains(response, "function tcCopyAll(btn, jsonText) {")
        # El rechazo del portapapeles debe manejarse explícitamente, no fallar en silencio.
        self.assertContains(response, "navigator.clipboard.writeText(jsonText).then(() => {")
        self.assertContains(response, "}).catch(() => {")

    def test_clean_history_keeps_full_valid_history(self):
        raw_history = [
            {"user": f"pregunta {i}", "assistant": f"respuesta {i}"}
            for i in range(20)
        ]

        self.assertEqual(_clean_history(raw_history), raw_history)

    def test_clean_history_preserves_tool_results(self):
        raw_history = [{
            "user": "u",
            "assistant": "a",
            "tool_results": [{
                "tool_name": "acf",
                "input": {"serie": [1, 2, 3]},
                "output": {"ok": True},
            }],
        }]

        self.assertEqual(_clean_history(raw_history), raw_history)

    @patch("apps.chatbot.views.OpenAI")
    @patch("apps.chatbot.views.settings.DEEPSEEK_API_KEY", "dummy-key")
    def test_chat_api_returns_response(self, mock_openai):
        self.client.force_login(self.user)

        mock_client = mock_openai.return_value
        mock_result = type("obj", (), {})()
        choice = type("obj", (), {})()
        choice.message = type("obj", (), {"content": "respuesta de prueba"})()
        choice.finish_reason = "stop"
        mock_result.choices = [choice]
        mock_client.chat.completions.create.return_value = mock_result

        response = self.client.post(
            reverse("chat_api"),
            data="{\"message\": \"hola\"}",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["response"], "respuesta de prueba")
        self.assertIn("usage", data)

    @patch("apps.chatbot.views.OpenAI")
    @patch("apps.chatbot.views.settings.DEEPSEEK_API_KEY", "dummy-key")
    def test_chat_stream_emits_response_chunks_and_persists_completed_message(self, mock_openai):
        self.client.force_login(self.user)

        first_chunk = type("obj", (), {})()
        first_chunk.choices = [type("obj", (), {
            "delta": type("obj", (), {"content": "Hola ", "tool_calls": None})(),
            "finish_reason": None,
        })()]
        last_chunk = type("obj", (), {})()
        last_chunk.choices = [type("obj", (), {
            "delta": type("obj", (), {"content": "mundo", "tool_calls": None})(),
            "finish_reason": "stop",
        })()]
        last_chunk.usage = None

        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = iter([first_chunk, last_chunk])

        response = self.client.post(
            "/chat/api/stream/",
            data="{\"message\": \"hola\"}",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        stream = b"".join(response.streaming_content).decode("utf-8")
        self.assertIn('event: delta\ndata: {"text": "Hola "}', stream)
        self.assertIn('event: delta\ndata: {"text": "mundo"}', stream)
        self.assertIn('event: done\ndata:', stream)
        self.assertEqual(self.client.session["chat_history"], [{
            "user": "hola",
            "assistant": "Hola mundo",
        }])
        self.assertTrue(mock_client.chat.completions.create.call_args.kwargs["stream"])

    @patch("apps.chatbot.views.OpenAI")
    @patch("apps.chatbot.views.settings.DEEPSEEK_API_KEY", "dummy-key")
    def test_chat_stream_reports_context_overflow(self, mock_openai):
        self.client.force_login(self.user)
        mock_openai.return_value.chat.completions.create.side_effect = Exception(
            "context_length_exceeded"
        )

        response = self.client.post(
            "/chat/api/stream/",
            data="{\"message\": \"hola\"}",
            content_type="application/json",
        )

        stream = b"".join(response.streaming_content).decode("utf-8")
        self.assertIn("event: error", stream)
        self.assertIn('"context_overflow": true', stream)

    @patch("apps.chatbot.views.ejecutar_herramienta", return_value={"ok": True})
    @patch("apps.chatbot.views.OpenAI")
    @patch("apps.chatbot.views.settings.DEEPSEEK_API_KEY", "dummy-key")
    def test_chat_stream_executes_native_tool_calls_before_final_response(
        self, mock_openai, mock_tool
    ):
        self.client.force_login(self.user)

        tool_chunk = type("obj", (), {})()
        tool_chunk.choices = [type("obj", (), {
            "delta": type("obj", (), {
                "content": None,
                "tool_calls": [type("obj", (), {
                    "index": 0,
                    "id": "call_1",
                    "function": type("obj", (), {
                        "name": "acf",
                        "arguments": "{\"serie\": [1, 2, 3]}",
                    })(),
                })()],
            })(),
            "finish_reason": "tool_calls",
        })()]
        final_chunk = type("obj", (), {})()
        final_chunk.choices = [type("obj", (), {
            "delta": type("obj", (), {"content": "Resultado final", "tool_calls": None})(),
            "finish_reason": "stop",
        })()]
        final_chunk.usage = None

        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = [
            iter([tool_chunk]),
            iter([final_chunk]),
        ]

        response = self.client.post(
            "/chat/api/stream/",
            data="{\"message\": \"analizá esta serie\"}",
            content_type="application/json",
        )

        stream = b"".join(response.streaming_content).decode("utf-8")
        self.assertIn('event: status\ndata: {"message": "Preparando respuesta…"}', stream)
        self.assertIn('event: status\ndata: {"message": "Preparando análisis…"}', stream)
        self.assertIn('event: status\ndata: {"message": "Ejecutando acf…"}', stream)
        self.assertIn('event: status\ndata: {"message": "Analizando los resultados…"}', stream)
        self.assertIn('event: tool\ndata: {"tool_name": "acf"}', stream)
        self.assertIn('event: delta\ndata: {"text": "Resultado final"}', stream)
        self.assertLess(
            stream.index('"message": "Preparando respuesta…"'),
            stream.index('"message": "Ejecutando acf…"'),
        )
        self.assertLess(
            stream.index('"message": "Preparando análisis…"'),
            stream.index('"message": "Ejecutando acf…"'),
        )
        self.assertLess(
            stream.index('"message": "Ejecutando acf…"'),
            stream.index('"message": "Analizando los resultados…"'),
        )
        self.assertEqual(self.client.session["chat_history"][0]["assistant"], "Resultado final")
        self.assertEqual(self.client.session["chat_history"][0]["tool_results"][0]["tool_name"], "acf")
        mock_tool.assert_called_once_with("acf", {"serie": [1, 2, 3]})

    @patch("apps.chatbot.views.ejecutar_herramienta", return_value={"ok": True})
    @patch("apps.chatbot.views.OpenAI")
    @patch("apps.chatbot.views.settings.DEEPSEEK_API_KEY", "dummy-key")
    def test_chat_stream_ignores_truncated_tool_call_when_finish_reason_is_length(
        self, mock_openai, mock_tool
    ):
        self.client.force_login(self.user)

        truncated_chunk = type("obj", (), {})()
        truncated_chunk.choices = [type("obj", (), {
            "delta": type("obj", (), {
                "content": None,
                "tool_calls": [type("obj", (), {
                    "index": 0,
                    "id": "call_1",
                    "function": type("obj", (), {
                        "name": "modelo_arimax",
                        "arguments": "{\"serie\": [1, 2,",
                    })(),
                })()],
            })(),
            "finish_reason": "length",
        })()]
        truncated_chunk.usage = None

        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = iter([truncated_chunk])

        response = self.client.post(
            "/chat/api/stream/",
            data="{\"message\": \"analizá esta serie\"}",
            content_type="application/json",
        )

        stream = b"".join(response.streaming_content).decode("utf-8")
        self.assertIn("event: error", stream)
        mock_tool.assert_not_called()
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)
        self.assertNotIn("chat_history", self.client.session)

    @patch("apps.chatbot.views.ejecutar_herramienta", return_value={"ok": True})
    @patch("apps.chatbot.views.OpenAI")
    @patch("apps.chatbot.views.settings.DEEPSEEK_API_KEY", "dummy-key")
    def test_chat_stream_rejects_tool_call_with_invalid_json_arguments(
        self, mock_openai, mock_tool
    ):
        self.client.force_login(self.user)

        bad_args_chunk = type("obj", (), {})()
        bad_args_chunk.choices = [type("obj", (), {
            "delta": type("obj", (), {
                "content": None,
                "tool_calls": [type("obj", (), {
                    "index": 0,
                    "id": "call_1",
                    "function": type("obj", (), {
                        "name": "modelo_arimax",
                        "arguments": "not-json",
                    })(),
                })()],
            })(),
            "finish_reason": "tool_calls",
        })()]
        bad_args_chunk.usage = None

        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = iter([bad_args_chunk])

        response = self.client.post(
            "/chat/api/stream/",
            data="{\"message\": \"analizá esta serie\"}",
            content_type="application/json",
        )

        stream = b"".join(response.streaming_content).decode("utf-8")
        self.assertIn("event: error", stream)
        mock_tool.assert_not_called()
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)
        self.assertNotIn("chat_history", self.client.session)

    @patch("apps.chatbot.views.ejecutar_herramienta", return_value={"ok": True})
    @patch("apps.chatbot.views.OpenAI")
    @patch("apps.chatbot.views.settings.DEEPSEEK_API_KEY", "dummy-key")
    def test_chat_stream_rejects_tool_call_with_non_object_json_arguments(
        self, mock_openai, mock_tool
    ):
        self.client.force_login(self.user)

        non_object_args_chunk = type("obj", (), {})()
        non_object_args_chunk.choices = [type("obj", (), {
            "delta": type("obj", (), {
                "content": None,
                "tool_calls": [type("obj", (), {
                    "index": 0,
                    "id": "call_1",
                    "function": type("obj", (), {
                        "name": "modelo_arimax",
                        "arguments": "[]",
                    })(),
                })()],
            })(),
            "finish_reason": "tool_calls",
        })()]
        non_object_args_chunk.usage = None

        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = iter([non_object_args_chunk])

        response = self.client.post(
            "/chat/api/stream/",
            data="{\"message\": \"analizá esta serie\"}",
            content_type="application/json",
        )

        stream = b"".join(response.streaming_content).decode("utf-8")
        self.assertIn("event: error", stream)
        mock_tool.assert_not_called()
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)
        self.assertNotIn("chat_history", self.client.session)

    @patch("apps.chatbot.views.ejecutar_herramienta", return_value={"ok": True})
    @patch("apps.chatbot.views.OpenAI")
    @patch("apps.chatbot.views.settings.DEEPSEEK_API_KEY", "dummy-key")
    def test_chat_api_saves_tool_results_in_history(self, mock_openai, mock_tool):
        self.client.force_login(self.user)

        first_result = type("obj", (), {})()
        first_choice = type("obj", (), {})()
        first_choice.finish_reason = "tool_calls"
        tool_call = type("obj", (), {})()
        tool_call.id = "call_1"
        tool_call.function = type("obj", (), {
            "name": "acf",
            "arguments": "{\"serie\": [1, 2, 3]}",
        })()
        first_choice.message = type("obj", (), {
            "content": "",
            "tool_calls": [tool_call],
        })()
        first_result.choices = [first_choice]

        second_result = type("obj", (), {})()
        second_choice = type("obj", (), {})()
        second_choice.finish_reason = "stop"
        second_choice.message = type("obj", (), {"content": "respuesta con herramienta"})()
        second_result.choices = [second_choice]

        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = [first_result, second_result]

        response = self.client.post(
            reverse("chat_api"),
            data="{\"message\": \"hola\"}",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        history = self.client.session["chat_history"]
        self.assertEqual(history[0]["assistant"], "respuesta con herramienta")
        self.assertEqual(history[0]["tool_results"][0]["tool_name"], "acf")
        self.assertEqual(history[0]["tool_results"][0]["output"], {"ok": True})
        mock_tool.assert_called_once_with("acf", {"serie": [1, 2, 3]})

    def test_completion_kwargs_disables_thinking_for_standard_model(self):
        kwargs = _completion_kwargs(
            MODEL_DEEPSEEK_FLASH,
            [{"role": "user", "content": "hola"}],
            {"temperature": 0.4, "max_tokens": 8192, "thinking": False},
        )

        self.assertEqual(kwargs["temperature"], 0.4)
        self.assertEqual(kwargs["max_tokens"], 8192)
        self.assertEqual(kwargs["extra_body"]["thinking"]["type"], "disabled")
        self.assertNotIn("reasoning_effort", kwargs)

    def test_completion_kwargs_omits_temperature_for_reasoning_model(self):
        kwargs = _completion_kwargs(
            MODEL_DEEPSEEK_PRO,
            [{"role": "user", "content": "hola"}],
            {"temperature": None, "max_tokens": 32768, "thinking": True},
        )

        self.assertEqual(kwargs["max_tokens"], 32768)
        self.assertEqual(kwargs["extra_body"]["thinking"]["type"], "enabled")
        self.assertEqual(kwargs["extra_body"]["reasoning_effort"], "high")
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertNotIn("temperature", kwargs)

    def test_save_config_model_switch_preserves_history(self):
        self.client.force_login(self.user)
        session = self.client.session
        session["chat_provider"] = "deepseek"
        session["chat_model"] = MODEL_DEEPSEEK_FLASH
        session["chat_history"] = [{"user": "u", "assistant": "a"}]
        session["chat_last_usage"] = {"prompt_tokens": 10}
        session.save()

        response = self.client.post(
            reverse("chat_config"),
            data=f'{{"provider": "deepseek", "model": "{MODEL_DEEPSEEK_PRO}", "api_key": ""}}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["history_cleared"])
        self.assertEqual(self.client.session["chat_history"], [{"user": "u", "assistant": "a"}])
