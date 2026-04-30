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

    def test_clean_history_keeps_full_valid_history(self):
        raw_history = [
            {"user": f"pregunta {i}", "assistant": f"respuesta {i}"}
            for i in range(20)
        ]

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
