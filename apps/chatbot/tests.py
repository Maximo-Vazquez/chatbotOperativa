from django.test import TestCase
from django.urls import reverse
from django.contrib.auth import get_user_model
from unittest.mock import patch


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

    @patch("apps.chatbot.views.OpenAI")
    @patch("apps.chatbot.views.settings.OPENAI_API_KEY", "dummy-key")
    def test_chat_api_returns_response(self, mock_openai):
        self.client.force_login(self.user)

        mock_client = mock_openai.return_value
        mock_result = type("obj", (), {})()
        choice = type("obj", (), {})()
        choice.message = type("obj", (), {"content": "respuesta de prueba"})()
        mock_result.choices = [choice]
        mock_client.chat.completions.create.return_value = mock_result

        response = self.client.post(
            reverse("chat_api"),
            data="{\"message\": \"hola\"}",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertJSONEqual(response.content, {"response": "respuesta de prueba"})