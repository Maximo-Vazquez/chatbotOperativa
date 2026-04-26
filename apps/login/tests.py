from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.login.views import SESION_INVITADO


class LoginTests(TestCase):
    def test_login_view_disponible(self):
        response = self.client.get(reverse("login"))
        self.assertEqual(response.status_code, 200)

    def test_login_local_esta_deshabilitado(self):
        response = self.client.post(
            reverse("login"),
            data={"gmail": "usuario@example.com", "contrasena": "clave"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertFalse(response.wsgi_request.user.is_authenticated)

    def test_registro_local_redirige_al_login(self):
        response = self.client.post(reverse("registro_local"))
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("login"))

    def test_login_invitado_habilita_sesion_invitada(self):
        response = self.client.post(reverse("login_invitado"))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("chat_home"))
        self.assertTrue(self.client.session.get(SESION_INVITADO))
        self.assertEqual(self.client.session.get("chat_provider"), "deepseek")
        self.assertEqual(self.client.session.get("chat_model"), "deepseek-chat")

    def test_login_invitado_no_crea_usuario_django(self):
        self.client.post(reverse("login_invitado"))
        self.assertEqual(get_user_model().objects.count(), 0)
