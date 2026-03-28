from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from apps.login.views import REGISTRATION_RESEND_COOLDOWN_SECONDS, REGISTRATION_SESSION_KEY


@override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")
class LoginAndRegistroTests(TestCase):
    def test_login_view_available(self):
        response = self.client.get(reverse("login"))
        self.assertEqual(response.status_code, 200)

    def test_envia_codigo_y_guarda_registro_pendiente(self):
        response = self.client.post(
            reverse("registro_local"),
            data={
                "action": "send_code",
                "gmail": "nuevo@example.com",
                "usuario": "usuario_local",
                "contrasena": "ClaveSuperSegura123!",
                "conf_contrasena": "ClaveSuperSegura123!",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.url.startswith(reverse("registro_local")))

        pending = self.client.session.get(REGISTRATION_SESSION_KEY)
        self.assertIsNotNone(pending)
        self.assertEqual(pending["email"], "nuevo@example.com")
        self.assertEqual(pending["username"], "usuario_local")
        self.assertEqual(len(pending["code"]), 6)

    def test_verifica_codigo_y_crea_usuario(self):
        session = self.client.session
        session[REGISTRATION_SESSION_KEY] = {
            "email": "nuevo2@example.com",
            "username": "usuario_local_2",
            "password": "ClaveSuperSegura123!",
            "code": "123456",
            "issued_at": timezone.now().isoformat(),
            "last_sent_at": timezone.now().isoformat(),
        }
        session.save()

        response = self.client.post(
            reverse("registro_local"),
            data={
                "action": "verify_code",
                "codigo_verificacion": "123456",
            },
        )

        self.assertEqual(response.status_code, 302)

        user_model = get_user_model()
        self.assertTrue(user_model.objects.filter(email="nuevo2@example.com").exists())
        self.assertNotIn(REGISTRATION_SESSION_KEY, self.client.session)

    def test_resend_code_permitido_tras_cooldown(self):
        session = self.client.session
        session[REGISTRATION_SESSION_KEY] = {
            "email": "reenvio@example.com",
            "username": "usuario_reenvio",
            "password": "ClaveSuperSegura123!",
            "code": "222222",
            "issued_at": timezone.now().isoformat(),
            "last_sent_at": (
                timezone.now() - timedelta(seconds=REGISTRATION_RESEND_COOLDOWN_SECONDS + 1)
            ).isoformat(),
        }
        session.save()

        response = self.client.post(
            reverse("registro_local"),
            data={"action": "resend_code"},
        )
        self.assertEqual(response.status_code, 302)
        pending = self.client.session.get(REGISTRATION_SESSION_KEY)
        self.assertIsNotNone(pending)
        self.assertNotEqual(pending["code"], "222222")