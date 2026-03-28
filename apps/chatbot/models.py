from django.conf import settings
from django.db import models


class UserAPIKey(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="api_keys",
    )
    provider = models.CharField(max_length=32)
    api_key = models.TextField()

    class Meta:
        unique_together = ("user", "provider")

    def __str__(self):
        return f"{self.user} / {self.provider}"
