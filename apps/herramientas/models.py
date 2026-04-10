from django.db import models


class ToolCall(models.Model):
    """Registro de cada vez que se llamó una herramienta durante el chat."""
    user = models.ForeignKey(
        "auth.User",
        on_delete=models.CASCADE,
        related_name="tool_calls",
    )
    tool_name = models.CharField(max_length=64)
    input_data = models.JSONField()
    output_data = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user} / {self.tool_name} @ {self.created_at:%Y-%m-%d %H:%M}"
