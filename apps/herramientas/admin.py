from django.contrib import admin
from .models import ToolCall


@admin.register(ToolCall)
class ToolCallAdmin(admin.ModelAdmin):
    list_display = ("user", "tool_name", "created_at")
    list_filter = ("tool_name",)
    readonly_fields = ("user", "tool_name", "input_data", "output_data", "created_at")
