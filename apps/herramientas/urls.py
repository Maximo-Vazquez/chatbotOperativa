from django.urls import path
from . import views

urlpatterns = [
    path("herramientas/api/list/", views.list_tools, name="herramientas_list"),
]
