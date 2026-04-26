from django.urls import path

from apps.login.views import cerrar_sesion_view, login_invitado_view, login_view, registro_view

urlpatterns = [
    path("login_registro/", login_view, name="login"),
    path("invitado/", login_invitado_view, name="login_invitado"),
    path("registro/", registro_view, name="registro"),
    path("registro/", registro_view, name="registro_local"),
    path("cerrar-sesion/", cerrar_sesion_view, name="cerrar_sesion"),
    path("login/", login_view),
    path("register/", registro_view),
    path("logout/", cerrar_sesion_view, name="logout"),
]
