from django.urls import path

from apps.chatbot import views

urlpatterns = [
    path("", views.chat_home, name="chat_home"),
    path("chat/api/message/", views.chat_api, name="chat_api"),
    path("chat/api/reset/", views.reset_chat, name="chat_reset"),
    path("chat/api/config/", views.save_config, name="chat_config"),
]