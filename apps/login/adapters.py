import re

from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from django.contrib.auth import get_user_model

User = get_user_model()


class MySocialAccountAdapter(DefaultSocialAccountAdapter):
    def pre_social_login(self, request, sociallogin):
        email = None
        if sociallogin.account.extra_data and "email" in sociallogin.account.extra_data:
            email = sociallogin.account.extra_data["email"]

        if email:
            try:
                user = User.objects.get(email=email)
                sociallogin.user = user
            except User.DoesNotExist:
                pass

    def _build_unique_username(self, user, extra_data):
        given_name = (extra_data.get("given_name") or "").strip()
        family_name = (extra_data.get("family_name") or "").strip()
        full_name = (extra_data.get("name") or "").strip()

        if not full_name:
            full_name = f"{given_name} {family_name}".strip()

        if full_name:
            base_username = re.sub(r"[^\w\s]", "", full_name)
            base_username = re.sub(r"\s+", " ", base_username).strip()
        else:
            email_prefix = (user.email or "").split("@")[0]
            base_username = re.sub(r"\W+", " ", email_prefix.lower()).strip()

        if not base_username:
            base_username = "usuario"

        max_length = User._meta.get_field("username").max_length
        base_username = base_username[:max_length]
        username = base_username
        counter = 1

        while User.objects.filter(username=username).exists():
            suffix = str(counter)
            trimmed_base = base_username[: max_length - len(suffix)]
            username = f"{trimmed_base}{suffix}"
            counter += 1

        return username

    def save_user(self, request, sociallogin, form=None):
        user = sociallogin.user
        extra_data = sociallogin.account.extra_data or {}

        given_name = (extra_data.get("given_name") or "").strip()
        family_name = (extra_data.get("family_name") or "").strip()
        full_name = (extra_data.get("name") or "").strip()

        if not user.first_name and given_name:
            user.first_name = given_name
        if not user.last_name and family_name:
            user.last_name = family_name

        if not user.first_name and not user.last_name and full_name:
            parts = full_name.split(" ", 1)
            user.first_name = parts[0]
            if len(parts) > 1:
                user.last_name = parts[1]

        if not user.username:
            user.username = self._build_unique_username(user, extra_data)

        return super().save_user(request, sociallogin, form)