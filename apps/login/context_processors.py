from allauth.socialaccount.models import SocialAccount
from django.conf import settings


def google_profile_context(request):
    picture = ""
    if getattr(request, "user", None) and request.user.is_authenticated:
        account = (
            SocialAccount.objects.filter(user=request.user, provider="google")
            .only("extra_data")
            .first()
        )
        if account and isinstance(account.extra_data, dict):
            raw_picture = account.extra_data.get("picture")
            if isinstance(raw_picture, str):
                picture = raw_picture.strip()

    return {"google_profile_picture_url": picture}


def app_version_context(request):
    return {"app_version": getattr(settings, "APP_VERSION", "dev-local")}