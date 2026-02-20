import os
import sys

from django.apps import AppConfig
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db.utils import OperationalError, ProgrammingError


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

    def ready(self) -> None:
        if not settings.DEBUG:
            return
        if "runserver" not in sys.argv:
            return
        if os.environ.get("RUN_MAIN") not in {"true", "True", "1"} and "--noreload" not in sys.argv:
            return

        username = os.environ.get("DJANGO_USER_NAME")
        password = os.environ.get("DJANGO_PASSWORD")
        if not username or not password:
            return

        try:
            user_model = get_user_model()
            lookup = {user_model.USERNAME_FIELD: username}
            user = user_model.objects.filter(**lookup).first()
            if user is None:
                user_model.objects.create_superuser(
                    **lookup,
                    password=password,
                )
                return

            changed = False
            if not user.is_staff:
                user.is_staff = True
                changed = True
            if not user.is_superuser:
                user.is_superuser = True
                changed = True
            if changed:
                user.save(update_fields=["is_staff", "is_superuser"])
        except (OperationalError, ProgrammingError):
            # Database isn't ready yet (migrations not applied).
            return
