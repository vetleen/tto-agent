from django import forms
from django.contrib.auth.forms import (
    AuthenticationForm,
    PasswordChangeForm,
    PasswordResetForm,
    SetPasswordForm,
    UserCreationForm,
)
from django.contrib.auth import get_user_model


class CustomAuthenticationForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base_classes = _input_classes()
        self.fields["username"].widget = forms.EmailInput(attrs={"class": base_classes})
        self.fields["password"].widget = forms.PasswordInput(attrs={"class": base_classes})


class CustomPasswordChangeForm(PasswordChangeForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base_classes = _input_classes()
        self.fields["old_password"].widget = forms.PasswordInput(attrs={"class": base_classes})
        self.fields["new_password1"].widget = forms.PasswordInput(attrs={"class": base_classes})
        self.fields["new_password2"].widget = forms.PasswordInput(attrs={"class": base_classes})


class CustomPasswordResetForm(PasswordResetForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["email"].widget = forms.EmailInput(attrs={"class": _input_classes()})


class CustomSetPasswordForm(SetPasswordForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base_classes = _input_classes()
        self.fields["new_password1"].widget = forms.PasswordInput(attrs={"class": base_classes})
        self.fields["new_password2"].widget = forms.PasswordInput(attrs={"class": base_classes})


def _input_classes() -> str:
    return (
        "mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-slate-900 "
        "focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
    )


class SignUpForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        model = get_user_model()
        fields = ("email",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base_classes = _input_classes()
        self.fields["email"].widget = forms.EmailInput(attrs={"class": base_classes})
        self.fields["password1"].widget = forms.PasswordInput(attrs={"class": base_classes})
        self.fields["password2"].widget = forms.PasswordInput(attrs={"class": base_classes})
