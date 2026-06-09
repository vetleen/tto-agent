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
    remember_me = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(
            attrs={
                "class": (
                    "h-4 w-4 rounded-sm border-default-medium text-brand "
                    "focus:ring-brand focus:ring-2"
                )
            }
        ),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Login form gets an extra left inset (pl-10) for the inline field icons.
        icon_classes = _input_classes() + " pl-10"
        self.fields["username"].widget = forms.EmailInput(attrs={"class": icon_classes})
        self.fields["password"].widget = forms.PasswordInput(attrs={"class": icon_classes})


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
    # White-inset field (.wf-input owns background, hairline, inset shadow and focus).
    return "wf-input mt-1 w-full rounded-base px-3 py-2 text-heading placeholder:text-body"


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
