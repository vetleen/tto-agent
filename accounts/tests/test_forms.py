"""Tests for accounts custom forms."""
from django.contrib.auth import get_user_model
from django.test import TestCase

from accounts.forms import (
    CustomAuthenticationForm,
    CustomPasswordChangeForm,
    CustomPasswordResetForm,
    CustomSetPasswordForm,
    SignUpForm,
    _input_classes,
)

User = get_user_model()


class InputClassesTests(TestCase):
    def test_returns_tailwind_classes(self) -> None:
        classes = _input_classes()
        self.assertIn("rounded-lg", classes)
        self.assertIn("focus:border-indigo-500", classes)

    def test_returns_string(self) -> None:
        self.assertIsInstance(_input_classes(), str)


class CustomAuthenticationFormTests(TestCase):
    def test_username_uses_email_input_widget(self) -> None:
        form = CustomAuthenticationForm()
        self.assertEqual(form.fields["username"].widget.__class__.__name__, "EmailInput")

    def test_password_uses_password_input_widget(self) -> None:
        form = CustomAuthenticationForm()
        self.assertEqual(form.fields["password"].widget.__class__.__name__, "PasswordInput")

    def test_widgets_have_tailwind_classes(self) -> None:
        form = CustomAuthenticationForm()
        self.assertIn("rounded-lg", form.fields["username"].widget.attrs.get("class", ""))
        self.assertIn("rounded-lg", form.fields["password"].widget.attrs.get("class", ""))


class CustomPasswordChangeFormTests(TestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(email="u@example.com", password="pass")

    def test_fields_use_password_input_widget(self) -> None:
        form = CustomPasswordChangeForm(user=self.user)
        for field_name in ("old_password", "new_password1", "new_password2"):
            self.assertEqual(
                form.fields[field_name].widget.__class__.__name__,
                "PasswordInput",
                f"{field_name} should use PasswordInput",
            )

    def test_widgets_have_tailwind_classes(self) -> None:
        form = CustomPasswordChangeForm(user=self.user)
        for field_name in ("old_password", "new_password1", "new_password2"):
            self.assertIn("rounded-lg", form.fields[field_name].widget.attrs.get("class", ""))


class CustomPasswordResetFormTests(TestCase):
    def test_email_uses_email_input_widget(self) -> None:
        form = CustomPasswordResetForm()
        self.assertEqual(form.fields["email"].widget.__class__.__name__, "EmailInput")

    def test_email_widget_has_tailwind_classes(self) -> None:
        form = CustomPasswordResetForm()
        self.assertIn("rounded-lg", form.fields["email"].widget.attrs.get("class", ""))


class CustomSetPasswordFormTests(TestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(email="u@example.com", password="pass")

    def test_fields_use_password_input_widget(self) -> None:
        form = CustomSetPasswordForm(user=self.user)
        for field_name in ("new_password1", "new_password2"):
            self.assertEqual(
                form.fields[field_name].widget.__class__.__name__,
                "PasswordInput",
            )


class SignUpFormTests(TestCase):
    def test_meta_model_is_user(self) -> None:
        self.assertEqual(SignUpForm.Meta.model, User)

    def test_meta_fields_is_email_only(self) -> None:
        self.assertEqual(SignUpForm.Meta.fields, ("email",))

    def test_valid_data_creates_user(self) -> None:
        form = SignUpForm(data={
            "email": "new@example.com",
            "password1": "secure-pass-123!",
            "password2": "secure-pass-123!",
        })
        self.assertTrue(form.is_valid(), form.errors)
        user = form.save()
        self.assertEqual(user.email, "new@example.com")
        self.assertTrue(user.check_password("secure-pass-123!"))

    def test_mismatched_passwords_invalid(self) -> None:
        form = SignUpForm(data={
            "email": "new@example.com",
            "password1": "secure-pass-123!",
            "password2": "different-pass-456!",
        })
        self.assertFalse(form.is_valid())
        self.assertIn("password2", form.errors)

    def test_missing_email_invalid(self) -> None:
        form = SignUpForm(data={
            "email": "",
            "password1": "secure-pass-123!",
            "password2": "secure-pass-123!",
        })
        self.assertFalse(form.is_valid())
        self.assertIn("email", form.errors)

    def test_duplicate_email_invalid(self) -> None:
        User.objects.create_user(email="dup@example.com", password="pass")
        form = SignUpForm(data={
            "email": "dup@example.com",
            "password1": "secure-pass-123!",
            "password2": "secure-pass-123!",
        })
        self.assertFalse(form.is_valid())

    def test_email_widget_has_tailwind_classes(self) -> None:
        form = SignUpForm()
        self.assertIn("rounded-lg", form.fields["email"].widget.attrs.get("class", ""))

    def test_password_widgets_have_tailwind_classes(self) -> None:
        form = SignUpForm()
        self.assertIn("rounded-lg", form.fields["password1"].widget.attrs.get("class", ""))
        self.assertIn("rounded-lg", form.fields["password2"].widget.attrs.get("class", ""))
