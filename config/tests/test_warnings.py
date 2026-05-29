"""Guard the benign-warning suppression configured in ``config.settings``.

Importing ``config.settings`` registers a ``warnings`` filter that drops the
cosmetic ``Pydantic serializer warnings`` emitted when Sentry's genai
instrumentation re-serializes OpenAI parsed responses. Without it the warning
reaches the ``py.warnings`` logger and Sentry's ``LoggingIntegration`` records
it as an error-level event. Regression guard for WILFRED-57 / WILFRED-58.
"""

import warnings

from django.test import SimpleTestCase

SAMPLE = (
    "Pydantic serializer warnings:\n"
    "  PydanticSerializationUnexpectedValue(Expected `none` - serialized value "
    "may not be as expected [field_name='parsed'])"
)


class PydanticSerializerWarningFilterTests(SimpleTestCase):
    def _matching_filters(self):
        return [
            entry
            for entry in warnings.filters
            if entry[0] == "ignore"
            and entry[2] is UserWarning
            and entry[1] is not None
            and entry[1].match(SAMPLE)
        ]

    def test_filter_is_registered(self):
        self.assertTrue(
            self._matching_filters(),
            "config.settings must register an 'ignore' filter for the benign "
            "Pydantic serializer UserWarning",
        )

    def test_filter_is_scoped_not_a_catch_all(self):
        # The filter must target the Pydantic message only — not swallow every
        # UserWarning, which would hide unrelated, potentially real warnings.
        for entry in self._matching_filters():
            self.assertIsNone(
                entry[1].match("some unrelated user warning"),
                "the Pydantic serializer filter must not match arbitrary warnings",
            )
