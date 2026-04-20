from django.test import SimpleTestCase

from core.sentry_scrub import REDACTED, scrub_event


class ScrubEventTests(SimpleTestCase):
    def test_none_event_returned_unchanged(self):
        self.assertIsNone(scrub_event(None))

    def test_user_keeps_id_only(self):
        event = {"user": {"id": 42, "email": "a@b.com", "username": "alice"}}
        scrub_event(event)
        self.assertEqual(event["user"], {"id": 42})

    def test_request_headers_authorization_and_cookie_redacted(self):
        event = {
            "request": {
                "headers": {
                    "Authorization": "Bearer sk-secret",
                    "Cookie": "sessionid=xyz",
                    "User-Agent": "Mozilla/5.0",
                }
            }
        }
        scrub_event(event)
        headers = event["request"]["headers"]
        self.assertEqual(headers["Authorization"], REDACTED)
        self.assertEqual(headers["Cookie"], REDACTED)
        self.assertEqual(headers["User-Agent"], "Mozilla/5.0")

    def test_request_cookies_and_data_redacted(self):
        event = {
            "request": {
                "cookies": {"sessionid": "xyz"},
                "data": {"email": "a@b.com", "password": "hunter2"},
            }
        }
        scrub_event(event)
        self.assertEqual(event["request"]["cookies"], REDACTED)
        self.assertEqual(event["request"]["data"], REDACTED)

    def test_query_string_with_sensitive_keys_redacted(self):
        event = {"request": {"query_string": "page=2&token=abc123"}}
        scrub_event(event)
        self.assertEqual(event["request"]["query_string"], REDACTED)

    def test_query_string_without_sensitive_keys_preserved(self):
        event = {"request": {"query_string": "page=2&sort=desc"}}
        scrub_event(event)
        self.assertEqual(event["request"]["query_string"], "page=2&sort=desc")

    def test_extra_deny_keys_redacted_case_insensitive(self):
        event = {
            "extra": {
                "Email": "user@example.com",
                "api_key": "sk-live-123",
                "Prompt": [{"role": "user", "content": "hi"}],
                "harmless": "ok",
            }
        }
        scrub_event(event)
        self.assertEqual(event["extra"]["Email"], REDACTED)
        self.assertEqual(event["extra"]["api_key"], REDACTED)
        self.assertEqual(event["extra"]["Prompt"], REDACTED)
        self.assertEqual(event["extra"]["harmless"], "ok")

    def test_extra_nested_deny_keys_redacted(self):
        event = {
            "extra": {
                "outer": {
                    "inner": {"token": "t1", "value": "kept"},
                    "messages": ["m1"],
                }
            }
        }
        scrub_event(event)
        self.assertEqual(event["extra"]["outer"]["inner"]["token"], REDACTED)
        self.assertEqual(event["extra"]["outer"]["inner"]["value"], "kept")
        self.assertEqual(event["extra"]["outer"]["messages"], REDACTED)

    def test_breadcrumb_query_message_redacted(self):
        event = {
            "breadcrumbs": {
                "values": [
                    {
                        "category": "query",
                        "message": "SELECT * FROM accounts_user WHERE email='a@b.com'",
                    },
                    {"category": "navigation", "message": "/threads/1"},
                ]
            }
        }
        scrub_event(event)
        crumbs = event["breadcrumbs"]["values"]
        self.assertEqual(crumbs[0]["message"], REDACTED)
        self.assertEqual(crumbs[1]["message"], "/threads/1")

    def test_breadcrumb_list_form_also_scrubbed(self):
        event = {
            "breadcrumbs": [
                {"category": "httplib", "message": "POST /v1/chat", "data": {"body": "secret"}}
            ]
        }
        scrub_event(event)
        self.assertEqual(event["breadcrumbs"][0]["message"], REDACTED)
        self.assertEqual(event["breadcrumbs"][0]["data"], REDACTED)

    def test_missing_sections_are_no_op(self):
        event = {"tags": {"x": "y"}}
        self.assertEqual(scrub_event(event), {"tags": {"x": "y"}})

    def test_contexts_with_deny_keys_redacted(self):
        event = {"contexts": {"state": {"user_email": "a@b.com", "content": "hi"}}}
        scrub_event(event)
        # user_email is not in the deny list (email is), content is.
        self.assertEqual(event["contexts"]["state"]["content"], REDACTED)
        self.assertEqual(event["contexts"]["state"]["user_email"], "a@b.com")
