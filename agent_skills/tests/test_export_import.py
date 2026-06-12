"""Tests for skill export / import (download + upload)."""

import json

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import Membership, Organization
from agent_skills.models import (
    MAX_INSTRUCTIONS_CHARS,
    MAX_TEMPLATE_CHARS,
    AgentSkill,
    SkillTemplate,
)
from agent_skills.services import (
    EXPORT_VERSION,
    MAX_SKILLS_PER_IMPORT,
    SkillImportError,
    dump_skills_json,
    export_skill,
    import_skill,
    parse_skill_export,
)

User = get_user_model()


def _make_skill(**overrides):
    """Create a system skill with multi-line text + templates for export tests."""
    defaults = dict(
        slug="research",
        name="Researcher",
        emoji="🔎",
        description="A skill that researches things.",
        instructions="# Researcher\n\nDo the research.\nThen report.",
        tool_names=["view_template"],
        level="system",
    )
    defaults.update(overrides)
    skill = AgentSkill.objects.create(**defaults)
    SkillTemplate.objects.create(skill=skill, name="Outline", content="# Title\n\n## Findings")
    SkillTemplate.objects.create(skill=skill, name="Notes", content="single line")
    return skill


class ExportSkillTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.skill = _make_skill()

    def test_export_shape_uses_line_arrays(self):
        data = export_skill(self.skill)
        self.assertEqual(data["slug"], "research")
        self.assertEqual(data["name"], "Researcher")
        self.assertEqual(data["emoji"], "🔎")
        # Multi-line fields become arrays of lines.
        self.assertEqual(
            data["instructions"],
            ["# Researcher", "", "Do the research.", "Then report."],
        )
        self.assertEqual(data["description"], ["A skill that researches things."])
        self.assertEqual(data["tool_names"], ["view_template"])
        # Templates ordered by name, content as line arrays.
        self.assertEqual([t["name"] for t in data["templates"]], ["Notes", "Outline"])
        outline = next(t for t in data["templates"] if t["name"] == "Outline")
        self.assertEqual(outline["content"], ["# Title", "", "## Findings"])

    def test_dump_json_is_readable_envelope(self):
        raw = dump_skills_json([self.skill])
        # Envelope present.
        data = json.loads(raw)
        self.assertEqual(data["wilfred_skill_export"], EXPORT_VERSION)
        self.assertEqual(len(data["skills"]), 1)
        # ensure_ascii=False keeps emoji literal (not \uXXXX) for readability.
        self.assertIn("🔎", raw)
        self.assertNotIn("\\ud", raw.lower())
        # Newlines are NOT crammed into one escaped string — they're array rows.
        self.assertNotIn("Do the research.\\nThen report.", raw)


class ParseImportServiceTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="imp@example.com", password="pw")
        self.other = User.objects.create_user(email="src@example.com", password="pw")

    def test_round_trip_identity(self):
        source = _make_skill()
        raw = dump_skills_json([source]).encode("utf-8")
        payloads = parse_skill_export(raw)
        imported = import_skill(self.user, payloads[0])

        self.assertEqual(imported.slug, source.slug)  # different owner → no dedup
        self.assertEqual(imported.name, source.name)
        self.assertEqual(imported.emoji, source.emoji)
        self.assertEqual(imported.description, source.description)
        self.assertEqual(imported.instructions, source.instructions)
        self.assertEqual(imported.tool_names, source.tool_names)
        self.assertEqual(imported.level, "user")
        self.assertEqual(imported.created_by, self.user)
        self.assertIsNone(imported.parent)
        # Templates survive with exact content.
        tmpls = {t.name: t.content for t in imported.templates.all()}
        self.assertEqual(tmpls["Outline"], "# Title\n\n## Findings")
        self.assertEqual(tmpls["Notes"], "single line")

    def test_import_strips_non_skill_tool_names(self):
        # view_template is a skills-section tool (kept); create_subagent is a
        # chat-section tool and totally_made_up_tool is unknown (both dropped).
        payload = {
            "name": "Tooly",
            "instructions": "x",
            "tool_names": ["view_template", "create_subagent", "totally_made_up_tool"],
            "templates": [],
        }
        skill = import_skill(self.user, payload)
        self.assertEqual(skill.tool_names, ["view_template"])

    def test_import_truncates_oversized_instructions(self):
        raw = json.dumps({"skills": [{
            "name": "Huge",
            "instructions": ["x" * (MAX_INSTRUCTIONS_CHARS + 5000)],
        }]}).encode("utf-8")
        payloads = parse_skill_export(raw)
        # Normalization caps the payload...
        self.assertEqual(len(payloads[0]["instructions"]), MAX_INSTRUCTIONS_CHARS)
        # ...and so does the persisted skill.
        skill = import_skill(self.user, payloads[0])
        self.assertEqual(len(skill.instructions), MAX_INSTRUCTIONS_CHARS)

    def test_accepts_string_form_for_text_fields(self):
        # A hand-author may collapse a line-array back to a plain string.
        raw = json.dumps(
            {"skills": [{"name": "Hand", "instructions": "line one\nline two"}]}
        ).encode("utf-8")
        payloads = parse_skill_export(raw)
        skill = import_skill(self.user, payloads[0])
        self.assertEqual(skill.instructions, "line one\nline two")

    def test_accepts_bare_skill_dict(self):
        raw = json.dumps({"name": "Bare", "instructions": ["hi"]}).encode("utf-8")
        payloads = parse_skill_export(raw)
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["name"], "Bare")

    def test_slug_deduped_per_user(self):
        AgentSkill.objects.create(
            slug="research", name="Mine", instructions="i",
            level="user", created_by=self.user,
        )
        source = _make_skill()
        payloads = parse_skill_export(dump_skills_json([source]).encode("utf-8"))
        imported = import_skill(self.user, payloads[0])
        self.assertEqual(imported.slug, "research-1")

    def test_default_name_when_missing(self):
        payloads = parse_skill_export(json.dumps({"skills": [{"instructions": ["x"]}]}).encode())
        self.assertEqual(payloads[0]["name"], "Imported skill")

    def test_field_caps(self):
        raw = json.dumps({"skills": [{
            "name": "x" * 300,
            "description": ["y" * 2000],
            "instructions": ["ok"],
        }]}).encode("utf-8")
        payloads = parse_skill_export(raw)
        self.assertEqual(len(payloads[0]["name"]), 255)
        self.assertEqual(len(payloads[0]["description"]), 1024)

    def test_duplicate_template_names_keep_first(self):
        raw = json.dumps({"skills": [{
            "name": "Dup",
            "instructions": ["x"],
            "templates": [
                {"name": "T", "content": ["first"]},
                {"name": "T", "content": ["second"]},
            ],
        }]}).encode("utf-8")
        payloads = parse_skill_export(raw)
        skill = import_skill(self.user, payloads[0])
        templates = list(skill.templates.all())
        self.assertEqual(len(templates), 1)
        self.assertEqual(templates[0].content, "first")

    def test_import_truncates_oversized_template_content(self):
        raw = json.dumps({"skills": [{
            "name": "Big Tmpl",
            "instructions": ["ok"],
            "templates": [
                {"name": "Huge", "content": ["x" * (MAX_TEMPLATE_CHARS + 5000)]},
            ],
        }]}).encode("utf-8")
        payloads = parse_skill_export(raw)
        # Normalization caps the payload...
        self.assertEqual(len(payloads[0]["templates"][0]["content"]), MAX_TEMPLATE_CHARS)
        # ...and so does the persisted template.
        skill = import_skill(self.user, payloads[0])
        tmpl = skill.templates.get(name="Huge")
        self.assertEqual(len(tmpl.content), MAX_TEMPLATE_CHARS)

    def test_rejects_too_many_skills(self):
        """A file over the per-import skill cap is rejected before any
        normalization work (the slug dedup is O(N²) per skill)."""
        entries = [
            {"name": f"S{i}", "instructions": ["x"]}
            for i in range(MAX_SKILLS_PER_IMPORT + 1)
        ]
        raw = json.dumps({"skills": entries}).encode("utf-8")
        with self.assertRaises(SkillImportError) as ctx:
            parse_skill_export(raw)
        self.assertIn(str(MAX_SKILLS_PER_IMPORT), str(ctx.exception))

    def test_accepts_exactly_max_skills(self):
        entries = [
            {"name": f"S{i}", "instructions": ["x"]}
            for i in range(MAX_SKILLS_PER_IMPORT)
        ]
        raw = json.dumps({"skills": entries}).encode("utf-8")
        payloads = parse_skill_export(raw)
        self.assertEqual(len(payloads), MAX_SKILLS_PER_IMPORT)

    def test_rejects_invalid_json(self):
        with self.assertRaises(SkillImportError):
            parse_skill_export(b"this is not json")

    def test_rejects_newer_version(self):
        raw = json.dumps({"wilfred_skill_export": 999, "skills": [{"name": "n", "instructions": ["i"]}]}).encode()
        with self.assertRaises(SkillImportError):
            parse_skill_export(raw)

    def test_rejects_empty_skills_list(self):
        with self.assertRaises(SkillImportError):
            parse_skill_export(json.dumps({"skills": []}).encode())


@override_settings(ALLOWED_HOSTS=["testserver"])
class DownloadViewTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="dl@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="Acme", slug="acme")
        Membership.objects.create(user=self.user, org=self.org, role=Membership.Role.MEMBER)

        self.system = AgentSkill.objects.create(
            slug="sys", name="Sys", instructions="i", level="system",
        )
        # Org must explicitly enable a system skill for it to be visible to
        # members (matches get_skill_for_user's org-hiding rules).
        self.org.preferences = {"skills": {"sys": {"enabled": True}}}
        self.org.save(update_fields=["preferences"])
        self.mine = AgentSkill.objects.create(
            slug="mine", name="Mine", instructions="i", level="user", created_by=self.user,
        )
        self.other_org = Organization.objects.create(name="Other", slug="other")
        self.foreign = AgentSkill.objects.create(
            slug="foreign", name="Foreign", instructions="i",
            level="org", organization=self.other_org,
        )

    def _download(self, skill):
        return self.client.get(reverse("agent_skills_download", kwargs={"skill_id": skill.id}))

    def test_requires_login(self):
        resp = self._download(self.system)
        self.assertEqual(resp.status_code, 302)

    def test_download_own_skill(self):
        self.client.force_login(self.user)
        resp = self._download(self.mine)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("attachment", resp["Content-Disposition"])
        self.assertIn("mine.json", resp["Content-Disposition"])
        self.assertEqual(resp["Content-Type"], "application/json")
        data = json.loads(b"".join(resp.streaming_content).decode("utf-8"))
        self.assertEqual(data["skills"][0]["slug"], "mine")

    def test_download_system_skill_allowed(self):
        self.client.force_login(self.user)
        resp = self._download(self.system)
        self.assertEqual(resp.status_code, 200)

    def test_download_org_hidden_system_skill_redirects(self):
        """A system skill the org hasn't enabled is hidden → download blocked."""
        hidden = AgentSkill.objects.create(
            slug="hidden-sys", name="Hidden", instructions="i", level="system",
        )
        self.client.force_login(self.user)
        resp = self._download(hidden)
        self.assertEqual(resp.status_code, 302)

    def test_download_foreign_org_skill_redirects(self):
        self.client.force_login(self.user)
        resp = self._download(self.foreign)
        self.assertEqual(resp.status_code, 302)


@override_settings(ALLOWED_HOSTS=["testserver"])
class ImportViewTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="up@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])

    def _upload(self, payload_obj, filename="skill.json"):
        raw = json.dumps(payload_obj).encode("utf-8")
        f = SimpleUploadedFile(filename, raw, content_type="application/json")
        return self.client.post(reverse("agent_skills_import"), {"file": f}, follow=False)

    def test_requires_login(self):
        resp = self._upload({"skills": [{"name": "n", "instructions": ["i"]}]})
        self.assertEqual(resp.status_code, 302)
        self.assertFalse(AgentSkill.objects.filter(name="n").exists())

    def test_single_import_redirects_to_detail(self):
        self.client.force_login(self.user)
        resp = self._upload({"skills": [{"slug": "shared", "name": "Shared Skill", "instructions": ["do it"]}]})
        self.assertEqual(resp.status_code, 302)
        skill = AgentSkill.objects.get(name="Shared Skill")
        self.assertEqual(skill.level, "user")
        self.assertEqual(skill.created_by, self.user)
        self.assertIn(str(skill.id), resp["Location"])

    def test_multi_import_redirects_to_list(self):
        self.client.force_login(self.user)
        resp = self._upload({"skills": [
            {"name": "One", "instructions": ["a"]},
            {"name": "Two", "instructions": ["b"]},
        ]})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp["Location"], reverse("agent_skills_list"))
        self.assertTrue(AgentSkill.objects.filter(name="One", created_by=self.user).exists())
        self.assertTrue(AgentSkill.objects.filter(name="Two", created_by=self.user).exists())

    def test_bad_file_creates_nothing(self):
        self.client.force_login(self.user)
        f = SimpleUploadedFile("bad.json", b"not json at all", content_type="application/json")
        resp = self.client.post(reverse("agent_skills_import"), {"file": f})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(AgentSkill.objects.filter(level="user").count(), 0)

    def test_missing_file_creates_nothing(self):
        self.client.force_login(self.user)
        resp = self.client.post(reverse("agent_skills_import"), {})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(AgentSkill.objects.filter(level="user").count(), 0)
