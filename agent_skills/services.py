"""Skill resolution and access control."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Callable

from django.db import IntegrityError, transaction
from django.utils.text import slugify

from agent_skills.models import (
    MAX_INSTRUCTIONS_CHARS,
    MAX_TEMPLATE_CHARS,
    AgentSkill,
    SkillTemplate,
)

if TYPE_CHECKING:
    from django.contrib.auth import get_user_model

    User = get_user_model()


# Priority order for shadowing — higher level wins.
_LEVEL_ORDER = {"system": 0, "org": 1, "user": 2}


def _next_free_slug(base_slug: str, taken: Callable[[str], bool]) -> str:
    """Return ``base_slug`` or the first ``base-N`` variant not ``taken``.

    Suffixes eat into the 64-char SlugField limit so the result always fits.
    """
    slug = base_slug
    counter = 1
    while taken(slug):
        suffix = f"-{counter}"
        slug = base_slug[: 64 - len(suffix)] + suffix
        counter += 1
    return slug


def _save_with_free_slug(base_slug: str, taken: Callable[[str], bool], persist, *, attempts: int = 3):
    """Run ``persist(slug)`` with a free slug, retrying on IntegrityError.

    The check-then-write slug dedup is racy: a concurrent request can claim
    the slug between the existence check and the INSERT/UPDATE, tripping the
    partial unique constraints. Each attempt recomputes the next free slug
    and runs inside ``transaction.atomic()`` so a failed write is rolled back
    before the retry. The last attempt re-raises.
    """
    for attempt in range(attempts):
        slug = _next_free_slug(base_slug, taken)
        try:
            with transaction.atomic():
                return persist(slug)
        except IntegrityError:
            if attempt == attempts - 1:
                raise


def skill_tool_audience_ok(tool_audience: str, skill_audience: str) -> bool:
    """Whether a tool of ``tool_audience`` may be listed by a skill of ``skill_audience``.

    ``shared`` tools work in any context. Otherwise a tool may only be carried
    by a skill of the same audience — a ``main`` skill can't grant a
    sub-agent-only tool and vice versa. A ``shared`` skill (seed-only) must run
    in both contexts, so it may only carry ``shared`` tools.
    """
    if tool_audience == "shared":
        return True
    if skill_audience == "shared":
        return False
    return tool_audience == skill_audience


def filter_to_skill_tools(tool_names, skill_audience: str | None = None) -> list[str]:
    """Allow-list: keep only registered ``section == "skills"`` tools, in order.

    Drops names unknown to the registry and tools from any other section
    (always-available chat tools, data-room doc tools). This is the single
    chokepoint enforced at save, import, edit, and runtime resolution so a
    hand-crafted import or a stale stored row can't smuggle a non-skill tool
    into the LLM's tool list.

    When ``skill_audience`` is given, also drop tools whose audience is
    incompatible with the carrying skill (see :func:`skill_tool_audience_ok`),
    so a sub-agent specialization can never grant a main-only tool and a main
    skill can never grant a sub-agent-only tool.
    """
    from llm.tools.registry import get_tool_registry

    registry = get_tool_registry()
    out: list[str] = []
    for name in tool_names or []:
        name = name if isinstance(name, str) else str(name)
        tool = registry.get_tool(name)
        if tool is None or getattr(tool, "section", "chat") != "skills":
            continue
        if skill_audience is not None and not skill_tool_audience_ok(
            getattr(tool, "audience", "shared"), skill_audience
        ):
            continue
        out.append(name)
    return out


def _org_disabled_info(user) -> tuple[set[str], set[str]]:
    """Return ``(all_tier_disabled, system_tier_disabled)`` slug sets.

    *all_tier_disabled*: slugs an admin explicitly set ``enabled: False`` —
    hides every tier (system, org, user) of that slug.

    *system_tier_disabled*: system-skill slugs not explicitly enabled — hides
    only the system tier.  Org / user skills sharing the slug stay visible.

    The org settings page queries ``AgentSkill`` directly and intentionally
    bypasses this so admins can still toggle disabled skills back on.
    """
    from accounts.models import Membership

    membership = Membership.objects.filter(user=user).select_related("org").first()
    if not membership or not membership.org:
        return set(), set()
    org_skills = (membership.org.preferences or {}).get("skills") or {}

    all_tier_disabled = {
        slug for slug, pref in org_skills.items()
        if isinstance(pref, dict) and pref.get("enabled") is False
    }

    system_slugs = set(
        AgentSkill.objects.filter(level="system", is_active=True)
        .values_list("slug", flat=True)
    )
    system_tier_disabled = set()
    for slug in system_slugs:
        pref = org_skills.get(slug)
        if not isinstance(pref, dict) or pref.get("enabled") is not True:
            system_tier_disabled.add(slug)

    return all_tier_disabled, system_tier_disabled


def _is_org_hidden(skill, all_disabled: set[str], system_disabled: set[str]) -> bool:
    if skill.slug in all_disabled:
        return True
    if skill.level == "system" and skill.slug in system_disabled:
        return True
    return False


def get_accessible_skills(user) -> list[AgentSkill]:
    """Return every active skill the user has access to (no shadowing)."""
    from django.db.models import Q

    from accounts.models import Membership

    membership = Membership.objects.filter(user=user).select_related("org").first()

    q = Q(level="system")
    if membership:
        q |= Q(level="org", organization=membership.org)
    q |= Q(level="user", created_by=user)

    all_disabled, system_disabled = _org_disabled_info(user)
    return [
        s for s in AgentSkill.objects.filter(q, is_active=True)
        if not _is_org_hidden(s, all_disabled, system_disabled)
    ]


def shadowing_default(candidates: list[AgentSkill]) -> AgentSkill:
    """Return the highest-priority candidate (system < org < user)."""
    return max(candidates, key=lambda s: _LEVEL_ORDER.get(s.level, 0))


def get_user_skill_prefs(user) -> dict:
    """Return the per-user skill preferences sub-dict from UserSettings."""
    from accounts.models import UserSettings

    try:
        us = UserSettings.objects.get(user=user)
    except UserSettings.DoesNotExist:
        return {}
    prefs = us.preferences or {}
    skills_prefs = prefs.get("skills", {})
    return skills_prefs if isinstance(skills_prefs, dict) else {}


# Audience groupings for resolving the effective skill list per surface.
_MAIN_AUDIENCES = frozenset({"main", "shared"})
_SUBAGENT_AUDIENCES = frozenset({"subagent", "shared"})


def _resolve_available_skills(user, audiences: frozenset[str]) -> list[AgentSkill]:
    """Shadow-resolve the accessible skills, scoped to an audience group.

    Filters by audience *before* grouping by slug so a main skill and a
    sub-agent skill that happen to share a slug across tiers never shadow each
    other — each surface resolves only within its own audience group.
    """
    accessible = [
        s for s in get_accessible_skills(user)
        if getattr(s, "audience", "main") in audiences
    ]
    by_slug: dict[str, list[AgentSkill]] = {}
    for skill in accessible:
        by_slug.setdefault(skill.slug, []).append(skill)

    user_skill_prefs = get_user_skill_prefs(user)

    result: list[AgentSkill] = []
    for slug, candidates in by_slug.items():
        pref = user_skill_prefs.get(slug)
        if pref is not None:
            sel_id = pref.get("selected_skill_id") if isinstance(pref, dict) else None
            if sel_id is None:
                continue  # explicitly disabled
            chosen = next((c for c in candidates if str(c.id) == str(sel_id)), None)
            if chosen is None:
                # Stale selection (skill was deleted or access was revoked).
                chosen = shadowing_default(candidates)
            result.append(chosen)
        else:
            result.append(shadowing_default(candidates))

    return sorted(result, key=lambda s: s.name)


def get_available_skills(user) -> list[AgentSkill]:
    """Return the effective *main-agent* skill list for a user.

    For each slug, the user's explicit selection in
    ``UserSettings.preferences["skills"][slug]`` wins. Absent rows fall back
    to shadowing defaults (system < org < user). A row whose
    ``selected_skill_id`` is ``None`` means the user has explicitly disabled
    that slug. Only ``main``/``shared`` audience skills are surfaced — sub-agent
    specializations never attach to a main thread.
    """
    return _resolve_available_skills(user, _MAIN_AUDIENCES)


def get_subagent_skills(user) -> list[AgentSkill]:
    """Return the effective *sub-agent specialization* list for a user.

    The sub-agent-audience analogue of :func:`get_available_skills`: same
    access gate, org-disable, and per-user shadowing, scoped to
    ``subagent``/``shared`` audience skills.
    """
    return _resolve_available_skills(user, _SUBAGENT_AUDIENCES)


def get_skill_for_user(user, skill_id: str) -> AgentSkill | None:
    """Return the skill if the user has access, else None.

    Access rules:
    - system: always accessible
    - org: user must be a member of that org
    - user: must be the creator
    """
    from accounts.models import Membership

    try:
        skill = AgentSkill.objects.get(pk=skill_id, is_active=True)
    except AgentSkill.DoesNotExist:
        return None

    all_disabled, system_disabled = _org_disabled_info(user)
    if _is_org_hidden(skill, all_disabled, system_disabled):
        return None

    if skill.level == "system":
        return skill

    if skill.level == "org":
        if skill.organization_id and Membership.objects.filter(
            user=user, org_id=skill.organization_id
        ).exists():
            return skill
        return None

    if skill.level == "user":
        if skill.created_by_id == user.pk:
            return skill
        return None

    return None


def can_edit_skill(user, skill: AgentSkill) -> bool:
    """Check if a user can edit a skill.

    - System skills: never editable
    - Org skills: editable by org admins
    - User skills: editable by the creator
    """
    from accounts.models import Membership

    if skill.level == "system":
        return False

    if skill.level == "org":
        return Membership.objects.filter(
            user=user, org_id=skill.organization_id, role=Membership.Role.ADMIN
        ).exists()

    if skill.level == "user":
        return skill.created_by_id == user.pk

    return False


def get_editable_skill_for_user(user, slug: str) -> AgentSkill | None:
    """Look up a skill by slug (with shadowing) and return it only if editable.

    When the visible tier isn't editable (e.g. the user selected the org or
    system version) — or the slug is hidden entirely (explicitly disabled, or
    org-disabled) — fall back to the user's own skill with that slug: the
    owner can always edit their own skill, and "edit/delete my skill X" should
    not fail just because X is currently shadowed or toggled off.
    """
    skills = get_available_skills(user)
    for skill in skills:
        if skill.slug == slug:
            if can_edit_skill(user, skill):
                return skill
            break  # visible tier not editable -> fall back to owned
    return AgentSkill.objects.filter(
        slug=slug, level="user", created_by=user, is_active=True
    ).first()


def create_user_skill(
    user, name: str, slug: str | None = None, *, audience: str = AgentSkill.Audience.MAIN
) -> AgentSkill:
    """Create a user-level skill with auto-generated slug."""
    if audience not in AgentSkill.Audience.values or audience == AgentSkill.Audience.SHARED:
        audience = AgentSkill.Audience.MAIN
    if not slug:
        slug = slugify(name)[:64]
    if not slug:
        slug = "skill"

    return _save_with_free_slug(
        slug,
        lambda s: AgentSkill.objects.filter(
            slug=s, level="user", created_by=user
        ).exists(),
        lambda s: AgentSkill.objects.create(
            slug=s,
            name=name,
            instructions="",
            description="",
            audience=audience,
            level="user",
            created_by=user,
        ),
    )


# Matches "Base name" or "Base name (3)"
_NAME_SUFFIX_RE = re.compile(r"^(?P<base>.*?)(?: \((?P<num>\d+)\))?$")


def _next_user_skill_name(user, source_name: str) -> str:
    """Find the next free copy name like 'Foo (1)', 'Foo (2)' for this user.

    Strips an existing trailing ``(N)`` from ``source_name`` so that copying
    "Foo (2)" produces "Foo (3)" rather than "Foo (2) (1)".
    """
    match = _NAME_SUFFIX_RE.match(source_name.strip())
    base = match.group("base") if match else source_name.strip()
    base = base.strip() or source_name.strip() or "Skill"

    pattern = re.compile(rf"^{re.escape(base)}(?: \((\d+)\))?$")
    used: set[int] = set()
    base_taken = False
    existing_names = AgentSkill.objects.filter(
        level="user", created_by=user
    ).values_list("name", flat=True)
    for existing in existing_names:
        m = pattern.match(existing)
        if not m:
            continue
        num = m.group(1)
        if num is None:
            base_taken = True
        else:
            used.add(int(num))

    if not base_taken:
        return base
    n = 1
    while n in used:
        n += 1
    return f"{base} ({n})"


def fork_skill(
    user, source_skill: AgentSkill, *, copy_templates: bool = True,
    audience: str | None = None,
) -> AgentSkill:
    """Fork a skill to a user-level copy, including templates.

    Parent semantics:
    - Copying a system or org skill: ``parent = source_skill``.
    - Copying a user skill: ``parent = source_skill.parent`` so iterate-and-
      delete chains stay flat (the user can throw away the intermediate
      version without losing the link to the original system/org source).

    The new skill's name receives a ``(1)``/``(2)`` suffix so the user can
    distinguish many copies. The slug uses the existing ``-1``/``-2``
    deduplication.

    ``copy_templates`` defaults to True for callers that want a complete
    standalone copy. The detail-page form action passes ``False`` because
    it then re-creates the templates from the submitted form data; copying
    them here would clash with the ``unique_template_per_skill`` constraint.
    """
    new_name = _next_user_skill_name(user, source_skill.name)

    parent = source_skill.parent if source_skill.level == "user" else source_skill

    # A personal copy is never "shared" (shared is seed-only). When the caller
    # forces an audience (the Skills page tab the copy was made from), drop a
    # shared source down to that audience; otherwise inherit the source's.
    # tool_names are copied verbatim — runtime resolution re-filters them by
    # audience, so a faithful copy stays a faithful copy.
    new_audience = audience or source_skill.audience
    if new_audience == AgentSkill.Audience.SHARED:
        new_audience = AgentSkill.Audience.MAIN

    new_skill = _save_with_free_slug(
        source_skill.slug,
        lambda s: AgentSkill.objects.filter(
            slug=s, level="user", created_by=user
        ).exists(),
        lambda s: AgentSkill.objects.create(
            slug=s,
            name=new_name,
            emoji=source_skill.emoji,
            description=source_skill.description,
            instructions=source_skill.instructions,
            tool_names=list(source_skill.tool_names or []),
            audience=new_audience,
            level="user",
            created_by=user,
            parent=parent,
        ),
    )

    if copy_templates:
        for tmpl in source_skill.templates.all():
            SkillTemplate.objects.create(
                skill=new_skill,
                name=tmpl.name,
                content=tmpl.content,
            )

    return new_skill


def set_user_skill_selection(user, skill: AgentSkill, enabled: bool) -> dict:
    """Toggle the user's enablement for the slug owned by ``skill``.

    Writes ``UserSettings.preferences["skills"][skill.slug]`` to either
    ``{"selected_skill_id": <skill.id>}`` (enabled) or
    ``{"selected_skill_id": None}`` (disabled). The mutual-exclusivity
    invariant is enforced naturally: only one ``selected_skill_id`` exists
    per slug.

    Returns a dict describing what was replaced (if anything), so the caller
    can craft a "Disabled the org version" toast.
    """
    from accounts.models import UserSettings
    from accounts.services import update_user_preferences

    us = UserSettings.objects.filter(user=user).first()
    skills_prefs = dict(((us.preferences if us else None) or {}).get("skills") or {})

    # Detect what was previously active for this slug — either an explicit
    # selection or, if none, the shadowing default among accessible skills.
    accessible = get_accessible_skills(user)
    candidates = [s for s in accessible if s.slug == skill.slug]
    previous: AgentSkill | None = None
    existing_pref = skills_prefs.get(skill.slug)
    if isinstance(existing_pref, dict):
        prev_id = existing_pref.get("selected_skill_id")
        if prev_id is not None:
            previous = next((c for c in candidates if str(c.id) == str(prev_id)), None)
    elif candidates:
        previous = shadowing_default(candidates)

    def mutate(prefs):
        sp = prefs.get("skills") or {}
        sp[skill.slug] = {"selected_skill_id": str(skill.id) if enabled else None}
        prefs["skills"] = sp

    update_user_preferences(user, mutate)

    replaced = None
    if enabled and previous and previous.pk != skill.pk:
        replaced = {
            "id": str(previous.id),
            "name": previous.name,
            "level": previous.level,
        }

    return {
        "now_active": bool(enabled),
        "replaced": replaced,
    }


def migrate_skill_slug_prefs(skill, old_slug: str, new_slug: str) -> None:
    """Re-key user skill prefs after ``skill``'s slug changed.

    ``UserSettings.preferences["skills"]`` is keyed by slug, so a rename would
    silently orphan any explicit enable selection pointing at this skill (the
    old key no longer matches a skill; the new slug has no entry, so it falls
    back to shadowing defaults).

    Moves only entries whose ``selected_skill_id`` is this skill's id. Slug-wide
    disables (``selected_skill_id: None``) and entries selecting a *different*
    skill that shares the old slug are left alone, and an existing entry under
    ``new_slug`` is never clobbered (the user's explicit choice there wins).
    """
    from accounts.models import UserSettings

    if old_slug == new_slug:
        return

    skill_id = str(skill.id)
    settings_qs = UserSettings.objects.filter(
        preferences__skills__has_key=old_slug
    )
    for us in settings_qs:
        prefs = dict(us.preferences or {})
        skills_prefs = dict(prefs.get("skills") or {})
        entry = skills_prefs.get(old_slug)
        if not isinstance(entry, dict):
            continue
        if str(entry.get("selected_skill_id")) != skill_id:
            continue
        del skills_prefs[old_slug]
        if new_slug not in skills_prefs:
            skills_prefs[new_slug] = entry
        prefs["skills"] = skills_prefs
        us.preferences = prefs
        us.save(update_fields=["preferences"])


def create_org_skill(
    user, name: str, organization, slug: str | None = None,
    *, audience: str = AgentSkill.Audience.MAIN,
) -> AgentSkill:
    """Create an org-level skill. Caller must be an admin of ``organization``.

    Raises ``PermissionError`` if the user is not an admin of the org.
    """
    from accounts.models import Membership

    is_admin = Membership.objects.filter(
        user=user, org=organization, role=Membership.Role.ADMIN
    ).exists()
    if not is_admin:
        raise PermissionError("Only org admins can create org skills.")

    if audience not in AgentSkill.Audience.values or audience == AgentSkill.Audience.SHARED:
        audience = AgentSkill.Audience.MAIN
    if not slug:
        slug = slugify(name)[:64]
    if not slug:
        slug = "skill"

    return _save_with_free_slug(
        slug,
        lambda s: AgentSkill.objects.filter(
            slug=s, level="org", organization=organization
        ).exists(),
        lambda s: AgentSkill.objects.create(
            slug=s,
            name=name,
            instructions="",
            description="",
            audience=audience,
            level="org",
            organization=organization,
        ),
    )


def promote_skill_to_org(
    user, source_skill: AgentSkill, organization, *, copy_templates: bool = True,
    audience: str | None = None,
) -> AgentSkill:
    """Create an org-level copy of ``source_skill`` (or a no-op if already org).

    The caller must be an admin of ``organization``. Templates are copied
    by default. The new skill's ``parent`` points back to the source so the
    link is preserved.

    ``copy_templates`` defaults to True for callers that want a complete
    standalone copy. The detail-page form action passes ``False`` because
    it then re-creates the templates from the submitted form data; copying
    them here would clash with the ``unique_template_per_skill`` constraint.
    """
    from accounts.models import Membership

    is_admin = Membership.objects.filter(
        user=user, org=organization, role=Membership.Role.ADMIN
    ).exists()
    if not is_admin:
        raise PermissionError("Only org admins can promote skills.")

    if source_skill.level == "org" and source_skill.organization_id == organization.id:
        return source_skill

    # Org skills aren't authored as "shared" (seed-only); inherit the source's
    # audience, dropping shared to the tab's audience or main. tool_names are
    # copied verbatim (runtime resolution re-filters them by audience).
    new_audience = audience or source_skill.audience
    if new_audience == AgentSkill.Audience.SHARED:
        new_audience = AgentSkill.Audience.MAIN

    new_skill = _save_with_free_slug(
        source_skill.slug,
        lambda s: AgentSkill.objects.filter(
            slug=s, level="org", organization=organization
        ).exists(),
        lambda s: AgentSkill.objects.create(
            slug=s,
            name=source_skill.name,
            emoji=source_skill.emoji,
            description=source_skill.description,
            instructions=source_skill.instructions,
            tool_names=list(source_skill.tool_names or []),
            audience=new_audience,
            level="org",
            organization=organization,
            parent=source_skill,
        ),
    )

    if copy_templates:
        for tmpl in source_skill.templates.all():
            SkillTemplate.objects.create(
                skill=new_skill,
                name=tmpl.name,
                content=tmpl.content,
            )

    return new_skill


def move_skill_to_org(user, skill: AgentSkill, organization) -> AgentSkill:
    """Promote a personal skill to org level **in place** (no copy).

    Unlike :func:`promote_skill_to_org` (which duplicates), this changes the
    level of the same row: the personal skill *becomes* the org skill, so
    templates, ``parent``, and the id are all preserved and nothing is left
    behind at the user tier. Caller must be an admin of ``organization``.

    Raises ``PermissionError`` if the user is not an admin, or ``ValueError``
    if the skill is not a personal (user-level) skill.
    """
    from accounts.models import Membership

    is_admin = Membership.objects.filter(
        user=user, org=organization, role=Membership.Role.ADMIN
    ).exists()
    if not is_admin:
        raise PermissionError("Only org admins can promote skills.")
    if skill.level != "user":
        raise ValueError("Only personal skills can be promoted to the organization.")

    old_slug = skill.slug

    def persist(slug: str) -> AgentSkill:
        skill.slug = slug
        skill.level = "org"
        skill.organization = organization
        skill.created_by = None
        skill.save(
            update_fields=["slug", "level", "organization", "created_by", "updated_at"]
        )
        return skill

    _save_with_free_slug(
        old_slug,
        lambda s: AgentSkill.objects.filter(
            slug=s, level="org", organization=organization
        ).exclude(pk=skill.pk).exists(),
        persist,
    )
    if skill.slug != old_slug:
        migrate_skill_slug_prefs(skill, old_slug, skill.slug)
    return skill


def move_skill_to_personal(user, skill: AgentSkill) -> AgentSkill:
    """Demote an org skill to the acting admin's personal skills **in place**.

    Changes the level of the same row from org to user, assigning ownership to
    ``user``. This **removes the skill from the organization** — other members
    lose access. Caller must be an admin of the skill's organization.

    Raises ``PermissionError`` if the user is not an admin, or ``ValueError``
    if the skill is not an org-level skill.
    """
    from accounts.models import Membership

    if skill.level != "org":
        raise ValueError("Only organization skills can be demoted.")
    is_admin = Membership.objects.filter(
        user=user, org_id=skill.organization_id, role=Membership.Role.ADMIN
    ).exists()
    if not is_admin:
        raise PermissionError("Only org admins can demote skills.")

    old_slug = skill.slug

    def persist(slug: str) -> AgentSkill:
        skill.slug = slug
        skill.level = "user"
        skill.created_by = user
        skill.organization = None
        skill.save(
            update_fields=["slug", "level", "created_by", "organization", "updated_at"]
        )
        return skill

    _save_with_free_slug(
        old_slug,
        lambda s: AgentSkill.objects.filter(
            slug=s, level="user", created_by=user
        ).exclude(pk=skill.pk).exists(),
        persist,
    )
    if skill.slug != old_slug:
        migrate_skill_slug_prefs(skill, old_slug, skill.slug)
    return skill


# ----- Export / import --------------------------------------------------

# Bump when the export format changes incompatibly. Importers reject files
# carrying a higher version than they understand.
EXPORT_VERSION = 1

# Upper bound on skills per import file. The per-user slug dedup does one
# EXISTS query per collision, so an unbounded list of same-slug entries would
# be quadratic in queries — a 2 MB file of tiny entries could hang a dyno.
MAX_SKILLS_PER_IMPORT = 50


class SkillImportError(Exception):
    """Raised when an uploaded skill export file is malformed or unsupported.

    The view catches this and surfaces ``args[0]`` to the user via the Django
    messages framework, so the message must be human-readable.
    """


def _lines(text: str) -> list[str]:
    """Split a text field into a list of lines for human-readable JSON.

    Multi-line fields (instructions, template content) would otherwise become
    one giant ``\\n``-escaped string under ``json.dumps``. Splitting puts each
    line on its own row when pretty-printed. Round-trips exactly via
    ``"\\n".join`` — ``""`` -> ``[""]`` -> ``""``.
    """
    return (text or "").split("\n")


def _join_lines(value) -> str:
    """Inverse of :func:`_lines`, lenient about the input shape.

    Accepts a list of lines (the export form) or a plain string (so a
    hand-author can collapse a field back to a single string and still import).
    Anything else normalizes to an empty string.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(x) for x in value)
    return ""


def export_skill(skill: AgentSkill) -> dict:
    """Serialize a skill's portable fields.

    Environment-specific columns (id, level, organization, created_by, parent,
    is_active, timestamps) are intentionally omitted — an export describes the
    skill, not where it lived.
    """
    return {
        "slug": skill.slug,
        "name": skill.name,
        "emoji": skill.emoji,
        "audience": skill.audience,
        "description": _lines(skill.description),
        "instructions": _lines(skill.instructions),
        "tool_names": list(skill.tool_names or []),
        "templates": [
            {"name": t.name, "content": _lines(t.content)}
            for t in skill.templates.order_by("name")
        ],
    }


def serialize_skills(skills) -> dict:
    """Wrap one or more exported skills in the versioned envelope."""
    return {
        "wilfred_skill_export": EXPORT_VERSION,
        "skills": [export_skill(s) for s in skills],
    }


def dump_skills_json(skills) -> str:
    """Render skills as pretty-printed JSON suitable for download.

    ``ensure_ascii=False`` keeps emoji and Norwegian characters literal instead
    of ``\\uXXXX`` escapes, which matters for readability.
    """
    return json.dumps(serialize_skills(skills), indent=2, ensure_ascii=False)


def _normalize_skill_payload(entry: dict) -> dict:
    """Normalize one exported skill dict into create-ready fields.

    Total (never raises): missing fields get sensible defaults, text fields
    accept string-or-list, lengths are capped to match the edit form, and
    duplicate template names are de-duplicated keep-first to avoid tripping the
    ``unique_template_per_skill`` constraint. ``tool_names`` are kept verbatim
    (resolution is graceful at use time).
    """
    name = (str(entry.get("name") or "").strip() or "Imported skill")[:255]
    emoji = str(entry.get("emoji") or "")[:16]
    description = _join_lines(entry.get("description"))[:1024]
    instructions = _join_lines(entry.get("instructions"))[:MAX_INSTRUCTIONS_CHARS]

    audience = str(entry.get("audience") or "").strip().lower()
    if audience not in AgentSkill.Audience.values:
        audience = AgentSkill.Audience.MAIN

    raw_slug = str(entry.get("slug") or "").strip()
    slug = slugify(raw_slug)[:64] if raw_slug else ""

    raw_tools = entry.get("tool_names")
    tool_names = (
        filter_to_skill_tools(raw_tools, skill_audience=audience)
        if isinstance(raw_tools, list) else []
    )

    templates: list[dict] = []
    seen_names: set[str] = set()
    raw_templates = entry.get("templates")
    if isinstance(raw_templates, list):
        for t in raw_templates:
            if not isinstance(t, dict):
                continue
            tname = str(t.get("name") or "").strip()[:255]
            if not tname or tname in seen_names:
                continue
            seen_names.add(tname)
            templates.append({
                "name": tname,
                "content": _join_lines(t.get("content"))[:MAX_TEMPLATE_CHARS],
            })

    return {
        "slug": slug,
        "name": name,
        "emoji": emoji,
        "audience": audience,
        "description": description,
        "instructions": instructions,
        "tool_names": tool_names,
        "templates": templates,
    }


def parse_skill_export(raw) -> list[dict]:
    """Parse + validate an uploaded export into normalized skill payloads.

    Accepts either the ``{"wilfred_skill_export": N, "skills": [...]}`` envelope
    or a single bare skill dict (hand-authored). Raises :class:`SkillImportError`
    with a user-facing message on any file-level problem. Per-skill creation is
    left to :func:`import_skill` so the caller can isolate individual failures.
    """
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SkillImportError("File is not valid UTF-8 text.") from exc
    else:
        text = raw or ""

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise SkillImportError("File is not valid JSON.") from exc

    if not isinstance(data, dict):
        raise SkillImportError("Unexpected file structure — expected a skill object.")

    version = data.get("wilfred_skill_export")
    if version is not None:
        try:
            version_num = int(version)
        except (TypeError, ValueError) as exc:
            raise SkillImportError("Unrecognized export version.") from exc
        if version_num > EXPORT_VERSION:
            raise SkillImportError(
                "This skill was exported from a newer version of Wilfred."
            )

    if "skills" in data:
        entries = data.get("skills")
        if not isinstance(entries, list):
            raise SkillImportError("The 'skills' field must be a list.")
    else:
        # Tolerate a single bare skill dict.
        entries = [data]

    if len(entries) > MAX_SKILLS_PER_IMPORT:
        raise SkillImportError(
            f"That file contains {len(entries)} skills — the limit is "
            f"{MAX_SKILLS_PER_IMPORT} per import."
        )

    payloads = [
        _normalize_skill_payload(e) for e in entries if isinstance(e, dict)
    ]
    if not payloads:
        raise SkillImportError("The file contains no skills.")
    return payloads


def import_skill(user, payload: dict) -> AgentSkill:
    """Create a personal (user-level) skill from a normalized payload.

    Mirrors :func:`fork_skill` but sourced from a dict: ``parent=None`` and no
    lineage, so an imported skill is indistinguishable from one created from
    scratch. Reuses the per-user slug de-duplication convention.
    """
    base_slug = (
        payload.get("slug")
        or slugify(payload.get("name") or "")[:64]
        or "skill"
    )

    # Allow-list tool_names and cap instructions/template content at the
    # persistence boundary so the rules hold even for a caller that bypassed
    # _normalize_skill_payload.
    audience = payload.get("audience") or AgentSkill.Audience.MAIN
    if audience not in AgentSkill.Audience.values:
        audience = AgentSkill.Audience.MAIN

    skill = _save_with_free_slug(
        base_slug,
        lambda s: AgentSkill.objects.filter(
            slug=s, level="user", created_by=user
        ).exists(),
        lambda s: AgentSkill.objects.create(
            slug=s,
            name=payload["name"],
            emoji=payload.get("emoji", ""),
            description=payload.get("description", ""),
            instructions=(payload.get("instructions", "") or "")[:MAX_INSTRUCTIONS_CHARS],
            tool_names=filter_to_skill_tools(
                payload.get("tool_names") or [], skill_audience=audience
            ),
            level="user",
            created_by=user,
            parent=None,
        ),
    )

    for tmpl in payload.get("templates", []):
        SkillTemplate.objects.create(
            skill=skill,
            name=tmpl["name"],
            content=(tmpl.get("content", "") or "")[:MAX_TEMPLATE_CHARS],
        )

    return skill
