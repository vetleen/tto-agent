"""Every ``{% static %}`` reference in a template must resolve to a real file.

``ManifestStaticStorage`` sets ``manifest_strict = False`` as a first-rollout safety valve
(``core/storage_backends.py``), so in production a reference to a file that isn't collected
does not raise — it silently falls back to the unhashed path. WhiteNoise then serves that
one asset with ``max-age=60`` instead of ``immutable``, so every navigation revalidates it
and the unstyled-content flash that hashing was introduced to fix quietly comes back for
that file. Nothing logs it and no test caught it.

This asserts the property that actually matters — every referenced file exists — which is
the failure mode behind a manifest miss. It runs in milliseconds against the finders, so
there is no need to run a full ``collectstatic``, and it leaves the safety valve in place
so a genuine miss still degrades rather than 500ing a live page.
"""

import re
from pathlib import Path

from django.conf import settings
from django.contrib.staticfiles import finders
from django.test import SimpleTestCase

# Matches {% static 'path/to/file.ext' %} with either quote style. References built from a
# variable ({% static var %}) are skipped: there is no literal path to check.
_STATIC_TAG = re.compile(r"""\{%\s*static\s+['"]([^'"]+)['"]\s*%\}""")


def _template_dirs():
    dirs = []
    for engine in settings.TEMPLATES:
        dirs.extend(Path(d) for d in engine.get("DIRS", []))
    return dirs


class StaticReferencesResolveTests(SimpleTestCase):
    def test_every_static_reference_in_templates_exists(self):
        missing = []
        checked = 0
        for root in _template_dirs():
            for template in root.rglob("*.html"):
                text = template.read_text(encoding="utf-8")
                for ref in _STATIC_TAG.findall(text):
                    checked += 1
                    if finders.find(ref) is None:
                        missing.append(
                            "%s -> %s" % (template.relative_to(root).as_posix(), ref)
                        )

        # A guard on the guard: if the regex or the template dirs ever stop matching,
        # the loop above would pass vacuously.
        self.assertGreater(checked, 0, "found no {% static %} references to check")
        self.assertEqual(
            missing,
            [],
            "These {%% static %%} references don't resolve to a collectable file. In "
            "production each one silently loses its content hash and drops from "
            "immutable caching to max-age=60:\n  %s" % "\n  ".join(missing),
        )
