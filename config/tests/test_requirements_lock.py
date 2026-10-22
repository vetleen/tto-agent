"""Guard the hash-locked dependency file (RUNBOOK.md, "Dependencies").

``requirements.txt`` is generated from ``requirements.in`` by ``uv pip compile
--generate-hashes``. Every entry must be an exact pin with hashes — one
unhashed line makes pip drop out of hash-checking mode for nothing, but a
missing hash on a needed package fails the Heroku build — and every package
declared in ``requirements.in`` must be in the lock, which catches editing the
``.in`` file without recompiling.
"""

import re
from pathlib import Path

from django.test import SimpleTestCase
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]


def _logical_lines(path: Path) -> list[str]:
    """Requirement lines with backslash continuations joined and comments dropped."""
    lines, current = [], ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        if line.endswith("\\"):
            current += line[:-1] + " "
            continue
        lines.append((current + line).strip())
        current = ""
    return lines


class RequirementsLockTests(SimpleTestCase):
    def setUp(self):
        self.lock_lines = _logical_lines(ROOT / "requirements.txt")
        self.in_lines = _logical_lines(ROOT / "requirements.in")

    def test_every_lock_entry_is_pinned_with_hashes(self):
        self.assertGreater(len(self.lock_lines), 50)
        for line in self.lock_lines:
            spec = line.split("--hash", 1)[0].strip()
            req = Requirement(spec)
            with self.subTest(package=req.name):
                self.assertRegex(str(req.specifier), r"^==[^,*]+$", "lock entries must be exact == pins")
                self.assertIsNotNone(
                    re.search(r"--hash=sha256:[0-9a-f]{64}", line),
                    "lock entries must carry hashes (recompile with --generate-hashes)",
                )

    def test_every_declared_package_is_locked(self):
        locked = {
            canonicalize_name(Requirement(line.split("--hash", 1)[0].strip()).name)
            for line in self.lock_lines
        }
        for line in self.in_lines:
            name = canonicalize_name(Requirement(line).name)
            with self.subTest(package=name):
                self.assertIn(
                    name, locked,
                    "requirements.in changed without recompiling requirements.txt "
                    "(see RUNBOOK.md, Dependencies)",
                )
