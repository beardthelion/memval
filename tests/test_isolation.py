"""Real-backend isolation checks. Skipped when the private checkouts or bun
are absent, so the suite stays green for outside contributors."""
import shutil
from pathlib import Path

import anyio
import pytest

from memval.isolation import check_isolation, surviving_artifacts
from memval.supervisor import RunRoot

MEMLAWB = Path.home() / "projects" / "memlawb"
SIGNET = Path.home() / "projects" / "signet"

need_bun = pytest.mark.skipif(shutil.which("bun") is None, reason="bun not installed")
need_memlawb = pytest.mark.skipif(
    not (MEMLAWB / "bin" / "memlawb.ts").exists(), reason="memlawb checkout absent"
)
need_signet = pytest.mark.skipif(
    not (SIGNET / "bin" / "signet.ts").exists(), reason="signet checkout absent"
)


@need_bun
@need_memlawb
def test_memlawb_isolation():
    result = anyio.run(check_isolation, "memlawb", MEMLAWB)
    assert result.ok, result.detail


@need_bun
@need_signet
def test_signet_isolation():
    result = anyio.run(check_isolation, "signet", SIGNET)
    assert result.ok, result.detail


def test_unknown_backend():
    result = anyio.run(check_isolation, "nope", Path("/tmp"))
    assert not result.ok


def test_surviving_artifacts_catches_leak():
    """The artifact check must not be vacuous: a leaked custody dir after
    cleanup is detected."""
    root = RunRoot()
    (root.path / "custody.json").write_text("{}")
    root.cleanup()
    assert surviving_artifacts(root) == []
    # Simulate a leak: state recreated under the run root after teardown.
    (root.path / "leaked-custody").mkdir(parents=True)
    assert root.path / "leaked-custody" in surviving_artifacts(root)
    root.cleanup()
