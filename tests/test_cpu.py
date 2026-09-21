"""CPU budget detection (extraction/cpu.py).

The point of the module is that a container's CPU *quota* — not the host's
core count — decides how many OCR threads to start. These tests pin the
parsing of both cgroup formats and the min(quota, affinity) rule.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from extraction import cpu


def _v1(tmp_path: Path, quota: str, period: str = "100000") -> Path:
    d = tmp_path / "cpu"
    d.mkdir()
    (d / "cpu.cfs_quota_us").write_text(quota + "\n")
    (d / "cpu.cfs_period_us").write_text(period + "\n")
    return d


class TestCgroupQuota:
    def test_v2_quota(self, tmp_path: Path) -> None:
        f = tmp_path / "cpu.max"
        f.write_text("100000 100000\n")
        assert cpu.cgroup_cpu_quota(f, ()) == 1.0

    def test_v2_fractional_quota(self, tmp_path: Path) -> None:
        f = tmp_path / "cpu.max"
        f.write_text("50000 100000\n")
        assert cpu.cgroup_cpu_quota(f, ()) == 0.5

    def test_v2_unlimited(self, tmp_path: Path) -> None:
        f = tmp_path / "cpu.max"
        f.write_text("max 100000\n")
        assert cpu.cgroup_cpu_quota(f, ()) is None

    def test_v1_quota(self, tmp_path: Path) -> None:
        d = _v1(tmp_path, "200000")
        assert cpu.cgroup_cpu_quota(tmp_path / "missing", (d,)) == 2.0

    def test_v1_unlimited(self, tmp_path: Path) -> None:
        d = _v1(tmp_path, "-1")
        assert cpu.cgroup_cpu_quota(tmp_path / "missing", (d,)) is None

    def test_no_cgroup_files(self, tmp_path: Path) -> None:
        assert cpu.cgroup_cpu_quota(tmp_path / "missing", (tmp_path / "nope",)) is None

    def test_garbage_is_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / "cpu.max"
        f.write_text("banana\n")
        assert cpu.cgroup_cpu_quota(f, ()) is None


class TestEffectiveCpus:
    def test_quota_below_visible_cores_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Render case: 1 CPU of quota on a host with many cores."""
        monkeypatch.setattr(cpu, "visible_cpus", lambda: 32)
        monkeypatch.setattr(cpu, "cgroup_cpu_quota", lambda: 1.0)
        assert cpu.effective_cpus() == 1

    def test_fractional_quota_floors_at_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cpu, "visible_cpus", lambda: 8)
        monkeypatch.setattr(cpu, "cgroup_cpu_quota", lambda: 0.5)
        assert cpu.effective_cpus() == 1

    def test_affinity_below_quota_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cpu, "visible_cpus", lambda: 2)
        monkeypatch.setattr(cpu, "cgroup_cpu_quota", lambda: 4.0)
        assert cpu.effective_cpus() == 2

    def test_no_quota_uses_visible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cpu, "visible_cpus", lambda: 6)
        monkeypatch.setattr(cpu, "cgroup_cpu_quota", lambda: None)
        assert cpu.effective_cpus() == 6


class TestOcrThreads:
    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LABEL_VERIFY_OCR_THREADS", "3")
        assert cpu.ocr_threads() == 3

    def test_bad_env_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LABEL_VERIFY_OCR_THREADS", "lots")
        monkeypatch.setattr(cpu, "effective_cpus", lambda: 1)
        assert cpu.ocr_threads() == 1

    def test_default_is_effective_cpus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LABEL_VERIFY_OCR_THREADS", raising=False)
        monkeypatch.setattr(cpu, "effective_cpus", lambda: 1)
        assert cpu.ocr_threads() == 1
