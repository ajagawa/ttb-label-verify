"""How many CPUs this process may actually use.

Why this exists: on a container platform the CPU limit is a *quota*, not a
count. A container on Render's Standard plan is allowed one CPU's worth of
time, but `os.cpu_count()` — and ONNX Runtime's default thread count, which is
derived the same way — reports every core on the host machine. The OCR engine
then starts one thread per host core, all competing for a one-CPU budget; the
scheduler lets them burn through the quota in a fraction of each period and
then freezes the whole process for the rest of it.

Measured under a one-CPU quota (cgroup cfs_quota) on the fixture label: one
thread 2.8 s per check, sixteen threads 3.6-3.7 s, with the process throttled in
most scheduling periods. `taskset`, which the first 1-CPU simulation used, hides
this, because it shrinks the visible core count as well as the budget — one
reason the deployed instance was slower than that simulation predicted.

The quota is read from the cgroup filesystem (v2 first, then v1). CPU affinity
is also honoured. The smaller of the two wins, floored at one.
"""

from __future__ import annotations

import contextlib
import math
import os
from pathlib import Path

CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
CGROUP_V1_DIRS = (Path("/sys/fs/cgroup/cpu"), Path("/sys/fs/cgroup/cpu,cpuacct"))


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def cgroup_cpu_quota(
    v2_path: Path = CGROUP_V2_CPU_MAX, v1_dirs: tuple[Path, ...] = CGROUP_V1_DIRS
) -> float | None:
    """CPUs allowed by the cgroup quota, or None when there is no quota.

    Args:
        v2_path: cgroup v2 `cpu.max` ("<quota> <period>" or "max <period>").
        v1_dirs: candidate cgroup v1 cpu controller directories.
    """
    raw = _read(v2_path)
    if raw:
        parts = raw.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota, period = int(parts[0]), int(parts[1])
                if quota > 0 and period > 0:
                    return quota / period
            except ValueError:
                pass
        return None

    for directory in v1_dirs:
        quota_raw = _read(directory / "cpu.cfs_quota_us")
        period_raw = _read(directory / "cpu.cfs_period_us")
        if quota_raw is None or period_raw is None:
            continue
        try:
            quota, period = int(quota_raw), int(period_raw)
        except ValueError:
            continue
        if quota > 0 and period > 0:
            return quota / period
        return None  # -1: unlimited
    return None


def visible_cpus() -> int:
    """Cores this process is scheduled on (affinity), else the host count."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def effective_cpus() -> int:
    """Whole CPUs usable by this process: min(quota, affinity), at least one."""
    visible = visible_cpus()
    quota = cgroup_cpu_quota()
    if quota is None:
        return max(1, visible)
    return max(1, min(visible, math.floor(quota)))


def ocr_threads() -> int:
    """ONNX intra-op threads for the interactive OCR engine.

    `LABEL_VERIFY_OCR_THREADS` overrides; otherwise one per usable CPU.
    """
    raw = os.environ.get("LABEL_VERIFY_OCR_THREADS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return effective_cpus()


def pin_to_quota(task_dir: Path = Path("/proc/self/task")) -> list[int] | None:
    """Confine the whole process to as many cores as its CPU quota allows.

    Why: batch work runs at a lower scheduling priority (nice) so that a
    single-label check always wins the CPU. But priority only arbitrates
    between threads queued on the *same* core. Under a quota the process may
    run on any of the host's cores — sixteen on Render — so the batch thread
    and the interactive thread run side by side on two different cores and
    split the one-CPU budget evenly; the priority never comes into play.
    Measured under a one-CPU quota: single-label p50 9.3 s during a batch
    against 4.4 s idle, i.e. an even split. On the live host: 6-6.5 s during a
    batch against 2.7 s idle.

    Pinning to `floor(quota)` cores makes the threads compete on the same run
    queue again, where the priority applies. The cost: the process cannot
    migrate off a core a neighbouring tenant is also using. Under a quota it
    could never use more than that many cores' worth of time anyway.

    `LABEL_VERIFY_PIN_CPUS`: unset or "auto" pins only when a quota is smaller
    than the visible cores; "0" never pins; a number N pins to N cores.

    Applied to every existing thread of the process (affinity is per thread on
    Linux, and only threads created afterwards inherit it), so call it as early
    as possible. Returns the cores pinned to, or None when nothing changed.
    """
    raw = os.environ.get("LABEL_VERIFY_PIN_CPUS", "auto").strip().lower()
    if raw == "0":
        return None
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None

    if raw in ("", "auto"):
        quota = cgroup_cpu_quota()
        if quota is None:
            return None
        target = max(1, math.floor(quota))
    else:
        try:
            target = max(1, int(raw))
        except ValueError:
            return None
    if target >= len(allowed):
        return None

    cores = set(allowed[:target])
    try:
        tids = [int(t.name) for t in task_dir.iterdir()]
    except OSError:
        tids = [0]
    for tid in tids:
        # OSError: a thread that exited in the meantime.
        with contextlib.suppress(OSError):
            os.sched_setaffinity(tid, cores)
    return sorted(cores)
