"""Small cross-process build lock based on atomic directory creation."""

from __future__ import annotations

from contextlib import suppress
import json
import os
from pathlib import Path
import shutil
import socket
from threading import Event, Thread
import time
from uuid import uuid4


_OWNER_FILENAME = "owner.json"
_HEARTBEAT_FILENAME = "heartbeat"
_GUARD_SUFFIX = ".guard"


def _process_start_token(pid: int) -> str | None:
    """Return a PID incarnation token where the platform exposes one."""

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if not handle:
                return None
            try:
                creation = wintypes.FILETIME()
                exit_time = wintypes.FILETIME()
                kernel_time = wintypes.FILETIME()
                user_time = wintypes.FILETIME()
                ok = ctypes.windll.kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_time),
                    ctypes.byref(kernel_time),
                    ctypes.byref(user_time),
                )
                if not ok:
                    return None
                value = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
                return str(value)
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except (AttributeError, OSError, TypeError, ValueError):
            return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(
            encoding="ascii"
        ).rsplit(")", 1)[1].split()
    except (OSError, UnicodeError, IndexError):
        return None
    # The post-comm field list starts at field 3; starttime is field 22.
    return fields[19] if len(fields) > 19 else None


def _owner_is_alive(owner: dict[str, object]) -> bool:
    try:
        pid = owner["pid"]
        host = owner["host"]
        start_token = owner["start_token"]
        if type(pid) is not int or type(host) is not str:
            return False
        if host != socket.gethostname() or pid <= 0:
            return False
        os.kill(pid, 0)
    except KeyError:
        return False
    except PermissionError:
        # An access-denied liveness probe is not proof of death.
        return True
    except OSError:
        # Unsupported/temporarily unavailable probes are conservatively live.
        return True
    current_token = _process_start_token(pid)
    if current_token is None:
        # Windows has no /proc incarnation token; heartbeat plus PID liveness
        # is the safe signal available on that platform.
        return True
    return current_token == start_token


def _owner_is_confirmed_dead(owner: dict[str, object]) -> bool:
    """Return true only when a same-host owner is provably another process."""

    try:
        pid = owner["pid"]
        host = owner["host"]
        start_token = owner["start_token"]
    except (KeyError, TypeError):
        return False
    if (
        type(pid) is not int
        or type(host) is not str
        or host != socket.gethostname()
        or pid <= 0
        or type(start_token) is not str
    ):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        # Permission and platform errors do not prove that the owner is dead.
        return False
    current_token = _process_start_token(pid)
    return current_token is not None and current_token != start_token


class _OperationGuard:
    """Crash-recoverable short guard for lock-path transitions."""

    def __init__(self, lock_dir: Path, timeout_s: float) -> None:
        self._guard_dir = lock_dir.with_name(lock_dir.name + _GUARD_SUFFIX)
        self._timeout_s = timeout_s
        self._owner = uuid4().hex

    def _try_reclaim(self) -> bool:
        try:
            age_s = time.time() - self._guard_dir.stat().st_mtime
        except OSError:
            return True
        if age_s <= self._timeout_s:
            return False
        try:
            owner = json.loads(
                (self._guard_dir / _OWNER_FILENAME).read_text(
                    encoding="ascii"
                )
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            owner = {}
        if isinstance(owner, dict) and _owner_is_alive(owner):
            return False
        reclaimed = self._guard_dir.with_name(
            f".{self._guard_dir.name}.stale-{uuid4().hex}"
        )
        try:
            os.replace(self._guard_dir, reclaimed)
        except FileNotFoundError:
            return True
        except OSError:
            return False
        shutil.rmtree(reclaimed, ignore_errors=True)
        return True

    def __enter__(self) -> _OperationGuard:
        self._guard_dir.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self._timeout_s
        while True:
            try:
                self._guard_dir.mkdir()
                try:
                    (self._guard_dir / _OWNER_FILENAME).write_text(
                        json.dumps(
                            {
                                "pid": os.getpid(),
                                "host": socket.gethostname(),
                                "start_token": _process_start_token(os.getpid()),
                                "token": self._owner,
                            },
                            sort_keys=True,
                        ),
                        encoding="ascii",
                    )
                except BaseException:
                    shutil.rmtree(self._guard_dir, ignore_errors=True)
                    raise
                return self
            except FileExistsError:
                if self._try_reclaim():
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for lock guard: {self._guard_dir}"
                    )
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

    def __exit__(self, *_args: object) -> None:
        owner_path = self._guard_dir / _OWNER_FILENAME
        try:
            owner = json.loads(owner_path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return
        if not isinstance(owner, dict) or owner.get("token") != self._owner:
            return
        shutil.rmtree(self._guard_dir, ignore_errors=True)


class ExclusiveBuildLock:
    """Bounded, heartbeat-backed lock usable on POSIX and Windows."""

    def __init__(
        self,
        target: Path,
        *,
        timeout_s: float | None = None,
        stale_after_s: float = 600.0,
    ) -> None:
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("lock timeout must be positive")
        if stale_after_s <= 0:
            raise ValueError("stale timeout must be positive")
        self._lock_dir = Path(target).with_name(
            f".{Path(target).name}.lock"
        )
        self._timeout_s = timeout_s
        self._stale_after_s = stale_after_s
        self._owner = uuid4().hex
        self._heartbeat_stop = Event()
        self._heartbeat_thread: Thread | None = None

    def _owner_payload(self) -> dict[str, object]:
        return {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "start_token": _process_start_token(os.getpid()),
            "token": self._owner,
        }

    def acquire(self) -> None:
        self._lock_dir.parent.mkdir(parents=True, exist_ok=True)
        deadline = (
            None
            if self._timeout_s is None
            else time.monotonic() + self._timeout_s
        )
        while True:
            guard_timeout = self._timeout_s or self._stale_after_s
            with _OperationGuard(self._lock_dir, guard_timeout):
                if self._lock_dir.is_symlink():
                    raise RuntimeError(
                        f"build lock is a symlink: {self._lock_dir}"
                    )
                try:
                    self._lock_dir.mkdir()
                    try:
                        (self._lock_dir / _OWNER_FILENAME).write_text(
                            json.dumps(self._owner_payload(), sort_keys=True),
                            encoding="ascii",
                        )
                        (self._lock_dir / _HEARTBEAT_FILENAME).touch()
                    except BaseException:
                        shutil.rmtree(self._lock_dir, ignore_errors=True)
                        raise
                    self._heartbeat_stop.clear()
                    self._heartbeat_thread = Thread(
                        target=self._heartbeat,
                        name=f"build-lock-{self._owner[:8]}",
                        daemon=True,
                    )
                    self._heartbeat_thread.start()
                    return
                except FileExistsError:
                    progressed = not self._try_reclaim_stale()
            if progressed:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for build lock: {self._lock_dir}"
                    )
                time.sleep(0.05)

    def _heartbeat(self) -> None:
        interval = min(1.0, max(0.01, self._stale_after_s / 3.0))
        heartbeat_path = self._lock_dir / _HEARTBEAT_FILENAME
        while not self._heartbeat_stop.wait(interval):
            try:
                heartbeat_path.touch(exist_ok=False)
            except FileExistsError:
                with suppress(OSError):
                    os.utime(heartbeat_path, None)
            except OSError:
                return

    def _try_reclaim_stale(self) -> bool:
        try:
            owner = json.loads(
                (self._lock_dir / _OWNER_FILENAME).read_text(encoding="ascii")
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            owner = {}

        # A crashed local process can leave a fresh heartbeat behind.  Once
        # the operation guard is held, a matching PID incarnation check makes
        # reclaiming that lock safe without waiting for stale_after_s.
        confirmed_dead = (
            isinstance(owner, dict) and _owner_is_confirmed_dead(owner)
        )
        heartbeat_path = self._lock_dir / _HEARTBEAT_FILENAME
        if not confirmed_dead:
            try:
                heartbeat_age_s = time.time() - heartbeat_path.stat().st_mtime
            except OSError:
                try:
                    heartbeat_age_s = time.time() - self._lock_dir.stat().st_mtime
                except OSError:
                    return True
            if heartbeat_age_s <= self._stale_after_s:
                return False
            if isinstance(owner, dict) and _owner_is_alive(owner):
                return False
        reclaimed = self._lock_dir.with_name(
            f".{self._lock_dir.name}.stale-{uuid4().hex}"
        )
        try:
            os.replace(self._lock_dir, reclaimed)
        except FileNotFoundError:
            return True
        except OSError:
            return False
        shutil.rmtree(reclaimed, ignore_errors=True)
        return True

    def release(self) -> None:
        if self._heartbeat_thread is not None:
            self._heartbeat_stop.set()
            self._heartbeat_thread.join(timeout=1.0)
            self._heartbeat_thread = None
        guard_timeout = self._timeout_s or self._stale_after_s
        with _OperationGuard(self._lock_dir, guard_timeout):
            owner_path = self._lock_dir / _OWNER_FILENAME
            try:
                owner = json.loads(owner_path.read_text(encoding="ascii"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                return
            if not isinstance(owner, dict) or owner.get("token") != self._owner:
                return
            shutil.rmtree(self._lock_dir, ignore_errors=True)

    def __enter__(self) -> ExclusiveBuildLock:
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()
