"""Host resource sampling (CPU and RAM) for pressure-aware routing."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class HostSample:
    cpu_percent: float = 0.0
    ram_available_mb: float = 65536.0
    ram_percent: float = 0.0


class HostMonitor:
    """Samples the host in the background so routing decisions never block on psutil."""

    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        psutil.cpu_percent(interval=None)  # the first call only primes the counter
        self.latest = self._read()
        self._task: asyncio.Task[None] | None = None

    @staticmethod
    def _read() -> HostSample:
        memory = psutil.virtual_memory()
        return HostSample(
            cpu_percent=psutil.cpu_percent(interval=None),
            ram_available_mb=round(memory.available / 2**20, 1),
            ram_percent=memory.percent,
        )

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval_s)
            self.latest = self._read()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None


class StaticMonitor:
    """A monitor whose reading is set by hand. Used by tests, the benchmark and demos."""

    def __init__(self, sample: HostSample | None = None) -> None:
        self.latest = sample or HostSample()

    def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass
