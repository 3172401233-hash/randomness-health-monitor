"""Raw random-byte health monitoring.

Feeds continuous 32-bit samples (with sampling timestamps) through a
bounded sliding window and evaluates simple statistical health signals:

- repeated blocks: duplicate fixed-size blocks inside the window
- window entropy: Shannon entropy of the 32-bit values, normalised
- bit bias: deviation of the global 1-bit fraction from 0.5
- constant runs: longest run of identical consecutive samples
- all-zero/all-one runs: longest runs of 0x00000000 / 0xFFFFFFFF

When any signal crosses its threshold the monitor reports ``DEGRADED``
and keeps a copy of the triggering window for forensics.  Recovery only
happens after a configurable number of consecutive healthy evaluation
windows (``confirm_windows``).

State can be serialised with :meth:`EntropyHealthMonitor.snapshot` and
restored later.  Restored monitors remember the last accepted timestamp,
so samples replayed after a restart are dropped instead of being counted
as new evidence.

The monitor is deterministic: it never reads the wall clock or any RNG;
all timing information comes from the caller-supplied timestamps.
"""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Sequence, Tuple

UINT32_MAX = 0xFFFFFFFF
SNAPSHOT_VERSION = 1


class HealthState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class MonitorConfig:
    """Tuning knobs for :class:`EntropyHealthMonitor`.

    window_size:            samples kept in the sliding window (memory bound)
    block_size:             samples per block for the repeated-block check
    step:                   evaluate every ``step`` newly accepted samples
    min_entropy_ratio:      min normalised Shannon entropy of window values
    max_repeat_block_ratio: max fraction of duplicate blocks in a window
    max_bit_bias:           max allowed |ones_fraction - 0.5|
    max_constant_run:       runs (identical / all-zero / all-one) >= this fail
    confirm_windows:        consecutive healthy evaluations needed to recover
    history_limit:          bounded number of evaluation reports kept
    """

    window_size: int = 256
    block_size: int = 4
    step: int = 32
    min_entropy_ratio: float = 0.85
    max_repeat_block_ratio: float = 0.20
    max_bit_bias: float = 0.05
    max_constant_run: int = 8
    confirm_windows: int = 3
    history_limit: int = 128

    def validate(self) -> None:
        if self.window_size < 8:
            raise ValueError("window_size must be >= 8")
        if self.block_size < 1 or self.window_size % self.block_size != 0:
            raise ValueError("block_size must divide window_size")
        if self.step < 1:
            raise ValueError("step must be >= 1")
        if not 0.0 < self.min_entropy_ratio <= 1.0:
            raise ValueError("min_entropy_ratio must be in (0, 1]")
        if not 0.0 < self.max_repeat_block_ratio <= 1.0:
            raise ValueError("max_repeat_block_ratio must be in (0, 1]")
        if not 0.0 < self.max_bit_bias < 0.5:
            raise ValueError("max_bit_bias must be in (0, 0.5)")
        if self.max_constant_run < 2:
            raise ValueError("max_constant_run must be >= 2")
        if self.confirm_windows < 1:
            raise ValueError("confirm_windows must be >= 1")
        if self.history_limit < 1:
            raise ValueError("history_limit must be >= 1")


@dataclass
class WindowReport:
    """Result of one evaluation over the sliding window."""

    eval_index: int
    window_end_count: int  # total accepted samples at evaluation time
    entropy_ratio: float
    repeat_block_ratio: float
    bit_bias: float
    longest_constant_run: int
    longest_zero_run: int
    longest_one_run: int
    failures: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


@dataclass
class TriggerWindow:
    """Forensic copy of the window that caused the last DEGRADED transition."""

    samples: List[int]
    timestamps: List[float]
    report: WindowReport


def _entropy_ratio(samples: Sequence[int]) -> float:
    """Shannon entropy of the values, normalised to [0, 1].

    The maximum is log2(min(n, 2**32)) so a window of distinct values
    scores 1.0 and a constant window scores 0.0.
    """
    n = len(samples)
    if n == 0:
        return 0.0
    counts = Counter(samples)
    entropy = 0.0
    for c in counts.values():
        p = c / n
        entropy -= p * math.log2(p)
    h_max = math.log2(min(n, 2 ** 32))
    return entropy / h_max if h_max > 0 else 0.0


def _repeat_block_ratio(samples: Sequence[int], block_size: int) -> float:
    """Fraction of non-overlapping blocks already seen earlier in the window."""
    blocks = [
        tuple(samples[i : i + block_size])
        for i in range(0, len(samples) - block_size + 1, block_size)
    ]
    if not blocks:
        return 0.0
    seen = set()
    repeats = 0
    for block in blocks:
        if block in seen:
            repeats += 1
        else:
            seen.add(block)
    return repeats / len(blocks)


def _bit_bias(samples: Sequence[int]) -> float:
    """|ones_fraction - 0.5| over all bits of the window."""
    if not samples:
        return 0.0
    ones = 0
    for v in samples:
        ones += bin(v).count("1")
    return abs(ones / (32 * len(samples)) - 0.5)


def _runs(samples: Sequence[int]) -> Tuple[int, int, int]:
    """Longest identical-value run, all-zero run and all-one run."""
    longest_const = cur_const = 0
    longest_zero = cur_zero = 0
    longest_one = cur_one = 0
    prev = None
    for v in samples:
        cur_const = cur_const + 1 if v == prev else 1
        prev = v
        cur_zero = cur_zero + 1 if v == 0 else 0
        cur_one = cur_one + 1 if v == UINT32_MAX else 0
        longest_const = max(longest_const, cur_const)
        longest_zero = max(longest_zero, cur_zero)
        longest_one = max(longest_one, cur_one)
    return longest_const, longest_zero, longest_one


class EntropyHealthMonitor:
    """Sliding-window health monitor for raw 32-bit random samples."""

    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()
        self.config.validate()
        self._samples: Deque[int] = deque(maxlen=self.config.window_size)
        self._timestamps: Deque[float] = deque(maxlen=self.config.window_size)
        self._state = HealthState.HEALTHY
        self._healthy_streak = 0
        self._since_eval = 0
        self._eval_count = 0
        self._total_accepted = 0
        self._total_dropped = 0
        self._last_timestamp: Optional[float] = None
        self._trigger: Optional[TriggerWindow] = None
        self._history: Deque[WindowReport] = deque(maxlen=self.config.history_limit)

    @property
    def state(self) -> HealthState:
        return self._state

    @property
    def trigger_window(self) -> Optional[TriggerWindow]:
        """The window that last triggered DEGRADED (kept after recovery)."""
        return self._trigger

    @property
    def history(self) -> List[WindowReport]:
        return list(self._history)

    @property
    def total_accepted(self) -> int:
        return self._total_accepted

    @property
    def total_dropped(self) -> int:
        return self._total_dropped

    @property
    def buffered_samples(self) -> int:
        return len(self._samples)

    def add_sample(self, value: int, timestamp: float) -> HealthState:
        """Ingest one 32-bit sample; returns the current health state.

        Samples with a non-increasing timestamp (replays, out-of-order
        deliveries after a restart) are dropped and never counted as new
        evidence.
        """
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("value must be an int")
        if not 0 <= value <= UINT32_MAX:
            raise ValueError("value must fit in 32 bits")
        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
            raise TypeError("timestamp must be a number")
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")

        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            self._total_dropped += 1
            return self._state

        self._last_timestamp = float(timestamp)
        self._samples.append(value)
        self._timestamps.append(float(timestamp))
        self._total_accepted += 1
        self._since_eval += 1

        if (
            len(self._samples) == self.config.window_size
            and self._since_eval >= self.config.step
        ):
            self._since_eval = 0
            self._evaluate()
        return self._state

    def add_batch(
        self, values: Sequence[int], timestamps: Sequence[float]
    ) -> HealthState:
        if len(values) != len(timestamps):
            raise ValueError("values and timestamps must have equal length")
        state = self._state
        for value, ts in zip(values, timestamps):
            state = self.add_sample(value, ts)
        return state

    def _evaluate(self) -> WindowReport:
        cfg = self.config
        window = list(self._samples)
        entropy = _entropy_ratio(window)
        repeat_ratio = _repeat_block_ratio(window, cfg.block_size)
        bias = _bit_bias(window)
        const_run, zero_run, one_run = _runs(window)

        failures: List[str] = []
        if entropy < cfg.min_entropy_ratio:
            failures.append("low_entropy")
        if repeat_ratio > cfg.max_repeat_block_ratio:
            failures.append("repeated_blocks")
        if bias > cfg.max_bit_bias:
            failures.append("bit_bias")
        if const_run >= cfg.max_constant_run:
            failures.append("constant_run")
        if zero_run >= cfg.max_constant_run:
            failures.append("all_zero_run")
        if one_run >= cfg.max_constant_run:
            failures.append("all_one_run")

        self._eval_count += 1
        report = WindowReport(
            eval_index=self._eval_count,
            window_end_count=self._total_accepted,
            entropy_ratio=entropy,
            repeat_block_ratio=repeat_ratio,
            bit_bias=bias,
            longest_constant_run=const_run,
            longest_zero_run=zero_run,
            longest_one_run=one_run,
            failures=failures,
        )
        self._history.append(report)

        if failures:
            self._state = HealthState.DEGRADED
            self._healthy_streak = 0
            self._trigger = TriggerWindow(
                samples=window,
                timestamps=list(self._timestamps),
                report=report,
            )
        elif self._state is HealthState.DEGRADED:
            self._healthy_streak += 1
            if self._healthy_streak >= cfg.confirm_windows:
                self._state = HealthState.HEALTHY
                self._healthy_streak = 0
        return report

    def snapshot(self) -> dict:
        """Serialisable state; ``json.dumps``-safe."""
        return {
            "version": SNAPSHOT_VERSION,
            "config": asdict(self.config),
            "state": self._state.value,
            "samples": list(self._samples),
            "timestamps": list(self._timestamps),
            "healthy_streak": self._healthy_streak,
            "since_eval": self._since_eval,
            "eval_count": self._eval_count,
            "total_accepted": self._total_accepted,
            "total_dropped": self._total_dropped,
            "last_timestamp": self._last_timestamp,
            "trigger": None
            if self._trigger is None
            else {
                "samples": self._trigger.samples,
                "timestamps": self._trigger.timestamps,
                "report": asdict(self._trigger.report),
            },
            "history": [asdict(r) for r in self._history],
        }

    def snapshot_json(self) -> str:
        return json.dumps(self.snapshot(), sort_keys=True)

    @classmethod
    def restore(cls, snap: dict) -> "EntropyHealthMonitor":
        """Rebuild a monitor from :meth:`snapshot` output.

        The restored monitor keeps the previous ``last_timestamp``, so
        old samples replayed after a restart are dropped instead of being
        double-counted as fresh evidence.
        """
        if snap.get("version") != SNAPSHOT_VERSION:
            raise ValueError(f"unsupported snapshot version: {snap.get('version')!r}")
        monitor = cls(MonitorConfig(**snap["config"]))
        monitor._state = HealthState(snap["state"])
        monitor._samples = deque(snap["samples"], maxlen=monitor.config.window_size)
        monitor._timestamps = deque(
            snap["timestamps"], maxlen=monitor.config.window_size
        )
        monitor._healthy_streak = snap["healthy_streak"]
        monitor._since_eval = snap["since_eval"]
        monitor._eval_count = snap["eval_count"]
        monitor._total_accepted = snap["total_accepted"]
        monitor._total_dropped = snap["total_dropped"]
        monitor._last_timestamp = snap["last_timestamp"]
        trigger = snap["trigger"]
        if trigger is not None:
            monitor._trigger = TriggerWindow(
                samples=list(trigger["samples"]),
                timestamps=list(trigger["timestamps"]),
                report=WindowReport(**trigger["report"]),
            )
        monitor._history = deque(
            (WindowReport(**r) for r in snap["history"]),
            maxlen=monitor.config.history_limit,
        )
        return monitor

    @classmethod
    def restore_json(cls, payload: str) -> "EntropyHealthMonitor":
        return cls.restore(json.loads(payload))
