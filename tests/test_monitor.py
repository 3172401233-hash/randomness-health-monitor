import json
import random
import unittest

from entropy_health import (
    EntropyHealthMonitor,
    HealthState,
    MonitorConfig,
)

UINT32 = 2 ** 32


def fast_config(**overrides):
    params = dict(
        window_size=64,
        block_size=4,
        step=16,
        min_entropy_ratio=0.85,
        max_repeat_block_ratio=0.20,
        max_bit_bias=0.05,
        max_constant_run=8,
        confirm_windows=3,
        history_limit=16,
    )
    params.update(overrides)
    return MonitorConfig(**params)


def random_values(n, rng):
    return [rng.randrange(UINT32) for _ in range(n)]


def timestamps(n, start=0):
    return [float(start + i) for i in range(n)]


class TestPseudoRandomStream(unittest.TestCase):
    def test_healthy_random_stream(self):
        rng = random.Random(1234)
        mon = EntropyHealthMonitor(fast_config())
        values = random_values(4000, rng)
        states = [mon.add_sample(v, float(i)) for i, v in enumerate(values)]
        self.assertTrue(mon.history)  # evaluations happened
        self.assertEqual(set(states), {HealthState.HEALTHY})
        self.assertIsNone(mon.trigger_window)
        for report in mon.history:
            self.assertTrue(report.ok, report)
            self.assertGreaterEqual(report.entropy_ratio, 0.85)

    def test_no_evaluation_before_window_full(self):
        rng = random.Random(7)
        mon = EntropyHealthMonitor(fast_config())
        for i, v in enumerate(random_values(63, rng)):
            mon.add_sample(v, float(i))
        self.assertEqual(mon.history, [])


class TestConstantStream(unittest.TestCase):
    def test_constant_stream_degrades(self):
        mon = EntropyHealthMonitor(fast_config())
        for i in range(200):
            mon.add_sample(0xDEADBEEF, float(i))
        self.assertEqual(mon.state, HealthState.DEGRADED)
        failures = mon.history[-1].failures
        self.assertIn("low_entropy", failures)
        self.assertIn("repeated_blocks", failures)
        self.assertIn("constant_run", failures)

    def test_trigger_window_retained(self):
        mon = EntropyHealthMonitor(fast_config())
        for i in range(200):
            mon.add_sample(0xABCDEF01, float(i + 1000))
        trigger = mon.trigger_window
        self.assertIsNotNone(trigger)
        self.assertEqual(len(trigger.samples), mon.config.window_size)
        self.assertTrue(all(v == 0xABCDEF01 for v in trigger.samples))
        self.assertEqual(trigger.timestamps[-1], 1000.0 + 191)
        self.assertIn("constant_run", trigger.report.failures)

    def test_all_zero_and_all_one_runs(self):
        rng = random.Random(99)
        mon = EntropyHealthMonitor(fast_config())
        values = random_values(64, rng) + [0] * 10 + random_values(64, rng)
        values += [0xFFFFFFFF] * 10 + random_values(64, rng)
        for i, v in enumerate(values):
            mon.add_sample(v, float(i))
        self.assertEqual(mon.state, HealthState.DEGRADED)
        all_failures = {f for r in mon.history for f in r.failures}
        self.assertIn("all_zero_run", all_failures)
        self.assertIn("all_one_run", all_failures)


class TestSlowRepetition(unittest.TestCase):
    def test_periodic_pattern_degrades(self):
        # 16 distinct values repeating: passes naive "is it stuck?" checks
        # but repeats blocks and has low window entropy.
        pattern = [0x1000 + i for i in range(16)]
        mon = EntropyHealthMonitor(fast_config())
        for i in range(400):
            mon.add_sample(pattern[i % 16], float(i))
        self.assertEqual(mon.state, HealthState.DEGRADED)
        failures = mon.history[-1].failures
        self.assertIn("low_entropy", failures)
        self.assertIn("repeated_blocks", failures)


class TestBitBias(unittest.TestCase):
    def test_biased_stream_degrades(self):
        rng = random.Random(2024)
        mon = EntropyHealthMonitor(fast_config())
        # Clearing the low 4 bits forces a 1-bit fraction of 14/32.
        for i in range(400):
            mon.add_sample(rng.randrange(UINT32) & 0xFFFFFFF0, float(i))
        self.assertEqual(mon.state, HealthState.DEGRADED)
        all_failures = {f for r in mon.history for f in r.failures}
        self.assertIn("bit_bias", all_failures)


class TestFalsePositiveSuppression(unittest.TestCase):
    def test_short_jitter_stays_healthy(self):
        rng = random.Random(555)
        mon = EntropyHealthMonitor(fast_config())
        values = random_values(2000, rng)
        # Brief glitch: 4 identical samples (below max_constant_run=8).
        values[500:504] = [0x77777777] * 4
        # A single repeated 4-sample block further on.
        values[900:904] = values[100:104]
        states = [mon.add_sample(v, float(i)) for i, v in enumerate(values)]
        self.assertEqual(set(states), {HealthState.HEALTHY})
        self.assertIsNone(mon.trigger_window)


class TestRecoveryWindow(unittest.TestCase):
    def degrade(self, mon):
        for i in range(200):
            mon.add_sample(0x5A5A5A5A, float(i))
        self.assertEqual(mon.state, HealthState.DEGRADED)
        return 200  # next timestamp

    def test_recovery_requires_confirm_windows(self):
        rng = random.Random(42)
        mon = EntropyHealthMonitor(fast_config(confirm_windows=3))
        ts = self.degrade(mon)
        events = []  # (report_ok, state) recorded after each evaluation
        seen = len(mon.history)
        for j in range(600):
            state = mon.add_sample(rng.randrange(UINT32), float(ts + j))
            if len(mon.history) > seen:
                seen = len(mon.history)
                events.append((mon.history[-1].ok, state))
        healthy_idx = [
            i for i, (_, s) in enumerate(events) if s is HealthState.HEALTHY
        ]
        self.assertTrue(healthy_idx)
        first_healthy = healthy_idx[0]
        # Everything before recovery stays degraded.
        for ok, state in events[:first_healthy]:
            self.assertIs(state, HealthState.DEGRADED)
        # The recovering evaluation itself is a clean window...
        self.assertTrue(events[first_healthy][0])
        # ...preceded by exactly confirm_windows - 1 consecutive clean
        # evaluations that were not enough to recover.
        trailing_ok = 0
        for ok, _ in reversed(events[:first_healthy]):
            if not ok:
                break
            trailing_ok += 1
        self.assertEqual(trailing_ok, mon.config.confirm_windows - 1)

    def test_recovery_streak_reset_by_relapse(self):
        rng = random.Random(43)
        mon = EntropyHealthMonitor(fast_config(confirm_windows=2))
        ts = self.degrade(mon)
        # Flush the window with random data -> healthy evaluations begin.
        for j in range(64):
            mon.add_sample(rng.randrange(UINT32), float(ts + j))
        ts += 64
        self.assertEqual(mon.state, HealthState.DEGRADED)  # streak 1 < 2
        # Relapse: constant burst re-fails and resets the streak.
        for j in range(16):
            mon.add_sample(0x11111111, float(ts + j))
        ts += 16
        self.assertEqual(mon.state, HealthState.DEGRADED)
        # Now feed random data; needs a fresh full confirm streak.
        recovered = False
        for j in range(300):
            if mon.add_sample(rng.randrange(UINT32), float(ts + j)) is HealthState.HEALTHY:
                recovered = True
                break
        self.assertTrue(recovered)

    def test_trigger_window_kept_after_recovery(self):
        rng = random.Random(44)
        mon = EntropyHealthMonitor(fast_config())
        ts = self.degrade(mon)
        for j in range(400):
            mon.add_sample(rng.randrange(UINT32), float(ts + j))
        self.assertEqual(mon.state, HealthState.HEALTHY)
        # The forensic window is still available after recovery.
        self.assertIsNotNone(mon.trigger_window)
        self.assertFalse(mon.trigger_window.report.ok)
        self.assertTrue(mon.trigger_window.report.failures)


class TestStateTransitions(unittest.TestCase):
    def test_transition_sequence(self):
        rng = random.Random(77)
        mon = EntropyHealthMonitor(fast_config(confirm_windows=2))
        seen = [HealthState.HEALTHY]
        for i in range(200):
            seen.append(mon.add_sample(0, float(i)))
        self.assertIn(HealthState.DEGRADED, seen)
        for j in range(400):
            seen.append(mon.add_sample(rng.randrange(UINT32), float(200 + j)))
        self.assertEqual(seen[-1], HealthState.HEALTHY)
        # healthy -> degraded -> healthy, no other states
        self.assertEqual(set(seen), {HealthState.HEALTHY, HealthState.DEGRADED})


class TestPersistence(unittest.TestCase):
    def test_snapshot_restore_continues_identically(self):
        rng = random.Random(900)
        values = random_values(600, rng)
        ts = timestamps(600)
        mon = EntropyHealthMonitor(fast_config())
        for v, t in zip(values[:300], ts[:300]):
            mon.add_sample(v, t)
        restored = EntropyHealthMonitor.restore_json(mon.snapshot_json())
        self.assertEqual(mon.snapshot_json(), restored.snapshot_json())
        for v, t in zip(values[300:], ts[300:]):
            s1 = mon.add_sample(v, t)
            s2 = restored.add_sample(v, t)
            self.assertIs(s1, s2)
        self.assertEqual(mon.snapshot_json(), restored.snapshot_json())

    def test_replayed_samples_not_counted_as_new_evidence(self):
        rng = random.Random(901)
        mon = EntropyHealthMonitor(fast_config())
        values = random_values(300, rng)
        ts = timestamps(300)
        for v, t in zip(values, ts):
            mon.add_sample(v, t)
        restored = EntropyHealthMonitor.restore_json(mon.snapshot_json())
        accepted_before = restored.total_accepted
        # Restart replays the old stream from the beginning.
        for v, t in zip(values, ts):
            restored.add_sample(v, t)
        self.assertEqual(restored.total_accepted, accepted_before)
        self.assertEqual(restored.total_dropped, 300)
        # New samples with fresh timestamps are accepted again.
        state = restored.add_sample(rng.randrange(UINT32), 300.0)
        self.assertIs(state, HealthState.HEALTHY)
        self.assertEqual(restored.total_accepted, accepted_before + 1)

    def test_restore_preserves_degraded_state_and_trigger(self):
        mon = EntropyHealthMonitor(fast_config())
        for i in range(200):
            mon.add_sample(0x12345678, float(i))
        self.assertEqual(mon.state, HealthState.DEGRADED)
        restored = EntropyHealthMonitor.restore(mon.snapshot())
        self.assertEqual(restored.state, HealthState.DEGRADED)
        self.assertIsNotNone(restored.trigger_window)
        self.assertEqual(
            restored.trigger_window.samples, mon.trigger_window.samples
        )

    def test_snapshot_is_json_roundtrippable(self):
        rng = random.Random(902)
        mon = EntropyHealthMonitor(fast_config())
        for i, v in enumerate(random_values(300, rng)):
            mon.add_sample(v, float(i))
        payload = json.dumps(mon.snapshot())
        restored = EntropyHealthMonitor.restore(json.loads(payload))
        self.assertEqual(mon.snapshot(), restored.snapshot())

    def test_restore_rejects_unknown_version(self):
        mon = EntropyHealthMonitor(fast_config())
        snap = mon.snapshot()
        snap["version"] = 999
        with self.assertRaises(ValueError):
            EntropyHealthMonitor.restore(snap)


class TestMemoryBounds(unittest.TestCase):
    def test_long_run_stays_bounded(self):
        rng = random.Random(31337)
        mon = EntropyHealthMonitor(fast_config())
        for i in range(50000):
            mon.add_sample(rng.randrange(UINT32), float(i))
        self.assertEqual(mon.buffered_samples, mon.config.window_size)
        self.assertLessEqual(len(mon.history), mon.config.history_limit)
        self.assertEqual(mon.total_accepted, 50000)

    def test_trigger_window_bounded(self):
        mon = EntropyHealthMonitor(fast_config())
        for i in range(5000):
            mon.add_sample(0xAB, float(i))
        self.assertEqual(mon.state, HealthState.DEGRADED)
        self.assertLessEqual(
            len(mon.trigger_window.samples), mon.config.window_size
        )


class TestDeterminism(unittest.TestCase):
    def test_same_input_same_result(self):
        def run():
            rng = random.Random(606)
            mon = EntropyHealthMonitor(fast_config())
            states = []
            for i in range(1500):
                v = rng.randrange(UINT32) if i % 97 else 0xC0FFEE
                states.append(mon.add_sample(v, float(i)))
            return states, mon.snapshot_json()

        states_a, snap_a = run()
        states_b, snap_b = run()
        self.assertEqual(states_a, states_b)
        self.assertEqual(snap_a, snap_b)


class TestInputValidation(unittest.TestCase):
    def test_rejects_out_of_range_value(self):
        mon = EntropyHealthMonitor(fast_config())
        with self.assertRaises(ValueError):
            mon.add_sample(2 ** 32, 0.0)
        with self.assertRaises(ValueError):
            mon.add_sample(-1, 0.0)

    def test_rejects_non_int_value(self):
        mon = EntropyHealthMonitor(fast_config())
        with self.assertRaises(TypeError):
            mon.add_sample("1234", 0.0)
        with self.assertRaises(TypeError):
            mon.add_sample(True, 0.0)

    def test_rejects_non_finite_timestamp(self):
        mon = EntropyHealthMonitor(fast_config())
        with self.assertRaises(ValueError):
            mon.add_sample(1, float("nan"))
        with self.assertRaises(ValueError):
            mon.add_sample(1, float("inf"))

    def test_batch_length_mismatch(self):
        mon = EntropyHealthMonitor(fast_config())
        with self.assertRaises(ValueError):
            mon.add_batch([1, 2], [0.0])

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            EntropyHealthMonitor(fast_config(window_size=4))
        with self.assertRaises(ValueError):
            EntropyHealthMonitor(fast_config(block_size=7))
        with self.assertRaises(ValueError):
            EntropyHealthMonitor(fast_config(confirm_windows=0))


if __name__ == "__main__":
    unittest.main()
