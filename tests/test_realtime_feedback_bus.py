import time
import unittest

from realtime_feedback_bus import RealtimeFeedbackBus


class TestRealtimeFeedbackBus(unittest.TestCase):
    def setUp(self):
        self.bus = RealtimeFeedbackBus()
        self.events = []

    def test_subscribe_and_notify(self):
        def listener(event, data):
            self.events.append((event, data))

        self.bus.subscribe(listener)
        self.bus.notify("param_change", {"param": "gain", "value": 0.8})
        self.assertIn(("param_change", {"param": "gain", "value": 0.8}), self.events)

    def test_latency_warning(self):
        pass

        def slow_listener(event, data):
            time.sleep(0.02)  # 20ms

        self.bus.subscribe(slow_listener)
        # Capture print output
        import io
        import sys

        captured = io.StringIO()
        sys.stdout = captured
        self.bus.notify("slow_event", {})
        sys.stdout = sys.__stdout__
        self.assertIn("überschreitet 10ms", captured.getvalue())

    def test_clear(self):
        self.bus.subscribe(lambda e, d: None)
        self.bus.clear()
        self.assertEqual(len(self.bus._listeners), 0)


if __name__ == "__main__":
    unittest.main()
