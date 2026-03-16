"""
Integrationstest für Module Cooperation & Inter-Communication
"""

import unittest

from module_communication_bus import ModuleCommunicationBus


class TestModuleCommunicationBus(unittest.TestCase):
    def test_publish_subscribe(self):
        bus = ModuleCommunicationBus()
        received = []

        def callback(msg):
            received.append(msg)

        bus.subscribe("test_topic", callback)
        bus.publish("test_topic", {"foo": 42})
        self.assertEqual(received, [{"foo": 42}])

    def test_unsubscribe(self):
        bus = ModuleCommunicationBus()
        received = []

        def callback(msg):
            received.append(msg)

        bus.subscribe("topic", callback)
        bus.unsubscribe("topic", callback)
        bus.publish("topic", {"bar": 1})
        self.assertEqual(received, [])

    def test_message_history(self):
        bus = ModuleCommunicationBus()
        bus.publish("a", 1)
        bus.publish("b", 2)
        history = bus.get_message_history()
        self.assertEqual(history, [{"topic": "a", "message": 1}, {"topic": "b", "message": 2}])


if __name__ == "__main__":
    unittest.main()
