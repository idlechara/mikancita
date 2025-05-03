import unittest
from src.notifier import Notifier

class TestNotifier(unittest.TestCase):
    def setUp(self):
        self.notifier = Notifier()

    def test_notify_cat_finish(self):
        cat_name = "Cat A"
        duration = 5
        expected_message = f"{cat_name} finished using the sandbox in {duration} seconds."
        
        with self.assertLogs(level='INFO') as log:
            self.notifier.notify_cat_finish(cat_name, duration)
            self.assertIn(expected_message, log.output)

    def test_notify_no_cat(self):
        duration = 0
        expected_message = "No cat finished using the sandbox."
        
        with self.assertLogs(level='INFO') as log:
            self.notifier.notify_cat_finish(None, duration)
            self.assertIn(expected_message, log.output)

if __name__ == '__main__':
    unittest.main()