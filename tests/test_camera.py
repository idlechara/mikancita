import unittest
from src.camera import Camera

class TestCamera(unittest.TestCase):

    def setUp(self):
        self.camera = Camera()

    def test_start_capture(self):
        self.camera.start_capture()
        self.assertTrue(self.camera.is_capturing)

    def test_stop_capture(self):
        self.camera.start_capture()
        self.camera.stop_capture()
        self.assertFalse(self.camera.is_capturing)

    def test_capture_frame(self):
        self.camera.start_capture()
        frame = self.camera.capture_frame()
        self.assertIsNotNone(frame)
        self.camera.stop_capture()

if __name__ == '__main__':
    unittest.main()