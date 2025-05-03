import unittest
from src.recorder import Recorder

class TestRecorder(unittest.TestCase):
    def setUp(self):
        self.recorder = Recorder()

    def test_start_recording(self):
        self.recorder.start_recording()
        self.assertTrue(self.recorder.is_recording)

    def test_stop_recording(self):
        self.recorder.start_recording()
        self.recorder.stop_recording()
        self.assertFalse(self.recorder.is_recording)

    def test_save_video(self):
        self.recorder.start_recording()
        self.recorder.stop_recording()
        self.recorder.save_video("test_video.mp4")
        self.assertTrue(self.recorder.video_saved)

if __name__ == '__main__':
    unittest.main()