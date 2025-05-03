import cv2

class Camera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.capture = None

    def start_capture(self):
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            raise Exception("Could not open video device")

    def stop_capture(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def read_frame(self):
        if self.capture is not None:
            ret, frame = self.capture.read()
            if not ret:
                return None
            return frame
        else:
            raise Exception("Camera is not started")