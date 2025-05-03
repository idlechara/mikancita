import cv2

class Recorder:
    def __init__(self, camera):
        self.camera = camera
        self.is_recording = False
        self.video_writer = None

    def start_recording(self, output_file):
        self.is_recording = True
        print(f"Recording started, saving to {output_file}")

        # Open a VideoWriter to save the video
        self.video_writer = cv2.VideoWriter(
            output_file,
            cv2.VideoWriter_fourcc(*'XVID'),  # Codec for .avi files
            20.0,  # Frames per second
            (640, 480)  # Frame size (adjust based on your camera)
        )

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()  # Release the VideoWriter
            print("Recording stopped")

    def record(self, output_file, display_feed=False):
        self.start_recording(output_file)

        while self.is_recording:
            frame = self.camera.read_frame()  # Capture a frame from the camera
            if frame is not None:
                self.video_writer.write(frame)  # Write the frame to the video file
                if display_feed:
                    cv2.imshow("Recording Feed", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                        self.stop_recording()
                        break
            else:
                print("No frame captured. Stopping recording.")
                self.stop_recording()
                break

        if display_feed:
            cv2.destroyAllWindows()