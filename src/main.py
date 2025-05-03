import argparse
from camera import Camera
from evaluator import Evaluator
from recorder import Recorder
from notifier import Notifier

def main():
    parser = argparse.ArgumentParser(description="Cat Sandbox Tracker")
    parser.add_argument('--record', action='store_true', help='Record video from camera')
    parser.add_argument('--eval-image', type=str, help='Evaluate an image to determine which cat is using the sandbox')
    parser.add_argument('--eval-video', type=str, help='Evaluate a video to determine when a cat used the sandbox')
    parser.add_argument('--production', action='store_true', help='Run in production mode to monitor the sandbox in real-time')
    parser.add_argument('--display-feed', action='store_true', help='Display the video feed during operation')

    args = parser.parse_args()

    if args.record:
        camera = Camera()
        camera.start_capture()
        recorder = Recorder(camera)
        recorder.record("output.avi", display_feed=args.display_feed)
        camera.stop_capture()
    elif args.eval_image:
        evaluator = Evaluator()
        result = evaluator.evaluate_image(args.eval_image)
        print(f"Evaluation result: {result}")
    elif args.eval_video:
        evaluator = Evaluator()
        results = evaluator.evaluate_video(args.eval_video, display_feed=args.display_feed)
        print(f"Evaluation results: {results}")
    elif args.production:
        camera = Camera()
        camera.start_capture()
        evaluator = Evaluator()
        notifier = Notifier()

        while True:
            frame = camera.read_frame()
            if frame is None:
                break

            # Placeholder for real-time evaluation logic
            result = evaluator.evaluate_image(frame)  # Assuming evaluate_image can handle frames
            if result:
                notifier.notify(result, duration=5)  # Example duration

            if args.display_feed:
                cv2.imshow("Production Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    break

        if args.display_feed:
            cv2.destroyAllWindows()
        camera.stop_capture()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()