# Cat Sandbox Tracker

This project is designed to monitor which of three cats are using a sandbox and when they do so. It utilizes a camera to capture video and images, processes the data to identify the cats, and provides output via the command line.

## Project Structure

```
cat-sandbox-tracker
├── src
│   ├── main.py          # Entry point for the application
│   ├── camera.py        # Manages video capture from the camera
│   ├── evaluator.py     # Evaluates images and videos to identify cats
│   ├── recorder.py      # Handles recording video from the camera
│   ├── notifier.py      # Sends messages to stdout when a cat finishes using the sandbox
│   └── utils
│       └── __init__.py  # Utility functions
├── tests
│   ├── test_camera.py   # Unit tests for the Camera class
│   ├── test_evaluator.py # Unit tests for the Evaluator class
│   ├── test_recorder.py  # Unit tests for the Recorder class
│   └── test_notifier.py  # Unit tests for the Notifier class
├── .devcontainer
│   ├── devcontainer.json # Configuration for the development container
│   └── Dockerfile        # Docker image for development
├── .vscode
│   ├── launch.json       # Debugging configuration
│   ├── tasks.json        # VSCode tasks for building and running the project
│   └── settings.json     # Workspace-specific settings
├── Dockerfile             # Docker image for production
├── requirements.txt       # Python dependencies
├── uv.yml                 # Package management with uv
├── README.md              # Project documentation
└── .gitignore             # Files to ignore in version control
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/cat-sandbox-tracker.git
   cd cat-sandbox-tracker
   ```

2. Set up the development environment using Docker:
   ```
   cd .devcontainer
   docker build -t cat-sandbox-tracker-dev .
   ```

3. Install dependencies:
   ```
   uv install
   ```

## Usage

The application can be run in different modes based on command-line arguments:

- **Record Mode**: To record video from the camera and save it to disk:
  ```
  python src/main.py --record
  ```

- **Evaluate Image**: To evaluate a single image and determine which cat is using the sandbox:
  ```
  python src/main.py --eval-image <path_to_image>
  ```

- **Evaluate Video**: To evaluate a video and determine when a cat used the sandbox:
  ```
  python src/main.py --eval-video <path_to_video>
  ```

- **Production Mode**: To continuously monitor the camera and report when a cat finishes using the sandbox:
  ```
  python src/main.py --production
  ```

## Topics to Learn

To effectively work on this project, consider learning the following topics:

- Python programming fundamentals
- Working with command-line arguments in Python
- Video processing and image recognition techniques
- Using OpenCV or similar libraries for video capture and processing
- Docker basics for containerization
- Unit testing in Python
- Using VSCode for development and debugging

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. Your feedback and contributions are welcome!

## License

This project is licensed under the MIT License. See the LICENSE file for details.