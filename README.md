# Mikancita: Intelligent Cat Monitoring System

Mikancita is a computer vision application that detects and records cats using your webcam. The system automatically detects when cats appear in the frame and captures them in videos or photos for later viewing.

## Project Context

Mikancita is one component of a larger cat health monitoring ecosystem. This project focuses specifically on the tracking and recording of cat activity without identifying individual cats. The overall system aims to obtain information about cat health based on tracked behavior.

### Cat Health Monitoring Pipeline

1. **Mikancita (Current Module)**: Tracks cat presence and records footage without identifying individual cats
2. **Cat Identification Module**: Takes Mikancita's recordings as input and identifies which specific cat appears in each event
3. **Behavioral Analysis**: Processes identified cat data to create usage distributions and behavior patterns
4. **Health Monitoring**: Analyzes behavioral patterns to detect potential health issues

For example, the complete system would eventually provide warnings if a specific cat uses their litter box in an abnormal manner (frequency, duration, or behavior), which could indicate health problems such as urinary obstruction. Other potential health indicators include changes in activity levels, feeding behaviors, or water consumption.

## Features

- **Real-time Cat Detection**: Uses YOLOv11n model to detect cats in webcam feed
- **Automatic Recording**: Starts recording when a cat is detected and stops when the cat leaves
- **Dual Recording Modes**: Capture videos or individual photos of detected cats
- **Region Masking**: Define specific areas of the frame for detection to avoid false positives
- **Metadata Collection**: Records timestamps, confidence scores, and detection statistics
- **Easy Configuration**: Simple configuration via command line or config file

## Installation

### Prerequisites

- Python 3.10 or higher
- Webcam or video input device
- OpenCV compatible operating system (Linux, Windows, macOS)

### Setup with Poetry

This project uses Poetry for dependency management.

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mikancita.git
   cd mikancita
   ```

2. Install dependencies with Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

### Manual Setup

If you prefer not to use Poetry, you can install dependencies manually:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install ultralytics opencv-python numpy pyyaml
   ```

## Usage

### Basic Usage

Run the monitoring system with default settings:

```bash
python src/main.py
```

Or use the provided shell script:

```bash
./run_cat_monitor.sh
```

### Video Source Options

Mikancita can use either a webcam or an RTMP stream as its video source:

```bash
# Use a specific webcam (if you have multiple cameras)
python src/main.py --webcam 1

# Use an RTMP stream
python src.main.py --rtmp rtmp://your-rtmp-server.com/live/stream
```

### Recording Mode Options

```bash
python src/main.py --mode photos  # Record individual photos (default)
python src/main.py --mode video   # Record videos
```

### Other Options

```bash
python src/main.py --mask         # Enable detection mask
python src/main.py --mask-path masks/custom_mask.png  # Use a specific mask file
```

### Interface Controls

While the application is running:
- Press `q` to quit
- Press `m` to toggle between video and photo recording modes
- Press `k` to configure detection masks

## Project Structure

```
mikancita/
├── cat_captures/       # Output directory for cat recordings
├── masks/              # Saved detection masks
├── src/
│   ├── config.py       # Configuration settings
│   ├── detector.py     # Cat detection using YOLO
│   ├── init.py         # Model initialization
│   ├── main.py         # Entry point
│   ├── mask.py         # Mask creation and management
│   ├── monitor.py      # Main application coordination
│   ├── recorder.py     # Video and photo recording
│   └── tracker.py      # Cat presence/absence tracking
├── README.md           # This file
├── NEXT_STEPS.md       # Future development plan
├── poetry.lock         # Poetry dependency lock
├── pyproject.toml      # Project configuration
└── run_cat_monitor.sh  # Convenience script
```

## Future Plans

The project roadmap includes implementing cat identification to recognize individual cats and track their visits over time. See `NEXT_STEPS.md` for the detailed implementation plan.

## Tools

This project leverages several technologies and tools:

- **YOLOv11n**: Ultra-lightweight object detection model
- **OpenCV**: Computer vision and image processing
- **Poetry**: Dependency management and packaging
- **NumPy**: Efficient numerical operations
- **PyYAML**: YAML file parsing and generation

Development assistance provided by **Claude 3.7 Sonnet**, an AI assistant from Anthropic.