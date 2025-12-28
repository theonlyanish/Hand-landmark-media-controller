# Hand Gesture Media Controller

Control your Mac's volume and media playback using hand gestures detected via your webcam. Built with MediaPipe hand landmark detection.

## Features

| Gesture | Action |
|---------|--------|
| 🤏 **Pinch + Move** | Control volume (move hand up/down while pinching) |
| ✊ **Fist** (hold) | Pause media |
| 👆 **Double Tap** (towards camera) | Play media |

## Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Grant camera permissions**:
   - Go to **System Preferences → Security & Privacy → Camera**
   - Enable camera access for Terminal (or your IDE)

## Usage

```bash
python gesture_controller.py
```

- A window will open showing your webcam feed with hand tracking overlay
- Use the gestures listed above to control media
- Press **Q** to quit

## How It Works

- **Volume Control**: Pinch your thumb and index finger together, then move your hand up (volume up) or down (volume down)
- **Pause**: Make a fist and hold for ~0.3 seconds
- **Play**: Push your hand towards the camera twice quickly (double tap motion)

## Requirements

- macOS (uses `osascript` for system control)
- Python 3.8+
- Webcam

## Tech Stack

- **MediaPipe** - Hand landmark detection
- **OpenCV** - Webcam capture and display
- **osascript** - Mac system volume/media control

## Troubleshooting

**Camera not working?**
- Ensure camera permissions are granted in System Preferences
- Try closing other apps that might be using the camera

**Volume control not working?**
- Make sure no other app has exclusive audio control
- Check that system volume isn't muted

**Gestures not detected?**
- Ensure good lighting
- Keep your hand within the camera frame
- Avoid cluttered backgrounds

## License

MIT
