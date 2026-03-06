# Hand Landmark Media Controller

Webcam-based hand gesture controls for media playback and system volume on macOS.

This project is an MVP focused on reliable, low-friction controls while you are at your desk.

## What It Does

- Detects hand and face landmarks with MediaPipe.
- Controls macOS output volume and mute.
- Sends play/pause commands to media targets.
- Auto-pauses when your face leaves the frame for ~2 seconds.
- Shows a live camera overlay with gesture, routing, and volume state.

## Gesture Controls

| Gesture | Action | Notes |
|---|---|---|
| ✌️ 2 fingers up | Volume map by hand height | Raise hand to increase volume, lower to decrease |
| 👇 Index finger down | Volume down (step) | Decreases volume in small steps |
| 🖐️ Hand facing down (low in frame) | Volume down (step) | Hand must be low enough in frame |
| ✊ Fist (hold ~0.4s) | Toggle mute | Works as mute/unmute toggle |
| ✋ Open palm (hold ~0.5s) | Pause | Requires stable open-palm hold |
| 👍 Thumbs up (hold ~0.3s) | Play | Requires stable thumbs-up hold |
| Face missing for ~2s | Auto-pause | Triggered when face is not detected |

## Media Routing Behavior

Play/pause uses this order:

1. Frontmost native app if supported (`Spotify`, `VLC`, `QuickTime Player`)
2. Space key to the current frontmost app

Routing feedback is shown in the overlay so you can see where commands were sent.

## Installation

1. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Grant camera access:

- Open `System Settings -> Privacy & Security -> Camera`
- Allow camera access for Terminal (or the app running Python)

## Run

```bash
python gesture_controller.py
```

- Press `Q` to quit.
- Keep one hand clearly visible for best tracking.
- Use steady holds for play/pause/mute gestures.

## Requirements

- macOS (uses `osascript`)
- Python 3.8+
- Webcam

## Tech Stack

- MediaPipe
- OpenCV
- AppleScript via `osascript`

## Troubleshooting

### Camera does not start

- Confirm camera permissions are granted.
- Close other apps that may be using the webcam.

### Gestures are inconsistent

- Improve lighting and keep a plain background.
- Keep your full hand in frame.
- Hold action gestures for the required duration.

### Volume UI looks wrong

- External changes (media keys, other apps) are polled and should resync quickly.
- If needed, pause gestures for a second to let state settle.

## License

MIT
