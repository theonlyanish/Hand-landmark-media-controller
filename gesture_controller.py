#!/usr/bin/env python3
"""
Hand Gesture Media Controller for Mac
Uses MediaPipe hand landmarks to control volume and media playback.

Gestures:
- 2 Fingers Up (index + middle): Volume control by hand height
- Index Finger Down / Hand Down (bottom 40%): Volume down
- Open Palm (stop sign) or Fist: Pause media
- Thumbs up: Play media
- Face Detection: Auto-pauses when face leaves frame (after 2 seconds)

Media target routing (play/pause):
  1. Native app (Spotify, VLC, QuickTime) if frontmost
  2. Space key to frontmost app (browsers, any player)
"""

import cv2
import mediapipe as mp
import subprocess
import time
import numpy as np

# AppleScript play/pause commands for native media apps.
_NATIVE_APP_SCRIPTS = {
    "Spotify": {
        "play":  'tell application "Spotify" to play',
        "pause": 'tell application "Spotify" to pause',
    },
    "VLC": {
        "play":  'tell application "VLC" to play',
        "pause": 'tell application "VLC" to pause',
    },
    "QuickTime Player": {
        "play":  'tell application "QuickTime Player" to play front document',
        "pause": 'tell application "QuickTime Player" to pause front document',
    },
}


class GestureController:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Face detection setup
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5,
            model_selection=0  # 0 for short-range (2 meters), 1 for full-range (5 meters)
        )
        
        # Gesture state tracking
        self.last_volume_update = 0
        self.volume_cooldown = 0.1  # seconds between volume updates
        self.volume_sync_interval = 0.4  # seconds between system volume polls
        self.last_volume_sync = 0
        
        # Thumbs up detection
        self.thumbs_up_start_time = None
        self.thumbs_up_hold_duration = 0.3  # How long to hold thumbs up for play
        self.last_play_action = 0
        self.play_cooldown = 1.0  # Cooldown after play action
        self.thumbs_up_triggered = False  # Track if action already triggered
        self.thumbs_up_was_active = False  # Track previous state
        
        # Open palm (stop sign) detection for pause
        self.open_palm_start_time = None
        self.open_palm_hold_duration = 0.5  # How long to hold open palm for pause
        self.last_pause_action = 0
        self.pause_cooldown = 1.0
        self.open_palm_triggered = False  # Track if action already triggered
        self.open_palm_was_active = False  # Track previous state
        
        # Media state tracking (to avoid unwanted toggles)
        self.media_is_playing = None  # None = unknown, True = playing, False = paused
        
        # Face tracking for auto-pause
        self.face_detected = False
        self.face_last_seen = time.time()
        self.face_absence_duration = 0.0
        self.auto_pause_threshold = 2.0  # Pause after 2 seconds of no face
        self.auto_pause_triggered = False  # Track if we already auto-paused
        
        # Visual feedback
        self.current_gesture = "None"
        self.current_volume = self.get_current_volume()
        self.last_volume_sync = time.time()
        self.route_status_text = ""
        self.route_status_until = 0.0
        
    def get_current_volume(self):
        """Get current Mac volume level (0-100)."""
        try:
            result = subprocess.run(
                ["osascript", "-e", "output volume of (get volume settings)"],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode != 0:
                return self.current_volume if hasattr(self, "current_volume") else 50
            return max(0, min(100, int(result.stdout.strip())))
        except Exception:
            return 50
    
    def set_volume(self, level):
        """Set Mac volume (0-100)."""
        level = max(0, min(100, level))
        try:
            result = subprocess.run(
                ["osascript", "-e", f"set volume output volume {level}"],
                capture_output=True,
                timeout=1
            )
            if result.returncode == 0:
                self.current_volume = level
                self.last_volume_sync = time.time()
        except Exception:
            pass

    def sync_volume_from_system(self, current_time, force=False):
        """Refresh cached volume so UI tracks external system volume changes."""
        if not force and current_time - self.last_volume_sync < self.volume_sync_interval:
            return

        system_volume = self.get_current_volume()
        if system_volume != self.current_volume:
            self.current_volume = system_volume
        self.last_volume_sync = current_time

    def _set_route_status(self, text, duration=1.8):
        """Show short-lived playback routing feedback in the overlay."""
        self.route_status_text = text
        self.route_status_until = time.time() + duration

    def _get_frontmost_app(self):
        """Return the frontmost macOS app name, or None on failure."""
        script = (
            'tell application "System Events" to '
            'name of first application process whose frontmost is true'
        )
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode != 0:
                return None
            name = result.stdout.strip()
            return name if name else None
        except Exception:
            return None
    
    def _control_native_app(self, action):
        """Control frontmost native media app (Spotify, VLC, QuickTime) via AppleScript."""
        app_name = self._get_frontmost_app()
        if app_name not in _NATIVE_APP_SCRIPTS:
            return False
        try:
            result = subprocess.run(
                ["osascript", "-e", _NATIVE_APP_SCRIPTS[app_name][action]],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                print(f"Native app {action}: {app_name}")
                return True
        except Exception:
            pass
        return False

    def _send_space_key(self):
        """Send space keystroke to the frontmost app via System Events."""
        try:
            subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to keystroke space'],
                capture_output=True, timeout=1
            )
            return True
        except Exception:
            return False

    def play_media(self):
        """Send play command.

        Routing:
          1. Frontmost native app (Spotify, VLC, QuickTime) via AppleScript
          2. Space key to frontmost app (works for browsers, any media player)
        """
        if self.media_is_playing is False or self.media_is_playing is None:
            if self._control_native_app("play"):
                self.media_is_playing = True
                self._set_route_status("Play: native app")
                return
            frontmost = self._get_frontmost_app() or "unknown"
            print(f"Play via space key -> {frontmost}")
            self._send_space_key()
            self.media_is_playing = True
            self._set_route_status(f"Play: {frontmost}")

    def pause_media(self):
        """Send pause command.

        Routing:
          1. Frontmost native app (Spotify, VLC, QuickTime) via AppleScript
          2. Space key to frontmost app (works for browsers, any media player)
        """
        if self.media_is_playing is True or self.media_is_playing is None:
            if self._control_native_app("pause"):
                self.media_is_playing = False
                self._set_route_status("Pause: native app")
                return
            frontmost = self._get_frontmost_app() or "unknown"
            print(f"Pause via space key -> {frontmost}")
            self._send_space_key()
            self.media_is_playing = False
            self._set_route_status(f"Pause: {frontmost}")
        
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two landmarks."""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def is_open_palm(self, landmarks):
        """Detect open palm (stop sign gesture) - all 5 fingers extended and spread.
        
        This is unambiguous and intuitive for 'stop/pause'.
        """
        # Check all 4 main fingers are extended (tip above PIP and MCP)
        # Finger tips: 8 (index), 12 (middle), 16 (ring), 20 (pinky)
        # Finger PIPs: 6, 10, 14, 18
        # Finger MCPs: 5, 9, 13, 17
        
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        mcps = [5, 9, 13, 17]
        
        extended_count = 0
        for tip, pip, mcp in zip(tips, pips, mcps):
            # Finger is extended if tip is above (lower y) both PIP and MCP
            if landmarks[tip].y < landmarks[pip].y and landmarks[tip].y < landmarks[mcp].y:
                extended_count += 1
        
        # Check thumb is extended (tip away from palm)
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]
        
        # Thumb extended if tip is further from wrist than MCP
        thumb_tip_dist = self.calculate_distance(thumb_tip, wrist)
        thumb_mcp_dist = self.calculate_distance(thumb_mcp, wrist)
        thumb_extended = thumb_tip_dist > thumb_mcp_dist
        
        # Check fingers are spread (not bunched together)
        # Distance between index tip and pinky tip should be significant
        index_tip = landmarks[8]
        pinky_tip = landmarks[20]
        finger_spread = abs(index_tip.x - pinky_tip.x)
        
        # Open palm: all 4 fingers extended + thumb extended + fingers spread
        is_open = (extended_count >= 4 and thumb_extended and finger_spread > 0.1)
        
        return is_open
    
    def is_two_fingers_up(self, landmarks):
        """Detect if index and middle fingers are extended (2-finger gesture)."""
        # Index finger: tip (8), PIP (6), MCP (5)
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        index_mcp = landmarks[5]
        
        # Middle finger: tip (12), PIP (10), MCP (9)
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        middle_mcp = landmarks[9]
        
        # Check if index finger is extended (tip above PIP and MCP)
        index_extended = (index_tip.y < index_pip.y and index_tip.y < index_mcp.y)
        
        # Check if middle finger is extended (tip above PIP and MCP)
        middle_extended = (middle_tip.y < middle_pip.y and middle_tip.y < middle_mcp.y)
        
        # Ring and pinky should be closed
        ring_tip = landmarks[16]
        ring_mcp = landmarks[13]
        pinky_tip = landmarks[20]
        pinky_mcp = landmarks[17]
        
        ring_closed = ring_tip.y > ring_mcp.y
        pinky_closed = pinky_tip.y > pinky_mcp.y
        
        # Thumb can be in any position (not critical for 2-finger gesture)
        
        # 2 fingers up: index and middle extended, ring and pinky closed
        return index_extended and middle_extended and ring_closed and pinky_closed
    
    def is_thumbs_up(self, landmarks):
        """Detect if hand is making a thumbs up gesture."""
        # Thumb tip and thumb IP (interphalangeal joint)
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        
        # For thumbs up, thumb should be extended (tip is above IP and MCP)
        thumb_extended = (thumb_tip.y < thumb_ip.y and thumb_tip.y < thumb_mcp.y)
        
        # Other fingers should be closed (fingertips below MCPs)
        tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
        mcps = [5, 9, 13, 17]   # Index, middle, ring, pinky MCPs
        
        fingers_closed = 0
        for tip, mcp in zip(tips, mcps):
            # If fingertip is below MCP (closer to wrist), finger is curled
            if landmarks[tip].y > landmarks[mcp].y:
                fingers_closed += 1
        
        # Thumbs up: thumb extended AND at least 3 other fingers closed
        return thumb_extended and fingers_closed >= 3
    
    def is_fist(self, landmarks):
        """Detect a closed fist (front or back of hand).

        All four fingers curled (tips below PIP joints) and thumb tucked
        (tip closer to palm center than thumb MCP is).
        """
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]

        fingers_curled = 0
        for tip, pip in zip(tips, pips):
            if landmarks[tip].y > landmarks[pip].y:
                fingers_curled += 1

        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        palm_center = landmarks[9]

        thumb_tip_dist = self.calculate_distance(thumb_tip, palm_center)
        thumb_mcp_dist = self.calculate_distance(thumb_mcp, palm_center)
        thumb_tucked = thumb_tip_dist < thumb_mcp_dist * 1.3

        return fingers_curled >= 4 and thumb_tucked

    def is_index_finger_down(self, landmarks):
        """Detect index finger pointing down (like the backhand index pointing down emoji).

        Index finger is extended downward while other fingers are curled.
        """
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        index_mcp = landmarks[5]

        # Index clearly pointing down: tip well below PIP, PIP below MCP
        index_down = (index_tip.y > index_pip.y + 0.04 and
                      index_pip.y > index_mcp.y)

        # Other fingers should NOT be extended downward
        other_tips = [12, 16, 20]
        other_pips = [10, 14, 18]
        others_not_down = 0
        for tip, pip in zip(other_tips, other_pips):
            if not (landmarks[tip].y > landmarks[pip].y + 0.04):
                others_not_down += 1

        return index_down and others_not_down >= 2

    def is_hand_facing_down(self, landmarks):
        """Detect hand facing/pointing downward with fingers extended.

        Fingertips are below the wrist and fingers are spread out (not a fist).
        """
        wrist = landmarks[0]

        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]

        # Most fingertips below the wrist (hand oriented downward)
        tips_below_wrist = sum(1 for t in tips if landmarks[t].y > wrist.y)

        # Fingers are extended (tip significantly away from PIP, not curled)
        fingers_extended = 0
        for tip, pip in zip(tips, pips):
            dist = abs(landmarks[tip].y - landmarks[pip].y)
            if dist > 0.03:
                fingers_extended += 1

        # Thumb tip also below wrist for a full downward orientation
        thumb_below = landmarks[4].y > wrist.y

        return tips_below_wrist >= 3 and fingers_extended >= 3 and thumb_below

    def process_frame(self, frame):
        """Process a frame and handle gestures."""
        current_time = time.time()
        self.sync_volume_from_system(current_time)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face for auto-pause feature
        face_results = self.face_detection.process(rgb_frame)
        if face_results.detections:
            self.face_detected = True
            self.face_last_seen = current_time
            self.face_absence_duration = 0.0
            self.auto_pause_triggered = False  # Reset when face returns
        else:
            self.face_detected = False
            self.face_absence_duration = current_time - self.face_last_seen
            
            # Auto-pause if face has been absent for threshold duration
            # Trigger if playing OR unknown (user likely watching without using gestures)
            # Only skip if we explicitly know media is paused
            if (self.face_absence_duration > self.auto_pause_threshold and 
                not self.auto_pause_triggered and 
                self.media_is_playing is not False):
                print(f"Auto-pausing: Face absent for {self.face_absence_duration:.1f}s")
                self.pause_media()
                self.auto_pause_triggered = True
        
        # Process hand gestures
        results = self.hands.process(rgb_frame)
        
        gesture_text = "No hand detected"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 170), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(0, 170, 255), thickness=2)
                )
                
                landmarks = hand_landmarks.landmark
                
                # Check for thumbs up gesture (PLAY)
                thumbs_up_active = self.is_thumbs_up(landmarks)
                open_palm_active = False  # Initialize to avoid UnboundLocalError
                
                if thumbs_up_active:
                    # Gesture just appeared (wasn't active before)
                    if not self.thumbs_up_was_active:
                        self.thumbs_up_start_time = current_time
                        self.thumbs_up_triggered = False
                    
                    # Check if we should trigger play
                    if (not self.thumbs_up_triggered and
                        current_time - self.thumbs_up_start_time > self.thumbs_up_hold_duration and
                        current_time - self.last_play_action > self.play_cooldown):
                        self.play_media()
                        self.last_play_action = current_time
                        self.thumbs_up_triggered = True
                        gesture_text = "PLAY!"
                        self.current_gesture = "Thumbs Up - Play"
                    elif not self.thumbs_up_triggered:
                        gesture_text = "Hold thumbs up to play..."
                        self.current_gesture = "Thumbs Up detected"
                    else:
                        gesture_text = "Playing..."
                        self.current_gesture = "Thumbs Up - Active"
                else:
                    # Gesture released - reset state
                    if self.thumbs_up_was_active:
                        self.thumbs_up_start_time = None
                        self.thumbs_up_triggered = False
                    
                    # Check for pause gestures: open palm OR fist
                    is_palm = self.is_open_palm(landmarks)
                    is_closed_fist = self.is_fist(landmarks)
                    open_palm_active = is_palm or is_closed_fist
                    pause_label = "Fist" if is_closed_fist else "Open Palm"
                    
                    if open_palm_active:
                        # Gesture just appeared (wasn't active before)
                        if not self.open_palm_was_active:
                            self.open_palm_start_time = current_time
                            self.open_palm_triggered = False
                        
                        # Check if we should trigger pause
                        if (not self.open_palm_triggered and
                            current_time - self.open_palm_start_time > self.open_palm_hold_duration and
                            current_time - self.last_pause_action > self.pause_cooldown):
                            self.pause_media()
                            self.last_pause_action = current_time
                            self.open_palm_triggered = True
                            gesture_text = "PAUSED"
                            self.current_gesture = f"{pause_label} - Paused"
                        elif not self.open_palm_triggered:
                            gesture_text = f"Hold {pause_label.lower()} to pause..."
                            self.current_gesture = f"{pause_label} detected"
                        else:
                            gesture_text = "Paused..."
                            self.current_gesture = f"{pause_label} - Active"
                    else:
                        # Gesture released - reset state
                        if self.open_palm_was_active:
                            self.open_palm_start_time = None
                            self.open_palm_triggered = False
                        
                        # Check for 2-finger gesture (VOLUME)
                        if self.is_two_fingers_up(landmarks):
                            index_y = landmarks[8].y
                            middle_y = landmarks[12].y
                            hand_y = (index_y + middle_y) / 2

                            clamped_y = max(0.05, min(0.95, hand_y))
                            target_volume = int(((0.95 - clamped_y) / 0.9) * 100)
                            target_volume = max(0, min(100, target_volume))
                            
                            gesture_text = f"Volume: {target_volume}%"
                            self.current_gesture = "2 Fingers Up"
                            
                            if current_time - self.last_volume_update > self.volume_cooldown:
                                diff = target_volume - self.current_volume
                                if abs(diff) > 1:
                                    step = max(2, abs(diff) // 2)
                                    if diff > 0:
                                        new_volume = min(100, self.current_volume + step)
                                    else:
                                        new_volume = max(0, self.current_volume - step)
                                    self.set_volume(new_volume)
                                    self.last_volume_update = current_time

                        # Volume DOWN: index finger pointing down
                        elif self.is_index_finger_down(landmarks):
                            gesture_text = f"Vol Down: {self.current_volume}%"
                            self.current_gesture = "Index Finger Down"
                            if (current_time - self.last_volume_update > self.volume_cooldown
                                    and self.current_volume > 0):
                                new_volume = max(0, self.current_volume - 2)
                                self.set_volume(new_volume)
                                self.last_volume_update = current_time

                        # Volume DOWN: hand facing down in bottom 20% of frame
                        elif self.is_hand_facing_down(landmarks):
                            wrist_y = landmarks[0].y
                            if wrist_y > 0.60:
                                gesture_text = f"Vol Down: {self.current_volume}%"
                                self.current_gesture = "Hand Down"
                                if (current_time - self.last_volume_update > self.volume_cooldown
                                        and self.current_volume > 0):
                                    new_volume = max(0, self.current_volume - 2)
                                    self.set_volume(new_volume)
                                    self.last_volume_update = current_time
                            else:
                                gesture_text = "Move hand lower to decrease vol"
                                self.current_gesture = "Hand Down (too high)"

                        else:
                            gesture_text = "Hand open - show 2 fingers for vol"
                            self.current_gesture = "Open Hand"
                
                # Update previous state for next frame
                self.thumbs_up_was_active = thumbs_up_active
                self.open_palm_was_active = open_palm_active
        
        # Draw face detection indicator
        self.draw_face_indicator(frame)
        
        # Draw UI overlay
        self.draw_overlay(frame, gesture_text)
        
        return frame
    
    def draw_face_indicator(self, frame):
        """Draw face detection status indicator."""
        h, w = frame.shape[:2]
        
        # Face indicator in top-right corner
        indicator_x = w - 150
        indicator_y = 20
        
        if self.face_detected:
            # Green indicator - face detected
            cv2.circle(frame, (indicator_x + 10, indicator_y + 10), 8, (0, 255, 0), -1)
            cv2.putText(frame, "Face", (indicator_x + 25, indicator_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # Red indicator - no face
            cv2.circle(frame, (indicator_x + 10, indicator_y + 10), 8, (0, 0, 255), -1)
            cv2.putText(frame, "No Face", (indicator_x + 25, indicator_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Show countdown if face is absent
            if self.face_absence_duration > 0:
                remaining = max(0, self.auto_pause_threshold - self.face_absence_duration)
                if remaining > 0 and not self.auto_pause_triggered:
                    cv2.putText(frame, f"Auto-pause in: {remaining:.1f}s", 
                               (indicator_x, indicator_y + 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                elif self.auto_pause_triggered:
                    cv2.putText(frame, "Auto-paused", 
                               (indicator_x, indicator_y + 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    def draw_overlay(self, frame, gesture_text):
        """Draw status overlay on frame."""
        h, w = frame.shape[:2]
        
        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (430, 140), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Gesture Controller", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
        
        # Current gesture
        cv2.putText(frame, gesture_text, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if time.time() < self.route_status_until:
            cv2.putText(frame, self.route_status_text, (20, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 215, 255), 1)
        
        # Volume bar
        cv2.putText(frame, f"Volume: {self.current_volume}%", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Volume bar visual
        bar_width = int(self.current_volume * 1.5)
        cv2.rectangle(frame, (120, 90), (120 + 150, 105), (60, 60, 60), -1)
        cv2.rectangle(frame, (120, 90), (120 + bar_width, 105), (0, 255, 170), -1)
        
        # Instructions at bottom with dark background for readability
        instructions = [
            "2 Fingers=Vol | Finger/Hand Down=Vol- | Palm/Fist=Pause | Thumbs Up=Play",
            "Press 'Q' to quit"
        ]
        bar_overlay = frame.copy()
        cv2.rectangle(bar_overlay, (0, h - 55), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(bar_overlay, 0.7, frame, 0.3, 0, frame)
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (20, h - 30 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    def run(self):
        """Main loop."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Gesture Controller Started!")
        print("-" * 40)
        print("Controls:")
        print("  2 Fingers Up + Move Up/Down -> Volume control")
        print("  Index Finger Down           -> Volume down")
        print("  Hand Down (bottom 40%)      -> Volume down")
        print("  Open Palm / Fist            -> Pause")
        print("  Thumbs Up (hold)            -> Play")
        print("  Face Detection              -> Auto-pause when you leave")
        print("")
        print("Play/Pause targets:")
        print("  1. Native app (Spotify, VLC, QuickTime) if frontmost")
        print("  2. Space key to frontmost app (browsers, any player)")
        print("-" * 40)
        print("Press 'Q' in the window to quit")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process gestures
                frame = self.process_frame(frame)
                
                # Display
                cv2.imshow("Gesture Controller", frame)
                
                # Quit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\nGesture Controller stopped.")


if __name__ == "__main__":
    controller = GestureController()
    controller.run()
