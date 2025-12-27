#!/usr/bin/env python3
"""
Hand Gesture Media Controller for Mac
Uses MediaPipe hand landmarks to control volume and media playback.

Gestures:
- 2 Fingers Up (index + middle): Control volume by moving up/down
- Fist: Pause media
- Thumbs up: Play media
"""

import cv2
import mediapipe as mp
import subprocess
import time
import numpy as np


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
        
        # Gesture state tracking
        self.last_volume_update = 0
        self.volume_cooldown = 0.1  # seconds between volume updates
        
        # Thumbs up detection
        self.thumbs_up_start_time = None
        self.thumbs_up_hold_duration = 0.3  # How long to hold thumbs up for play
        self.last_play_action = 0
        self.play_cooldown = 1.0  # Cooldown after play action
        
        # Fist detection
        self.fist_start_time = None
        self.fist_hold_duration = 0.3  # How long to hold fist for pause
        self.last_pause_action = 0
        self.pause_cooldown = 1.0
        
        # Visual feedback
        self.current_gesture = "None"
        self.current_volume = self.get_current_volume()
        
    def get_current_volume(self):
        """Get current Mac volume level (0-100)."""
        try:
            result = subprocess.run(
                ["osascript", "-e", "output volume of (get volume settings)"],
                capture_output=True, text=True
            )
            return int(result.stdout.strip())
        except:
            return 50
    
    def set_volume(self, level):
        """Set Mac volume (0-100)."""
        level = max(0, min(100, level))
        subprocess.run(
            ["osascript", "-e", f"set volume output volume {level}"],
            capture_output=True
        )
        self.current_volume = level
        
    def play_media(self):
        """Send play command using space key (media play/pause toggle)."""
        subprocess.run([
            "osascript", "-e",
            'tell application "System Events" to keystroke space'
        ], capture_output=True)
        
    def pause_media(self):
        """Send pause command."""
        subprocess.run([
            "osascript", "-e",
            'tell application "System Events" to keystroke space'
        ], capture_output=True)
        
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two landmarks."""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def is_fist(self, landmarks):
        """Detect if hand is making a fist."""
        # Finger tip landmarks
        tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
        # Finger MCP (knuckle) landmarks
        mcps = [5, 9, 13, 17]
        
        fist_count = 0
        for tip, mcp in zip(tips, mcps):
            # If fingertip is below MCP (closer to wrist), finger is curled
            if landmarks[tip].y > landmarks[mcp].y:
                fist_count += 1
        
        # Check thumb separately (using x coordinate)
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        # Thumb curled if tip is close to palm
        palm_center = landmarks[9]
        thumb_dist = self.calculate_distance(thumb_tip, palm_center)
        if thumb_dist < 0.1:
            fist_count += 1
            
        return fist_count >= 4
    
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
    
    def process_frame(self, frame):
        """Process a frame and handle gestures."""
        current_time = time.time()
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                if self.is_thumbs_up(landmarks):
                    if self.thumbs_up_start_time is None:
                        self.thumbs_up_start_time = current_time
                    elif (current_time - self.thumbs_up_start_time > self.thumbs_up_hold_duration and
                          current_time - self.last_play_action > self.play_cooldown):
                        self.play_media()
                        self.last_play_action = current_time
                        gesture_text = "👍 PLAY!"
                        self.current_gesture = "Thumbs Up - Play"
                    else:
                        gesture_text = f"👍 Hold thumbs up to play..."
                        self.current_gesture = "Thumbs Up detected"
                else:
                    self.thumbs_up_start_time = None
                    
                    # Check for fist gesture (PAUSE)
                    if self.is_fist(landmarks):
                        if self.fist_start_time is None:
                            self.fist_start_time = current_time
                        elif (current_time - self.fist_start_time > self.fist_hold_duration and
                              current_time - self.last_pause_action > self.pause_cooldown):
                            self.pause_media()
                            self.last_pause_action = current_time
                            gesture_text = "✋ PAUSED"
                            self.current_gesture = "Fist - Paused"
                        else:
                            gesture_text = f"✊ Hold fist to pause..."
                            self.current_gesture = "Fist detected"
                    else:
                        self.fist_start_time = None
                        
                        # Check for 2-finger gesture (VOLUME)
                        if self.is_two_fingers_up(landmarks):
                            gesture_text = "✌️ 2 Fingers - move up/down for volume"
                            self.current_gesture = "2 Fingers Up"
                            
                            # Use average Y position of index and middle finger tips for volume control
                            index_y = landmarks[8].y  # Index finger tip Y
                            middle_y = landmarks[12].y  # Middle finger tip Y
                            hand_y = (index_y + middle_y) / 2  # Average Y position
                            
                            if current_time - self.last_volume_update > self.volume_cooldown:
                                # Map Y position to volume (top = 100, bottom = 0)
                                target_volume = int((1 - hand_y) * 100)
                                target_volume = max(0, min(100, target_volume))
                                
                                # Smooth volume change
                                diff = target_volume - self.current_volume
                                if abs(diff) > 2:
                                    new_volume = self.current_volume + (diff // 3)
                                    self.set_volume(new_volume)
                                    self.last_volume_update = current_time
                                    
                        else:
                            gesture_text = f"👋 Open hand - 2 Fingers Up to control volume"
                            self.current_gesture = "Open Hand"
        
        # Draw UI overlay
        self.draw_overlay(frame, gesture_text)
        
        return frame
    
    def draw_overlay(self, frame, gesture_text):
        """Draw status overlay on frame."""
        h, w = frame.shape[:2]
        
        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Gesture Controller", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
        
        # Current gesture
        cv2.putText(frame, gesture_text, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Volume bar
        cv2.putText(frame, f"Volume: {self.current_volume}%", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Volume bar visual
        bar_width = int(self.current_volume * 1.5)
        cv2.rectangle(frame, (120, 90), (120 + 150, 105), (60, 60, 60), -1)
        cv2.rectangle(frame, (120, 90), (120 + bar_width, 105), (0, 255, 170), -1)
        
        # Instructions at bottom
        instructions = [
            "Controls: 2 Fingers Up + Move = Volume | Fist = Pause | Thumbs Up = Play",
            "Press 'Q' to quit"
        ]
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (20, h - 30 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    
    def run(self):
        """Main loop."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🎮 Gesture Controller Started!")
        print("━" * 40)
        print("Controls:")
        print("  ✌️  2 Fingers Up + Move Up/Down → Volume control")
        print("  ✊ Fist (hold)                  → Pause")
        print("  👍 Thumbs Up (hold)            → Play")
        print("━" * 40)
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
            print("\n👋 Gesture Controller stopped.")


if __name__ == "__main__":
    controller = GestureController()
    controller.run()

