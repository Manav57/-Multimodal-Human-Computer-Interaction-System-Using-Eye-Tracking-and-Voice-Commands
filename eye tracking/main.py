import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import threading
import customtkinter as ctk
import speech_recognition as sr
from PIL import Image, ImageTk

# --- iOS TUNING PARAMETERS ---
SMOOTHING = 0.18  # Fluid Apple-style movement
DWELL_TIME = 1.3  # Stare time for auto-click
BLINK_THRESH = 0.015  # Sensitivity for blink-clicking
SCROLL_SPEED = 300  # Pixels to scroll
# ----------------------------

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.9)
screen_w, screen_h = pyautogui.size()


class IOSEyeTrackerUltimate(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("iOS NEURAL LINK - FULL CONTROL")
        self.attributes("-fullscreen", True)
        self.configure(fg_color="#000000")

        # States
        self.phase = "START"
        self.calib_points = []
        self.target_idx = 0
        self.targets = [(150, 150), (screen_w - 150, 150), (screen_w - 150, screen_h - 150),
                        (150, screen_h - 150), (screen_w // 2, screen_h // 2)]

        self.prev_x, self.prev_y = screen_w // 2, screen_h // 2
        self.dwell_start = time.time()
        self.last_pos = (0, 0)

        self.setup_ui()
        self.cap = cv2.VideoCapture(0)

        # Start Voice Command Thread
        threading.Thread(target=self.voice_controller, daemon=True).start()
        self.main_loop()

    def setup_ui(self):
        self.canvas = ctk.CTkCanvas(self, width=screen_w, height=screen_h, bg="black", highlightthickness=0)
        self.canvas.pack()
        self.draw_start_screen()

    def draw_start_screen(self):
        self.canvas.delete("all")
        self.canvas.create_text(screen_w // 2, screen_h // 2 - 50, text="NEXUS EYE CONTROL", fill="white",
                                font=("Arial", 44, "bold"))
        self.btn = ctk.CTkButton(self, text="INITIALIZE EYE TRACKING", command=self.begin_calib,
                                 fg_color="#007AFF", hover_color="#0051A8", corner_radius=20)
        self.btn.place(relx=0.5, rely=0.7, anchor="center")

    def begin_calib(self):
        self.btn.destroy()
        self.phase = "CALIBRATING"

    def voice_controller(self):
        """Voice Command Interface for Hands-Free Control"""
        r = sr.Recognizer()
        with sr.Microphone() as source:
            while True:
                try:
                    audio = r.listen(source, phrase_time_limit=1.5)
                    cmd = r.recognize_google(audio).lower()
                    print(f"Voice Command Detected: {cmd}")

                    if "left" in cmd or "select" in cmd:
                        pyautogui.click()
                    elif "right" in cmd:
                        pyautogui.rightClick()
                    elif "scroll up" in cmd or "up" in cmd:
                        pyautogui.scroll(SCROLL_SPEED)
                    elif "scroll down" in cmd or "down" in cmd:
                        pyautogui.scroll(-SCROLL_SPEED)
                    elif "double" in cmd:
                        pyautogui.doubleClick()
                except:
                    continue

    def get_blink_dist(self, landmarks):
        """Measures eye closure for blink features"""
        # Landmark 159 (Upper eyelid) and 145 (Lower eyelid)
        return abs(landmarks[159].y - landmarks[145].y)

    def main_loop(self):
        success, frame = self.cap.read()
        if not success: return
        frame = cv2.flip(frame, 1)
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            mesh = results.multi_face_landmarks[0].landmark

            # Pupil centers (Average for stability)
            px = (mesh[468].x + mesh[473].x) / 2
            py = (mesh[468].y + mesh[473].y) / 2

            # Visual Eye Grid (iOS Style)
            self.draw_debug_grid(frame, mesh)

            if self.phase == "CALIBRATING":
                self.run_calibration_ui(px, py)
            elif self.phase == "ACTIVE":
                self.run_tracking(px, py, mesh)

        self.after(10, self.main_loop)

    def draw_debug_grid(self, frame, mesh):
        h, w, _ = frame.shape
        # Draw pupil lock indicators
        for i in [468, 473]:
            pos = (int(mesh[i].x * w), int(mesh[i].y * h))
            cv2.circle(frame, pos, 4, (0, 255, 255), -1)

    def run_calibration_ui(self, px, py):
        self.canvas.delete("all")
        tx, ty = self.targets[self.target_idx]
        self.canvas.create_oval(tx - 30, ty - 30, tx + 30, ty + 30, outline="#007AFF", width=4)
        self.canvas.create_oval(tx - 10, ty - 10, tx + 10, ty + 10, fill="white")

        self.calib_points.append({'iris': (px, py), 'screen': (tx, ty)})

        if len(self.calib_points) > (self.target_idx + 1) * 35:
            self.target_idx += 1
            if self.target_idx >= len(self.targets):
                self.phase = "ACTIVE"
                self.finalize_calibration()

    def finalize_calibration(self):
        pts = np.array([p['iris'] for p in self.calib_points])
        self.min_x, self.max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
        self.min_y, self.max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
        self.canvas.delete("all")

    def run_tracking(self, px, py, mesh):
        # 1. 100% Accurate Mapping
        target_x = np.interp(px, [self.min_x, self.max_x], [0, screen_w])
        target_y = np.interp(py, [self.min_y, self.max_y], [0, screen_h])

        # 2. iPhone-Style Interpolation (Smoothing)
        curr_x = self.prev_x + (target_x - self.prev_x) * SMOOTHING
        curr_y = self.prev_y + (target_y - self.prev_y) * SMOOTHING

        pyautogui.moveTo(curr_x, curr_y, _pause=False)

        # 3. Blink to Click (iPhone Accessibility Feature)
        if self.get_blink_dist(mesh) < BLINK_THRESH:
            pyautogui.click()
            time.sleep(0.3)  # Prevent multiple clicks

        # 4. Dwell Control (Stare to Select)
        dist = np.linalg.norm(np.array([curr_x, curr_y]) - np.array(self.last_pos))
        if dist < 20:
            if time.time() - self.dwell_start > DWELL_TIME:
                pyautogui.click()
                self.dwell_start = time.time()
        else:
            self.dwell_start = time.time()
            self.last_pos = (curr_x, curr_y)

        self.prev_x, self.prev_y = curr_x, curr_y


if __name__ == "__main__":
    app = IOSEyeTrackerUltimate()
    app.mainloop()