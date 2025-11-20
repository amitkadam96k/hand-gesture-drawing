import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import mediapipe as mp
import random
import threading
import time
from collections import deque

# ---------------- SETTINGS ----------------
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
PROCESS_WIDTH, PROCESS_HEIGHT = 320, 240   # smaller frame for MediaPipe processing (faster)
MAX_WIDTH, MAX_HEIGHT = 3000, 3000

BRUSH_BASE_THICKNESS = 10
SPARKLE_COUNT = 5
SPARKLE_MAX_RADIUS = 8
SPARKLE_OPACITY = 0.6

COLOR_CYAN = (255, 255, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_WHITE = (255, 255, 255)

# Camera on Big Canvas start
cam_x = (MAX_WIDTH - FRAME_WIDTH) // 2
cam_y = (MAX_HEIGHT - FRAME_HEIGHT) // 2

# ---------------- GPU CHECK ----------------
cuda_available = False
try:
    cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
    cuda_available = (cuda_count and cuda_count > 0)
except Exception:
    cuda_available = False

print(f"CUDA available: {cuda_available}")

# ---------------- MediaPipe worker ----------------
mp_hands = mp.solutions.hands

# We'll run MediaPipe in a background thread to avoid blocking the main loop.
class HandWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.frame_queue = deque(maxlen=1)
        self.result = None
        self.lock = threading.Lock()
        self.running = True

    def submit(self, frame_rgb):
        # Keep only the latest frame
        self.frame_queue.clear()
        self.frame_queue.append(frame_rgb)

    def run(self):
        while self.running:
            if self.frame_queue:
                frame = self.frame_queue.pop()
                # Do not modify frame in-place
                res = self.hands.process(frame)
                with self.lock:
                    self.result = res
            else:
                time.sleep(0.002)

    def get_result(self):
        with self.lock:
            return self.result

    def stop(self):
        self.running = False
        self.hands.close()


hand_worker = HandWorker()
hand_worker.start()

# ---------------- Canvas ----------------
big_canvas = np.zeros((MAX_HEIGHT, MAX_WIDTH, 3), np.uint8)

# If CUDA available, create GPU mats for canvas
if cuda_available:
    gpu_big_canvas = cv2.cuda_GpuMat()
    gpu_big_canvas.upload(big_canvas)
else:
    gpu_big_canvas = None

# ---------------- Capture ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

xp, yp = 0, 0
rainbow_hue = 0

print("---  Air Canvas (GPU-optimized) ---")
print("1 Finger:  DRAW (Rainbow Sparkle)")
print("2 Fingers: HOVER")
print("3 Fingers: PAN (Move Canvas)")
print("4. 'c' = Clear | 'q' = Quit")

try:
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)  # mirror

        # Submit smaller frame to MediaPipe for detection (faster)
        proc = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        proc_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        hand_worker.submit(proc_rgb)

        # Get latest detection result (non-blocking)
        results = hand_worker.get_result()

        # Draw overlay on camera for visualization
        vis = frame.copy()

        if results and results.multi_hand_landmarks:
            # Use the first hand found
            hand_lms = results.multi_hand_landmarks[0]
            # landmarks are in normalized coords relative to PROCESS_WIDTH x PROCESS_HEIGHT.
            # We'll map them to FRAME_WIDTH x FRAME_HEIGHT
            lm = hand_lms.landmark
            h_proc, w_proc = PROCESS_HEIGHT, PROCESS_WIDTH
            h_disp, w_disp = FRAME_HEIGHT, FRAME_WIDTH

            # --- draw stylized skeleton on vis (camera preview) ---
            # Draw connections and joints (scale up coordinates)
            for connection in mp_hands.HAND_CONNECTIONS:
                a = connection[0]
                b = connection[1]
                sx = int(lm[a].x * w_disp)
                sy = int(lm[a].y * h_disp)
                ex = int(lm[b].x * w_disp)
                ey = int(lm[b].y * h_disp)
                cv2.line(vis, (sx, sy), (ex, ey), COLOR_CYAN, 2)

            for idx, landmark in enumerate(lm):
                cx = int(landmark.x * w_disp)
                cy = int(landmark.y * h_disp)
                if idx in [4, 8, 12, 16, 20]:
                    cv2.circle(vis, (cx, cy), 8, COLOR_MAGENTA, cv2.FILLED)
                    cv2.circle(vis, (cx, cy), 8, COLOR_WHITE, 1)
                else:
                    cv2.circle(vis, (cx, cy), 5, COLOR_MAGENTA, 1)

            # Extract index, middle, ring tip positions in display coords
            x1 = int(lm[8].x * w_disp)
            y1 = int(lm[8].y * h_disp)
            x2 = int(lm[12].x * w_disp)
            y2 = int(lm[12].y * h_disp)
            x3 = int(lm[16].x * w_disp)
            y3 = int(lm[16].y * h_disp)

            # Determine finger up states (use proc coords for more stable comparisons)
            i_up = lm[8].y < lm[6].y
            m_up = lm[12].y < lm[10].y
            r_up = lm[16].y < lm[14].y

            # ---------------- Gesture Logic ----------------
            if i_up and m_up and r_up:
                cv2.putText(vis, "PANNING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_CYAN, 2)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                dx = x1 - xp
                dy = y1 - yp
                cam_x -= dx
                cam_y -= dy
                cam_x = max(0, min(cam_x, MAX_WIDTH - FRAME_WIDTH))
                cam_y = max(0, min(cam_y, MAX_HEIGHT - FRAME_HEIGHT))
                xp, yp = x1, y1

            elif i_up and m_up:
                # hover (do nothing)
                xp, yp = 0, 0

            elif i_up:
                # DRAW
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                global_xp = cam_x + xp
                global_yp = cam_y + yp
                global_x1 = cam_x + x1
                global_y1 = cam_y + y1

                # Rainbow color (HSV -> BGR)
                rainbow_hue = (rainbow_hue + 2) % 180
                hsv_color = np.uint8([[[rainbow_hue, 255, 255]]])
                bgr_color = tuple(int(c) for c in cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0])

                # Create a small overlay (tight bounding box) to draw the stroke + sparkles on CPU
                x_min = max(0, min(global_xp, global_x1) - 40)
                x_max = min(MAX_WIDTH, max(global_xp, global_x1) + 40)
                y_min = max(0, min(global_yp, global_y1) - 40)
                y_max = min(MAX_HEIGHT, max(global_yp, global_y1) + 40)

                w_box = x_max - x_min
                h_box = y_max - y_min
                if w_box <= 0 or h_box <= 0:
                    xp, yp = x1, y1
                    continue

                # Extract region from big_canvas (CPU)
                region = big_canvas[y_min:y_max, x_min:x_max].copy()

                # Translate local coordinates into region-local coordinates
                lx0 = global_xp - x_min
                ly0 = global_yp - y_min
                lx1 = global_x1 - x_min
                ly1 = global_y1 - y_min

                # Draw line and sparkles on region
                cv2.line(region, (lx0, ly0), (lx1, ly1), bgr_color, BRUSH_BASE_THICKNESS, lineType=cv2.LINE_AA)
                # Sparkles (draw on region)
                for _ in range(SPARKLE_COUNT):
                    ox = random.randint(-BRUSH_BASE_THICKNESS, BRUSH_BASE_THICKNESS)
                    oy = random.randint(-BRUSH_BASE_THICKNESS, BRUSH_BASE_THICKNESS)
                    rad = random.randint(1, SPARKLE_MAX_RADIUS)
                    cx_s = int(max(0, min(w_box - 1, lx1 + ox)))
                    cy_s = int(max(0, min(h_box - 1, ly1 + oy)))
                    cv2.circle(region, (cx_s, cy_s), rad, bgr_color, cv2.FILLED)

                # Blend region back onto big_canvas with alpha (CPU blend; small so it's cheap)
                # Use cv2.addWeighted locally
                blended = cv2.addWeighted(region, SPARKLE_OPACITY, big_canvas[y_min:y_max, x_min:x_max], 1 - SPARKLE_OPACITY, 0)
                big_canvas[y_min:y_max, x_min:x_max] = blended

                # If CUDA is available, upload the modified region to gpu_big_canvas to keep GPU copy synced
                if cuda_available:
                    # Upload only the modified ROI
                    gpu_big_canvas.upload(big_canvas)

                xp, yp = x1, y1

            else:
                xp, yp = 0, 0

        # ---------------- Rendering ----------------
        # Get view slice from big_canvas
        view = big_canvas[cam_y:cam_y + FRAME_HEIGHT, cam_x:cam_x + FRAME_WIDTH]

        # If CUDA is available, do gray/threshold and bitwise ops on GPU
        if cuda_available:
            gpu_view = cv2.cuda_GpuMat()
            gpu_view.upload(view)

            # Convert camera preview to GPU mat
            gpu_cam = cv2.cuda_GpuMat()
            gpu_cam.upload(vis)  # vis is the current camera frame with skeleton

            # Create grayscale of canvas view
            gpu_gray = cv2.cuda.cvtColor(gpu_view, cv2.COLOR_BGR2GRAY)
            # threshold: note cuda.threshold returns (retval, dst)
            _, gpu_thresh = cv2.cuda.threshold(gpu_gray, 50, 255, cv2.THRESH_BINARY_INV)
            gpu_inv_bgr = cv2.cuda.cvtColor(gpu_thresh, cv2.COLOR_GRAY2BGR)

            # bitwise_and (camera & inv) then bitwise_or with canvas view
            gpu_cam_and = cv2.cuda.bitwise_and(gpu_cam, gpu_inv_bgr)
            gpu_result = cv2.cuda.bitwise_or(gpu_cam_and, gpu_view)

            result = gpu_result.download()
        else:
            # CPU fallback
            img_gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
            _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
            img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(vis, img_inv)
            result = cv2.bitwise_or(img, view)

        # HUD
        cv2.putText(result, f"Pos: {cam_x}, {cam_y}", (10, FRAME_HEIGHT - 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        cv2.imshow("Cyber Air Canvas", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            big_canvas = np.zeros((MAX_HEIGHT, MAX_WIDTH, 3), np.uint8)
            if cuda_available:
                gpu_big_canvas.upload(big_canvas)

finally:
    # Cleanup
    hand_worker.stop()
    hand_worker.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
