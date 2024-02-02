import numpy as np
import cv2
from time import time, sleep
import os
import threading

class VideoCapture:
    def __init__(self, video_path):
        self.video_path = video_path
        self.start_time = time()

    def get_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()

    def save_frames(self):
        frame_count = 1
        for frame in self.get_frame():
            cv2.imwrite(os.path.join("images", f"frame_{frame_count}.jpg"), frame)
            frame_count += 1
            elapsed_time = time() - self.start_time
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}", end='\r')
            sleep(0.6)

    def start_capture_thread(self):
        capture_thread = threading.Thread(target=self.save_frames)
        capture_thread.daemon = True
        capture_thread.start()

# Video dosyasının yolu
video_path = "videos/video.mp4"

# Video yakalama ve kaydetme işlemini başlat
video_cap = VideoCapture(video_path)
video_cap.start_capture_thread()
input("Çıkış yapmak için herhangi bir tuşa basın...\n")
