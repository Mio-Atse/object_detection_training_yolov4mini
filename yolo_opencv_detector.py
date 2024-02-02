import numpy as np
import cv2 as cv
import os
import time
import threading  

class VideoProcessor:
    def __init__(self, video_path, img_size, cfg_file, weights_file, class_names):
        self.video_path = video_path
        self.frame = None
        self.classes = {}
        np.random.seed(42)
        self.net = cv.dnn.readNetFromDarknet(cfg_file, weights_file)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL_FP16)  
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i-1] for i in self.net.getUnconnectedOutLayers()]
        self.W = img_size[0]
        self.H = img_size[1]
        self.start_time = time.time()
        self.frame_count = 0

        with open('yolov4-tiny/obj.names', 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            self.classes[i] = line.strip()

        self.colors = [
            (255, 0, 0),    
            (255, 69, 0),   
            (255, 99, 71),  
            (233, 150, 122),
            (240, 128, 128),
            (205, 92, 92),  
            (219, 112, 147),
            (220, 20, 60),  
            (178, 34, 34),  
            (139, 0, 0)     
        ]

    def process_video(self):
        cap = cv.VideoCapture(self.video_path)
        while cap.isOpened():
            ret, self.frame = cap.read()
            if not ret:
                break
            
            coordinates = self.process_frame()
            self.draw_identified_objects(coordinates)

            self.show_fps()
            
            cv.namedWindow('View Window', cv.WINDOW_NORMAL)
            cv.resizeWindow('View Window', 800, 600)
            cv.imshow('View Window', self.frame)

            if cv.waitKey(1) == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()

    def process_frame(self):
        blob = cv.dnn.blobFromImage(self.frame, 1/255.0, (320, 320), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        coordinates = self.get_coordinates(outputs, 0.3)  
        return coordinates

    def get_coordinates(self, outputs, conf):
        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > conf and classID == 0:
                    center_x = int(detection[0] * self.frame.shape[1])
                    center_y = int(detection[1] * self.frame.shape[0])
                    w = int(detection[2] * self.frame.shape[1])
                    h = int(detection[3] * self.frame.shape[0])
                    x = center_x - w // 2
                    y = center_y - h // 2
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
        coordinates = []
        if len(indices) > 0:
            for i in indices.flatten():
                coordinates.append({'x': boxes[i][0], 'y': boxes[i][1], 'w': boxes[i][2], 'h': boxes[i][3]})
        return coordinates

    def draw_identified_objects(self, coordinates):
        for coordinate in coordinates:
            x, y, w, h = coordinate['x'], coordinate['y'], coordinate['w'], coordinate['h']
            color = self.colors[0]
            cv.rectangle(self.frame, (x, y), (x+w, y+h), color, 2)
            cv.putText(self.frame, self.classes[0], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def show_fps(self):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        self.frame_count += 1
        if elapsed_time > 1:
            fps = self.frame_count / elapsed_time
            print("FPS:", fps)
            self.start_time = current_time
            self.frame_count = 0


video_path = "videos/video.mp4"
cfg_file = "./yolov4-tiny/yolov4-tiny-custom.cfg"
weights_file = "yolov4-tiny-custom_last.weights"
class_names = ["red_circle"]


video_proc = VideoProcessor(video_path, (320, 320), cfg_file, weights_file, class_names)
video_proc.process_video()
