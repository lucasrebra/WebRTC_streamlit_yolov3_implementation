import os
from streamlit_webrtc import VideoTransformerBase
import cv2
import numpy as np
import av

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_CFG_PATH = os.path.join(BASE_DIR, "../yolov3.cfg")
YOLO_WEIGHTS_PATH = os.path.join(BASE_DIR, "../yolov3.weights")
COCO_NAMES_PATH = os.path.join(BASE_DIR, "../coco.names")

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.net = cv2.dnn.readNet(YOLO_WEIGHTS_PATH, YOLO_CFG_PATH)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        with open(COCO_NAMES_PATH, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def test_video_processor():
    processor = VideoProcessor()
    img = np.zeros((480, 640, 3), np.uint8)
    frame = av.VideoFrame.from_ndarray(img, format="bgr24")
    output_frame = processor.transform(frame)
    assert output_frame is not None, "VideoProcessor.transform returned None"

