import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_CFG_PATH = os.path.join(BASE_DIR, "../yolov3.cfg")
YOLO_WEIGHTS_PATH = os.path.join(BASE_DIR, "../yolov3.weights")
COCO_NAMES_PATH = os.path.join(BASE_DIR, "../coco.names")

# Load YOLO model
net = cv2.dnn.readNet(YOLO_WEIGHTS_PATH, YOLO_CFG_PATH)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open(COCO_NAMES_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]

def test_yolo_model_loading():
    assert net is not None, "Failed to load YOLO model"

def test_yolo_detection():
    # Create a dummy image
    img = np.zeros((416, 416, 3), np.uint8)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    assert len(outs) > 0, "YOLO forward pass returned no output"

def test_yolo_classes():
    assert len(classes) > 0, "No classes loaded from coco.names"

