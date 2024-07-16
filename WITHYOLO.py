import cv2
import mss
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
sct = mss.mss()
monitor = sct.monitors[1]  
monitor_region = {
    "top": monitor["top"],
    "left": monitor["left"],
    "width": monitor["width"] // 2,  
    "height": monitor["height"]
}
while True:
  
    sct_img = sct.grab(monitor_region)
    frame = np.array(sct_img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    frame = cv2.resize(frame, (640, 480))
    model.predict(frame, save=False, imgsz=256,show=True,show_boxes=True)
