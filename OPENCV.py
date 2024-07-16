import cv2
import mss
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  

def run_inference_and_annotate(frame):
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    for r in results:
        boxes = r.boxes.xywh
        for box in boxes:
            x_center, y_center, width, height = box.tolist()
            x = int(x_center - (width / 2))
            y = int(y_center - (height / 2))
            w = int(width)
            h = int(height)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return frame

sct = mss.mss()
monitor = sct.monitors[1]  
monitor_region = {
    "top": monitor["top"],
    "left": monitor["left"],
    "width": monitor["width"] // 2, 
    "height": monitor["height"]
}
while True:
   
    sct_img = sct.grab(monitor)
    frame = np.array(sct_img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    frame = cv2.resize(frame, (640, 480))  
    annotated_frame = run_inference_and_annotate(frame)
    cv2.imshow("YOLOv8", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
