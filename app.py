import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO

app = FastAPI(
    title="Warehouse Hazard Detection Microservice",
    version="1.0",
)


model = YOLO("runs/detect/train-2/weights/best.pt")


@app.get("/")
def home():
    return {
        "status": "online",
        "model": "Warehouse Safety YOLOv8",
        "gpu": "Active",
    }


@app.post("/detect")
async def detect_hazards(file: UploadFile = File(...)):
    
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(img, device=0)

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            xyxy = box.xyxy[0].tolist() 

            detections.append(
                {
                    "class": class_name,
                    "confidence": round(conf, 2),
                    "bounding_box": {
                        "xmin": round(xyxy[0], 2),
                        "ymin": round(xyxy[1], 2),
                        "xmax": round(xyxy[2], 2),
                        "ymax": round(xyxy[3], 2),
                    },
                }
            )

    return {"total_detections": len(detections), "detections": detections}