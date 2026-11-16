from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="dataset/data.yaml",
        epochs=15,
        imgsz=640,
        device=0,
        batch=8,
    )

    print("Training complete")

