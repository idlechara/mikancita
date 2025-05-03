import torch
from ultralytics import YOLO

def export_model():
    # Load YOLOv11n model
    model = YOLO('yolo11n.pt')  # Load YOLOv11n model
    
    # Export the model to NCNN format
    model.export(format='ncnn')
    print("YOLOv11n model exported to NCNN format successfully")

if __name__ == "__main__":
    export_model()