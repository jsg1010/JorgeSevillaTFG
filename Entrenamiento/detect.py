import sys
from ultralytics import YOLO

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python detect.py <nombre_del_video>")
        sys.exit(1)
    
    nombre_video = sys.argv[1]

    # Cargar modelo YOLOv8 preentrenado
    model = YOLO("best.pt")

    # Usar el modelo para detectar objetos
    model.predict(source=nombre_video, show=True, save_txt=True, save_frames=True, imgsz=640)
