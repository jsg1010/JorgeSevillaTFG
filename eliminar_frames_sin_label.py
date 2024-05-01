import os

# Directorios
labels_dir = "runs/detect/predict3/labels"
frames_dir = "runs/detect/predict3/video2_frames"

# Obtener números de los archivos de texto
label_numbers = set()
for file in os.listdir(labels_dir):
    if file.startswith("video2_") and file.endswith(".txt"):
        label_numbers.add(file.split("_")[1].split(".")[0])

# Obtener números de los archivos de imagen
frame_numbers = set()
for file in os.listdir(frames_dir):
    if file.endswith(".jpg"):
        frame_numbers.add(file.split(".")[0])

# Eliminar archivos de imagen que no tienen un archivo de texto correspondiente
for frame_number in frame_numbers:
    if frame_number not in label_numbers:
        os.remove(os.path.join(frames_dir, frame_number + ".jpg"))


