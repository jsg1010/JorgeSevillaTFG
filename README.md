# JorgeSevillaTFG

# Image Annotation Tool with YOLO Integration for Road Defect Detection

## Description
This project is part of a Final Degree Project (TFG) in Computer Engineering. It provides a graphical tool to assist in creating datasets for training YOLO models to detect road defects. The tool uses a pre-trained YOLO model to perform initial detections on images, allowing users to manually add, edit, or remove annotations. The resulting datasets can be exported in COCO, YOLO, and Pascal VOC formats for further model training.

## Features
- Load images from a directory or extract frames from a video.
- Run YOLO-based detection on individual images or all images in a directory.
- Manually annotate images with bounding boxes and class labels.
- Edit, delete, or undo/redo annotations.
- Zoom in/out on images for precise annotation.
- Export annotations in COCO, YOLO, or Pascal VOC formats with train/val/test splits.
- Mark images with annotations as [A] and deleted images as [D] in the UI.

## Dependencies
- **Python 3.8+**
- Required libraries:
  - `ultralytics`
  - `opencv-python`
  - `numpy`
  - `pillow`
  - `pandas`
  - `scikit-learn`
  - `tkinter` (typically included with Python, but may need to be installed separately on some systems, e.g., `sudo apt-get install python3-tk` on Ubuntu)

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Project Structure
- `main.py`: Entry point to launch the application.
- `app.py`: Core application class (`ImageAnnotationTool`) managing the main window and logic.
- `ui_components.py`: Functions to create UI components (menu, toolbar, sidebars, etc.).
- `annotation_utils.py`: Functions for handling annotations (drawing, editing, deleting, etc.).
- `yolo_utils.py`: Functions for loading YOLO models and running detections.
- `export_utils.py`: Functions for exporting annotations in COCO, YOLO, and VOC formats.
- `utils.py`: General utility functions (preferences management, dataset splitting).

## Usage
1. Run the application:
   ```bash
   python main.py
   ```
2. Use the "File > Open Directory" menu to load a directory of images, or "File > Load Video" to extract frames from a video.
3. Load a pre-trained YOLO model using the "Load YOLO Model" button in the toolbar.
4. Run detection on the current image or all images using the "Run Detection" or "Run Detection on ALL" buttons.
5. Manually add/edit annotations using the canvas and right sidebar.
6. Export the dataset via "File > Export" in the desired format (COCO, YOLO, or Pascal VOC).

## Screenshots

![capture1](https://github.com/user-attachments/assets/4e95c869-aca8-4424-bb07-6f8deeed87ab)

## Author
Jorge Sevilla Garcia  
Computer Engineering Degree  
UBU
July 2025

IMPORTANT!!!
To dowload the pretrained model, do so from here: https://www.swisstransfer.com/d/5bcb78fa-ce58-4c29-b184-8f63037a3e0c

![DiagramaClases](https://github.com/user-attachments/assets/c3acfe84-bbff-4f04-87bf-1c7945b89021)


