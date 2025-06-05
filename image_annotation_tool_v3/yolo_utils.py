"""
Utilities for handling YOLO model operations in the Image Annotation Tool.

This module contains functions for loading YOLO models and running detections.
"""

import tkinter as tk
import cv2
import pandas as pd
from ultralytics import YOLO
import threading
from pathlib import Path  # Added import for Path

def load_default_model(app):
    """Load the default YOLO model specified in preferences."""
    try:
        if app.default_model_path and Path(app.default_model_path).exists():  # Convert to Path and check existence
            app.yolo_model = YOLO(app.default_model_path)
            app.model_loaded = True
            app.classes = list(app.yolo_model.names.values())
            app.class_combobox['values'] = app.classes
            app.class_var.set(app.classes[0] if app.classes else "")
            app.detect_btn.configure(state=tk.NORMAL)
            app.detect_all_btn.configure(state=tk.NORMAL)
            app.model_btn.configure(text="Model Loaded")
        else:
            tk.messagebox.showwarning("Warning", f"Default model not found at {app.default_model_path}. Please load a model manually.")
    except Exception as e:
        tk.messagebox.showerror("Error", f"Failed to load default YOLO model: {e}")
        app.model_loaded = False

def load_yolo_model(app):
    """Load a YOLO model from a user-selected file."""
    try:
        model_path = tk.filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("YOLO Model", "*.pt *.pth")]
        )
        
        if model_path:
            app.yolo_model = YOLO(model_path)
            app.model_loaded = True
            app.classes = list(app.yolo_model.names.values())
            app.class_combobox['values'] = app.classes
            app.class_var.set(app.classes[0] if app.classes else "")
            app.detect_btn.configure(state=tk.NORMAL)
            app.detect_all_btn.configure(state=tk.NORMAL)
            app.model_btn.configure(text="Model Loaded")
        else:
            return
            
    except Exception as e:
        tk.messagebox.showerror("Error", f"Failed to load YOLO model: {e}")
        app.model_loaded = False
        app.detect_btn.configure(state=tk.DISABLED)
        app.detect_all_btn.configure(state=tk.DISABLED)
        app.model_btn.configure(text="Load YOLO Model")

def run_detection(app):
    """Run YOLO detection on the current image and add detected bounding boxes as annotations."""
    from annotation_utils import refresh_annotations_display

    if not app.model_loaded or app.current_image_cv is None:
        tk.messagebox.showinfo("Info", "Please load a YOLO model and an image first.")
        return
    
    try:
        results = app.yolo_model(app.current_image_cv)
        
        if hasattr(results, "pandas"):
            detections = results.pandas().xyxy[0]
            class_names = app.yolo_model.names
            detections["name"] = detections["class"].map(class_names)
        else:
            results = results[0]
            class_names = app.yolo_model.model.names
            detections = pd.DataFrame({
                "xmin": results.boxes.xyxy[:, 0].tolist(),
                "ymin": results.boxes.xyxy[:, 1].tolist(),
                "xmax": results.boxes.xyxy[:, 2].tolist(),
                "ymax": results.boxes.xyxy[:, 3].tolist(),
                "confidence": results.boxes.conf.tolist(),
                "class": results.boxes.cls.tolist(),
            })
            detections["name"] = detections["class"].map(class_names)

        if app.current_image_path in app.annotations:
            app.annotations[app.current_image_path] = []
        else:
            app.annotations[app.current_image_path] = []
        
        for _, det in detections.iterrows():
            x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
            class_name = det['name']
            
            if class_name not in app.classes:
                app.classes.append(class_name)
                app.class_combobox['values'] = app.classes
            
            app.annotations[app.current_image_path].append({
                'bbox': [x1, y1, x2, y2],
                'class': class_name
            })
        
        refresh_annotations_display(app)
        app.update_treeview_item()  # Update [A] indicator for the current image without changing selection
        app.class_combobox['values'] = app.classes
    except Exception as e:
        tk.messagebox.showerror("Error", f"Detection failed: {e}")

def run_detection_on_all(app):
    """Run YOLO detection on all images in the directory."""
    from annotation_utils import refresh_annotations_display

    if not app.model_loaded or not app.image_files:
        tk.messagebox.showinfo("Info", "Please load a YOLO model and some images first.")
        return
    
    progress_window = tk.Toplevel(app.root)
    progress_window.title("Running Detection on All Images")
    progress_window.geometry("300x100")
    progress_window.transient(app.root)
    progress_window.grab_set()
    
    # Center the progress window
    x = (app.root.winfo_screenwidth() - progress_window.winfo_width()) // 2
    y = (app.root.winfo_screenheight() - progress_window.winfo_height()) // 2
    progress_window.geometry(f"+{x}+{y}")
    
    label = tk.ttk.Label(progress_window, text="Processing images, please wait...")
    label.pack(pady=10)
    
    progress_bar = tk.ttk.Progressbar(progress_window, orient=tk.HORIZONTAL, length=200, mode='determinate')
    progress_bar.pack(pady=10)
    
    def detection_task():
        """Perform YOLO detection on all images in a separate thread."""
        try:
            total_images = len(app.image_files)
            processed_images = 0
            
            for img_path in app.image_files:
                img_cv = cv2.imread(img_path)
                if img_cv is None:
                    processed_images += 1
                    continue
                
                results = app.yolo_model(img_cv)
                
                if hasattr(results, "pandas"):
                    detections = results.pandas().xyxy[0]
                    class_names = app.yolo_model.names
                    detections["name"] = detections["class"].map(class_names)
                else:
                    results = results[0]
                    class_names = app.yolo_model.model.names
                    detections = pd.DataFrame({
                        "xmin": results.boxes.xyxy[:, 0].tolist(),
                        "ymin": results.boxes.xyxy[:, 1].tolist(),
                        "xmax": results.boxes.xyxy[:, 2].tolist(),
                        "ymax": results.boxes.xyxy[:, 3].tolist(),
                        "confidence": results.boxes.conf.tolist(),
                        "class": results.boxes.cls.tolist(),
                    })
                    detections["name"] = detections["class"].map(class_names)
                
                app.annotations[img_path] = []
                for _, det in detections.iterrows():
                    x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                    class_name = det['name']
                    
                    if class_name not in app.classes:
                        app.classes.append(class_name)
                        app.root.after(0, lambda: app.class_combobox.config(values=app.classes))
                    
                    app.annotations[img_path].append({
                        'bbox': [x1, y1, x2, y2],
                        'class': class_name
                    })
                
                processed_images += 1
                progress = (processed_images / total_images) * 100
                app.root.after(0, lambda p=progress: progress_bar.config(value=p))
                app.root.update_idletasks()
            
            app.root.after(0, progress_window.destroy)
            app.root.after(0, lambda: refresh_annotations_display(app))
            # Update Treeview without changing the current image
            current_index = app.current_image_index if 0 <= app.current_image_index < len(app.image_files) else 0
            app.root.after(0, lambda: app.populate_tree(app.current_directory))
            app.root.after(0, lambda: app.image_tree.selection_set(app.image_tree.get_children()[current_index]))
        
        except Exception as e:
            app.root.after(0, lambda: tk.messagebox.showerror("Error", f"Detection on all images failed: {e}"))
            app.root.after(0, progress_window.destroy)
    
    threading.Thread(target=detection_task, daemon=True).start()