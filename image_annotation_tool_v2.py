import os
import json
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import glob
from datetime import datetime
from ultralytics import YOLO
import pandas as pd
import threading
import time

class ImageAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Annotation Tool with YOLO Integration")
        self.root.geometry("1200x800")
        
        # Variables
        self.current_image_path = None
        self.current_image = None
        self.current_image_cv = None
        self.image_files = []
        self.current_image_index = -1
        self.annotations = {}
        self.classes = []
        self.drawing = False
        self.start_x, self.start_y = 0, 0
        self.rect_id = None
        self.temp_rect = None
        self.selected_annotation_index = -1
        self.yolo_model = None
        self.default_model_path = "train/crack_detection/yolo11l_exp1/weights/best.pt"
        self.model_loaded = False
        self.original_image_size = True
        self.zoom_factor = 1.0
        self.current_directory = ""
        
        # Preferencias
        self.preferences_file = "preferences.json"
        self.preferences = {
            "frame_extraction_interval": 10,  # Intervalo en segundos
            "default_model_path": self.default_model_path
        }
        self.load_preferences()

        # Create the main layout
        self.create_menu()
        self.create_toolbar()
        self.create_left_sidebar()
        self.create_main_frame()
        self.create_right_sidebar()
        
        # Bind keyboard shortcuts
        self.root.bind("<Delete>", self.delete_selected_annotation)
        self.root.bind("<Left>", lambda event: self.prev_image())
        self.root.bind("<Right>", lambda event: self.next_image())

        self.undo_stack = []
        self.redo_stack = []

        # Bind mouse events for moving/resizing bboxes
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-4>", self.zoom)
        self.canvas.bind("<Button-5>", self.zoom)

        # Bind keyboard shortcuts for undo/redo
        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Control-y>", self.redo)

        # Bind close event to save preferences
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Load the default YOLO model at startup
        self.load_default_model()
    
    def load_preferences(self):
        """Cargar preferencias desde un archivo JSON."""
        try:
            if os.path.exists(self.preferences_file):
                with open(self.preferences_file, 'r') as f:
                    loaded_preferences = json.load(f)
                    self.preferences.update(loaded_preferences)
                    # Actualizar default_model_path con el valor cargado
                    self.default_model_path = self.preferences["default_model_path"]
        except Exception as e:
            messagebox.showwarning("Warning", f"Failed to load preferences: {e}")

    def save_preferences(self):
        """Guardar preferencias en un archivo JSON."""
        try:
            with open(self.preferences_file, 'w') as f:
                json.dump(self.preferences, f, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save preferences: {e}")

    def on_closing(self):
        """Guardar preferencias y cerrar la aplicación."""
        self.save_preferences()
        self.root.destroy()
    
    def load_default_model(self):
        try:
            if os.path.exists(self.default_model_path):
                self.yolo_model = YOLO(self.default_model_path)
                self.model_loaded = True
                self.classes = list(self.yolo_model.names.values())
                self.class_combobox['values'] = self.classes
                self.class_var.set(self.classes[0] if self.classes else "")
                self.detect_btn.configure(state=tk.NORMAL)
                self.detect_all_btn.configure(state=tk.NORMAL)
                self.model_btn.configure(text="Model Loaded")
            else:
                messagebox.showwarning("Warning", f"Default model not found at {self.default_model_path}. Please load a model manually.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load default YOLO model: {e}")
            self.model_loaded = False
    
    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Directory", command=self.open_directory)
        file_menu.add_command(label="Load Video", command=self.load_video)
        file_menu.add_separator()
        file_menu.add_command(label="Save Annotations", command=self.save_annotations)
        file_menu.add_separator()

        export_menu = tk.Menu(file_menu, tearoff=0)
        export_menu.add_command(label="COCO Format", command=lambda: self.export_annotations("coco"))
        export_menu.add_command(label="YOLO Format", command=lambda: self.export_annotations("yolo"))
        export_menu.add_command(label="Pascal VOC Format", command=lambda: self.export_annotations("voc"))
        file_menu.add_cascade(label="Export", menu=export_menu)
        
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        model_menu = tk.Menu(menubar, tearoff=0)
        model_menu.add_command(label="Load YOLO Model", command=self.load_yolo_model)
        model_menu.add_command(label="Run Detection", command=self.run_detection)
        menubar.add_cascade(label="Model", menu=model_menu)
        
        # Menú de Preferencias
        preferences_menu = tk.Menu(menubar, tearoff=0)
        preferences_menu.add_command(label="Settings", command=self.open_preferences)
        menubar.add_cascade(label="Preferences", menu=preferences_menu)
        
        self.root.config(menu=menubar)
    
    def open_preferences(self):
        """Abrir ventana de preferencias."""
        pref_window = tk.Toplevel(self.root)
        pref_window.title("Preferences")
        pref_window.geometry("400x200")
        pref_window.transient(self.root)
        pref_window.grab_set()

        # Centrar la ventana
        pref_window.update_idletasks()
        x = (self.root.winfo_screenwidth() - pref_window.winfo_width()) // 2
        y = (self.root.winfo_screenheight() - pref_window.winfo_height()) // 2
        pref_window.geometry(f"+{x}+{y}")

        # Frame Extraction Interval
        tk.Label(pref_window, text="Frame Extraction Interval (seconds):").pack(pady=5)
        interval_var = tk.IntVar(value=self.preferences["frame_extraction_interval"])
        interval_entry = ttk.Entry(pref_window, textvariable=interval_var)
        interval_entry.pack(pady=5)

        # Default Model Path
        tk.Label(pref_window, text="Default YOLO Model Path:").pack(pady=5)
        model_frame = tk.Frame(pref_window)
        model_frame.pack(pady=5, fill=tk.X, padx=5)
        model_var = tk.StringVar(value=self.preferences["default_model_path"])
        model_entry = ttk.Entry(model_frame, textvariable=model_var)
        model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(model_frame, text="Browse", command=lambda: self.select_model_path(model_var)).pack(side=tk.LEFT)

        # Botones para guardar o cancelar
        btn_frame = tk.Frame(pref_window)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Save", command=lambda: self.save_preferences_from_window(
            pref_window, interval_var.get(), model_var.get())).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=pref_window.destroy).pack(side=tk.LEFT, padx=5)

    def select_model_path(self, model_var):
        """Permitir al usuario seleccionar un archivo de modelo YOLO."""
        model_path = filedialog.askopenfilename(
            title="Select Default YOLO Model",
            filetypes=[("YOLO Model", "*.pt *.pth")]
        )
        if model_path:
            model_var.set(model_path)

    def save_preferences_from_window(self, window, interval, model_path):
        """Guardar las preferencias desde la ventana de configuración."""
        try:
            # Validar el intervalo
            if interval <= 0:
                messagebox.showerror("Error", "Frame extraction interval must be greater than 0.")
                return
            
            # Validar el archivo del modelo
            if model_path and not os.path.exists(model_path):
                messagebox.showerror("Error", "Selected model file does not exist.")
                return
            
            # Actualizar preferencias
            self.preferences["frame_extraction_interval"] = interval
            self.preferences["default_model_path"] = model_path
            self.default_model_path = model_path
            
            # Intentar cargar el nuevo modelo por defecto
            if model_path:
                self.load_default_model()
            
            window.destroy()
            messagebox.showinfo("Success", "Preferences saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save preferences: {e}")
    
    def create_toolbar(self):
        toolbar_frame = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        
        prev_btn = ttk.Button(toolbar_frame, text="Previous", command=self.prev_image)
        prev_btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        next_btn = ttk.Button(toolbar_frame, text="Next", command=self.next_image)
        next_btn.pack(side=tk.LEFT, padx=2, pady=2)

        delete_btn = ttk.Button(toolbar_frame, text="Delete Image", command=self.delete_current_image)
        delete_btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)
        
        delete_btn = ttk.Button(toolbar_frame, text="Delete Annotation", command=self.delete_selected_annotation)
        delete_btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)
        
        self.model_btn = ttk.Button(toolbar_frame, text="Load YOLO Model", command=self.load_yolo_model)
        self.model_btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.detect_btn = ttk.Button(toolbar_frame, text="Run Detection", command=self.run_detection, state=tk.DISABLED)
        self.detect_btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.detect_all_btn = ttk.Button(toolbar_frame, text="Run Detection on ALL", command=self.run_detection_on_all, state=tk.DISABLED)
        self.detect_all_btn.pack(side=tk.LEFT, padx=2, pady=2)
    
    def create_left_sidebar(self):
        self.left_sidebar = tk.Frame(self.root, width=250, bd=2, relief=tk.RAISED)
        self.left_sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        images_frame = tk.LabelFrame(self.left_sidebar, text="Images", padx=5, pady=5)
        images_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.image_tree = ttk.Treeview(images_frame, show="tree")
        self.image_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.image_tree.bind("<<TreeviewSelect>>", self.on_tree_select)
    
    def create_main_frame(self):
        self.main_frame = tk.Frame(self.root, bd=2, relief=tk.SUNKEN)
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.main_frame, bg="gray90", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        h_scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonPress-3>", self.on_right_click)
    
    def create_right_sidebar(self):
        self.sidebar = tk.Frame(self.root, width=250, bd=2, relief=tk.RAISED)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        classes_frame = tk.LabelFrame(self.sidebar, text="Classes", padx=5, pady=5)
        classes_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.class_var = tk.StringVar(value=self.classes[0] if self.classes else "")
        self.class_combobox = ttk.Combobox(classes_frame, textvariable=self.class_var, values=self.classes)
        self.class_combobox.pack(fill=tk.X, padx=5, pady=5)
        
        add_class_btn = ttk.Button(classes_frame, text="Add Class", command=self.add_class)
        add_class_btn.pack(fill=tk.X, padx=5, pady=5)
        
        annotations_frame = tk.LabelFrame(self.sidebar, text="Annotations", padx=5, pady=5)
        annotations_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.annotation_tree = ttk.Treeview(annotations_frame, columns=("Class", "Coordinates"), show="headings")
        self.annotation_tree.heading("Class", text="Class")
        self.annotation_tree.heading("Coordinates", text="Coordinates")
        self.annotation_tree.column("Class", width=100)
        self.annotation_tree.column("Coordinates", width=120)
        self.annotation_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.annotation_tree.bind("<<TreeviewSelect>>", self.on_annotation_select)
        
        btn_frame = tk.Frame(annotations_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        delete_btn = ttk.Button(btn_frame, text="Delete", command=self.delete_selected_annotation)
        delete_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        edit_btn = ttk.Button(btn_frame, text="Edit", command=self.edit_selected_annotation)
        edit_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
    
    def open_directory(self):
        directory = filedialog.askdirectory(title="Select Directory with Images")
        
        if directory:
            image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
            has_images = False
            try:
                for item in os.listdir(directory):
                    if item.lower().endswith(image_extensions):
                        has_images = True
                        break
            except Exception as e:
                messagebox.showerror("Error", f"Failed to access directory: {e}")
                return
            
            if not has_images:
                messagebox.showinfo("Info", "The selected directory does not contain any images.")
                return
            
            self.current_directory = directory
            self.image_files = []
            self.current_image_index = -1
            self.current_image_path = None
            self.canvas.delete("all")
            self.refresh_annotations_display()
            self.populate_tree(directory)
    
    def load_video(self):
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not video_path:
            return
        
        output_folder = filedialog.askdirectory(title="Select Output Folder for Frames")
        if not output_folder:
            return
        
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Extracting Frames")
        progress_window.geometry("300x100")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        x = (self.root.winfo_screenwidth() - progress_window.winfo_width()) // 2
        y = (self.root.winfo_screenheight() - progress_window.winfo_height()) // 2
        progress_window.geometry(f"+{x}+{y}")
        
        label = ttk.Label(progress_window, text="Extracting frames, please wait...")
        label.pack(pady=10)
        
        progress_bar = ttk.Progressbar(progress_window, orient=tk.HORIZONTAL, length=200, mode='determinate')
        progress_bar.pack(pady=10)
        
        def extract_frames_task():
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    self.root.after(0, lambda: messagebox.showerror("Error", "Could not open video."))
                    self.root.after(0, progress_window.destroy)
                    return
                
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                os.makedirs(output_folder, exist_ok=True)
                
                frame_interval = fps * self.preferences["frame_extraction_interval"]
                frame_count = 0
                image_count = 0
                
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    if frame_count % frame_interval == 0:
                        image_path = os.path.join(output_folder, f"frame_{image_count:04d}.jpg")
                        cv2.imwrite(image_path, frame)
                        image_count += 1
                    
                    frame_count += 1
                    
                    progress = (frame_count / total_frames) * 100
                    self.root.after(0, lambda p=progress: progress_bar.config(value=p))
                    self.root.update_idletasks()
                
                cap.release()
                
                self.root.after(0, progress_window.destroy)
                self.root.after(0, lambda: self.open_directory_after_extraction(output_folder))
            
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Frame extraction failed: {e}"))
                self.root.after(0, progress_window.destroy)
        
        threading.Thread(target=extract_frames_task, daemon=True).start()
    
    def open_directory_after_extraction(self, directory):
        self.current_directory = directory
        self.image_files = []
        self.current_image_index = -1
        self.current_image_path = None
        self.canvas.delete("all")
        self.refresh_annotations_display()
        self.populate_tree(directory)
    
    def populate_tree(self, directory):
        for item in self.image_tree.get_children():
            self.image_tree.delete(item)
        
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        image_items = []
        try:
            for item in sorted(os.listdir(directory)):
                full_path = os.path.join(directory, item)
                if os.path.isfile(full_path) and item.lower().endswith(image_extensions):
                    image_items.append(item)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to access directory: {e}")
            return
        
        self.image_files = []
        for index, item in enumerate(image_items, start=1):
            full_path = os.path.join(directory, item)
            display_text = f"{index}. {item}"
            self.image_tree.insert("", "end", text=display_text, tags=("image",))
            self.image_files.append(full_path)
        
        if self.image_files:
            self.current_image_index = 0
            self.image_tree.selection_set(self.image_tree.get_children()[0])
            self.load_image(self.image_files[self.current_image_index])
    
    def on_tree_select(self, event):
        selected_item = self.image_tree.selection()
        if not selected_item:
            return
        
        item = selected_item[0]
        tags = self.image_tree.item(item, "tags")
        
        if "image" in tags:
            index = self.image_tree.index(item)
            self.current_image_index = index
            self.load_image(self.image_files[index])
    
    def load_image(self, image_path):
        try:
            self.current_image_path = image_path
            self.current_image = Image.open(self.current_image_path)
            
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img_width, img_height = self.current_image.size
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            self.zoom_factor = min(scale_w, scale_h)
            self.original_image_size = (img_width, img_height)
            
            resized_image = self.current_image.resize(
                (int(img_width * self.zoom_factor), 
                 int(img_height * self.zoom_factor)),
                Image.LANCZOS
            )
            
            self.current_image_tk = ImageTk.PhotoImage(resized_image)
            
            self.current_image_cv = cv2.imread(self.current_image_path)
            
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, image=self.current_image_tk, anchor=tk.NW, tags="base_image")
            
            self.selected_annotation_index = -1
            self.refresh_annotations_display()
            
            self.current_image.close()
            self.current_image = None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
            self.current_image_path = None
            self.canvas.delete("all")
    
    def prev_image(self):
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image(self.image_files[self.current_image_index])
            self.image_tree.selection_set(self.image_tree.get_children()[self.current_image_index])
    
    def next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image(self.image_files[self.current_image_index])
            self.image_tree.selection_set(self.image_tree.get_children()[self.current_image_index])
    
    def save_state(self):
        import copy
        self.undo_stack.append(copy.deepcopy(self.annotations))
        self.redo_stack.clear()
    
    def undo(self, event=None):
        if self.undo_stack:
            self.redo_stack.append(self.annotations)
            self.annotations = self.undo_stack.pop()
            self.refresh_annotations_display()

    def redo(self, event=None):
        if self.redo_stack:
            self.undo_stack.append(self.annotations)
            self.annotations = self.redo_stack.pop()
            self.refresh_annotations_display()
    
    def split_dataset(self):
        from sklearn.model_selection import train_test_split
        train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
        
        img_paths = list(self.annotations.keys())
        train_imgs, temp_imgs = train_test_split(img_paths, train_size=train_ratio)
        val_imgs, test_imgs = train_test_split(temp_imgs, train_size=val_ratio / (val_ratio + test_ratio))
        
        return {"train": train_imgs, "val": val_imgs, "test": test_imgs}
    
    def on_press(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        image_x = canvas_x / self.zoom_factor
        image_y = canvas_y / self.zoom_factor
        
        self.selected_bbox = None
        
        if self.current_image_path in self.annotations:
            for i, ann in enumerate(self.annotations[self.current_image_path]):
                x1, y1, x2, y2 = ann['bbox']
                if x1 <= image_x <= x2 and y1 <= image_y <= y2:
                    self.selected_bbox = i
                    self.selected_annotation_index = i
                    self.refresh_annotations_display()
                    break
        
        self.start_x, self.start_y = image_x, image_y
        
        if self.selected_bbox is None:
            self.drawing = True
            self.temp_rect = self.canvas.create_rectangle(
                image_x * self.zoom_factor, image_y * self.zoom_factor, 
                image_x * self.zoom_factor, image_y * self.zoom_factor,
                outline="red", width=2, 
                tags=("temp_annotation")
            )
    
    def on_drag(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        image_x = canvas_x / self.zoom_factor
        image_y = canvas_y / self.zoom_factor
        
        if self.selected_bbox is not None:
            ann = self.annotations[self.current_image_path][self.selected_bbox]
            x1, y1, x2, y2 = ann['bbox']
            
            dx = image_x - self.start_x
            dy = image_y - self.start_y
            
            ann['bbox'] = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
            
            self.start_x = image_x
            self.start_y = image_y
            
            self.refresh_annotations_display()
            
        elif self.drawing and self.temp_rect:
            self.canvas.coords(
                self.temp_rect, 
                self.start_x * self.zoom_factor, self.start_y * self.zoom_factor, 
                image_x * self.zoom_factor, image_y * self.zoom_factor
            )
    
    def on_release(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        image_x = canvas_x / self.zoom_factor
        image_y = canvas_y / self.zoom_factor
        
        if self.selected_bbox is not None:
            self.save_state()
            self.selected_bbox = None
            
        elif self.drawing and self.temp_rect and self.current_image_path:
            if abs(image_x - self.start_x) > 5 and abs(image_y - self.start_y) > 5:
                class_name = self.class_var.get()
                if not class_name:
                    class_name = self.classes[0] if self.classes else "default"
                
                x1, y1 = min(self.start_x, image_x), min(self.start_y, image_y)
                x2, y2 = max(self.start_x, image_x), max(self.start_y, image_y)
                
                if self.current_image_path not in self.annotations:
                    self.annotations[self.current_image_path] = []
                
                self.annotations[self.current_image_path].append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class': class_name
                })
                
                self.save_state()
                
                self.refresh_annotations_display()
            
            self.canvas.delete("temp_annotation")
        
        self.temp_rect = None
        self.drawing = False
    
    def on_right_click(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        selected = False
        if self.current_image_path in self.annotations:
            for i, ann in enumerate(self.annotations[self.current_image_path]):
                x1, y1, x2, y2 = ann['bbox']
                if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                    self.selected_annotation_index = i
                    selected = True
                    
                    context_menu = tk.Menu(self.root, tearoff=0)
                    context_menu.add_command(label=f"Edit {ann['class']}", command=self.edit_selected_annotation)
                    context_menu.add_command(label="Delete", command=self.delete_selected_annotation)
                    context_menu.post(event.x_root, event.y_root)
                    
                    self.refresh_annotations_display()
                    break
        
        if not selected:
            self.selected_annotation_index = -1
            self.refresh_annotations_display()
    
    def draw_annotation(self, index):
        if self.current_image_path in self.annotations and 0 <= index < len(self.annotations[self.current_image_path]):
            ann = self.annotations[self.current_image_path][index]
            x1, y1, x2, y2 = [coord * self.zoom_factor for coord in ann['bbox']]
            class_name = ann['class']
            color = "green" if index == self.selected_annotation_index else "blue"
            width = 3 if index == self.selected_annotation_index else 2
            
            rect_id = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color, width=width,
                tags=("annotation", f"ann_{index}")
            )
            
            label_bg = self.canvas.create_rectangle(
                x1, y1-20, x1+len(class_name)*8+10, y1,
                fill=color, outline=color,
                tags=("annotation", f"ann_{index}")
            )
            
            label = self.canvas.create_text(
                x1+5, y1-10,
                text=class_name,
                fill="white",
                font=("Arial", 10, "bold"),
                anchor=tk.W,
                tags=("annotation", f"ann_{index}")
            )
            
            return rect_id
    
    def refresh_annotations_display(self):
        for item in self.canvas.find_withtag("annotation"):
            self.canvas.delete(item)
        
        for item in self.annotation_tree.get_children():
            self.annotation_tree.delete(item)
        
        if self.current_image_path in self.annotations:
            for i, ann in enumerate(self.annotations[self.current_image_path]):
                self.draw_annotation(i)
                
                x1, y1, x2, y2 = ann['bbox']
                self.annotation_tree.insert("", tk.END, values=(ann['class'], f"({x1},{y1},{x2},{y2})"))
    
    def on_annotation_select(self, event):
        selected_items = self.annotation_tree.selection()
        if selected_items:
            index = self.annotation_tree.index(selected_items[0])
            self.selected_annotation_index = index
            self.refresh_annotations_display()
    
    def delete_selected_annotation(self, event=None):
        if self.selected_annotation_index >= 0 and self.current_image_path in self.annotations:
            if self.selected_annotation_index < len(self.annotations[self.current_image_path]):
                self.save_state()
                
                del self.annotations[self.current_image_path][self.selected_annotation_index]
                
                self.selected_annotation_index = -1
                
                self.refresh_annotations_display()
    
    def edit_selected_annotation(self):
        if self.selected_annotation_index >= 0 and self.current_image_path in self.annotations:
            ann = self.annotations[self.current_image_path][self.selected_annotation_index]
            
            class_name = simpledialog.askstring("Edit Annotation", "Class name:", initialvalue=ann['class'])
            
            if class_name:
                ann['class'] = class_name
                
                if class_name not in self.classes:
                    self.classes.append(class_name)
                    self.class_combobox['values'] = self.classes
                
                self.refresh_annotations_display()
    
    def add_class(self):
        class_name = simpledialog.askstring("Add Class", "Enter new class name:")
        if class_name and class_name not in self.classes:
            self.classes.append(class_name)
            self.class_combobox['values'] = self.classes
            self.class_var.set(class_name)
    
    def save_annotations(self):
        if not self.annotations:
            messagebox.showinfo("Info", "No annotations to save.")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Save Annotations",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        
        if save_path:
            try:
                save_dir = os.path.dirname(save_path)
                relative_annotations = {}
                
                for img_path, anns in self.annotations.items():
                    try:
                        rel_path = os.path.relpath(img_path, save_dir)
                    except ValueError:
                        rel_path = img_path
                    relative_annotations[rel_path] = anns
                
                with open(save_path, 'w') as f:
                    json.dump({
                        'classes': self.classes,
                        'annotations': relative_annotations
                    }, f, indent=2)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save annotations: {e}")
    
    def load_yolo_model(self):
        try:
            model_path = filedialog.askopenfilename(
                title="Select YOLO Model",
                filetypes=[("YOLO Model", "*.pt *.pth")]
            )
            
            if model_path:
                self.yolo_model = YOLO(model_path)
                self.model_loaded = True
                self.classes = list(self.yolo_model.names.values())
                self.class_combobox['values'] = self.classes
                self.class_var.set(self.classes[0] if self.classes else "")
                self.detect_btn.configure(state=tk.NORMAL)
                self.detect_all_btn.configure(state=tk.NORMAL)
                self.model_btn.configure(text="Model Loaded")
            else:
                return
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")
            self.model_loaded = False
            self.detect_btn.configure(state=tk.DISABLED)
            self.detect_all_btn.configure(state=tk.DISABLED)
            self.model_btn.configure(text="Load YOLO Model")
    
    def run_detection(self):
        if not self.model_loaded or self.current_image_cv is None:
            messagebox.showinfo("Info", "Please load a YOLO model and an image first.")
            return
        
        try:
            results = self.yolo_model(self.current_image_cv)
            
            if hasattr(results, "pandas"):
                detections = results.pandas().xyxy[0]
                class_names = self.yolo_model.names
                detections["name"] = detections["class"].map(class_names)
            else:
                results = results[0]
                class_names = self.yolo_model.model.names
                detections = pd.DataFrame({
                    "xmin": results.boxes.xyxy[:, 0].tolist(),
                    "ymin": results.boxes.xyxy[:, 1].tolist(),
                    "xmax": results.boxes.xyxy[:, 2].tolist(),
                    "ymax": results.boxes.xyxy[:, 3].tolist(),
                    "confidence": results.boxes.conf.tolist(),
                    "class": results.boxes.cls.tolist(),
                })
                detections["name"] = detections["class"].map(class_names)

            if self.current_image_path in self.annotations:
                self.annotations[self.current_image_path] = []
            else:
                self.annotations[self.current_image_path] = []
            
            for _, det in detections.iterrows():
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                class_name = det['name']
                
                if class_name not in self.classes:
                    self.classes.append(class_name)
                    self.class_combobox['values'] = self.classes
                
                self.annotations[self.current_image_path].append({
                    'bbox': [x1, y1, x2, y2],
                    'class': class_name
                })
            
            self.refresh_annotations_display()
            self.class_combobox['values'] = self.classes
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {e}")
    
    def run_detection_on_all(self):
        if not self.model_loaded or not self.image_files:
            messagebox.showinfo("Info", "Please load a YOLO model and some images first.")
            return
        
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Running Detection on All Images")
        progress_window.geometry("300x100")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        x = (self.root.winfo_screenwidth() - progress_window.winfo_width()) // 2
        y = (self.root.winfo_screenheight() - progress_window.winfo_height()) // 2
        progress_window.geometry(f"+{x}+{y}")
        
        label = ttk.Label(progress_window, text="Processing images, please wait...")
        label.pack(pady=10)
        
        progress_bar = ttk.Progressbar(progress_window, orient=tk.HORIZONTAL, length=200, mode='determinate')
        progress_bar.pack(pady=10)
        
        def detection_task():
            try:
                total_images = len(self.image_files)
                processed_images = 0
                
                for img_path in self.image_files:
                    img_cv = cv2.imread(img_path)
                    if img_cv is None:
                        processed_images += 1
                        continue
                    
                    results = self.yolo_model(img_cv)
                    
                    if hasattr(results, "pandas"):
                        detections = results.pandas().xyxy[0]
                        class_names = self.yolo_model.names
                        detections["name"] = detections["class"].map(class_names)
                    else:
                        results = results[0]
                        class_names = self.yolo_model.model.names
                        detections = pd.DataFrame({
                            "xmin": results.boxes.xyxy[:, 0].tolist(),
                            "ymin": results.boxes.xyxy[:, 1].tolist(),
                            "xmax": results.boxes.xyxy[:, 2].tolist(),
                            "ymax": results.boxes.xyxy[:, 3].tolist(),
                            "confidence": results.boxes.conf.tolist(),
                            "class": results.boxes.cls.tolist(),
                        })
                        detections["name"] = detections["class"].map(class_names)
                    
                    self.annotations[img_path] = []
                    for _, det in detections.iterrows():
                        x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                        class_name = det['name']
                        
                        if class_name not in self.classes:
                            self.classes.append(class_name)
                            self.root.after(0, lambda: self.class_combobox.config(values=self.classes))
                        
                        self.annotations[img_path].append({
                            'bbox': [x1, y1, x2, y2],
                            'class': class_name
                        })
                    
                    processed_images += 1
                    progress = (processed_images / total_images) * 100
                    self.root.after(0, lambda p=progress: progress_bar.config(value=p))
                    self.root.update_idletasks()
                
                self.root.after(0, progress_window.destroy)
                self.root.after(0, self.refresh_annotations_display)
            
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Detection on all images failed: {e}"))
                self.root.after(0, progress_window.destroy)
        
        threading.Thread(target=detection_task, daemon=True).start()
    
    def export_annotations(self, format_type):
        if not self.annotations:
            messagebox.showinfo("Info", "No annotations to export.")
            return
        
        export_dir = filedialog.askdirectory(title=f"Select Directory to Export {format_type.upper()} Format")
        if not export_dir:
            return
        
        try:
            if format_type == "coco":
                self.export_coco_format(export_dir)
            elif format_type == "yolo":
                self.export_yolo_format(export_dir)
            elif format_type == "voc":
                self.export_voc_format(export_dir)
            
            messagebox.showinfo("Success", f"Annotations exported in {format_type.upper()} format.")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def export_coco_format(self, export_dir):
        coco_data = {
            "info": {
                "description": "Dataset exported from Image Annotation Tool",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        for i, class_name in enumerate(self.classes, 1):
            coco_data["categories"].append({
                "id": i,
                "name": class_name,
                "supercategory": "none"
            })
        
        image_id = 1
        annotation_id = 1
        
        for img_path, annotations in self.annotations.items():
            img = Image.open(img_path)
            width, height = img.size
            
            img_filename = os.path.basename(img_path)
            coco_data["images"].append({
                "id": image_id,
                "file_name": img_filename,
                "width": width,
                "height": height,
                "license": 1,
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            for ann in annotations:
                bbox = ann["bbox"]
                coco_bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1]
                ]
                
                category_id = self.classes.index(ann["class"]) + 1
                
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": coco_bbox,
                    "area": coco_bbox[2] * coco_bbox[3],
                    "segmentation": [],
                    "iscrowd": 0
                })
                
                annotation_id += 1
            
            image_id += 1
        
        with open(os.path.join(export_dir, "annotations.json"), 'w') as f:
            json.dump(coco_data, f, indent=2)
    
    def export_yolo_format(self, export_dir):
        labels_dir = os.path.join(export_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        
        images_dir = os.path.join(export_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        with open(os.path.join(export_dir, "classes.txt"), 'w') as f:
            for class_name in self.classes:
                f.write(f"{class_name}\n")
        
        for img_path, annotations in self.annotations.items():
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            img_filename = os.path.basename(img_path)
            label_filename = os.path.splitext(img_filename)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_filename)
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    bbox = ann["bbox"]
                    class_id = self.classes.index(ann["class"])
                    
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) / 2 / img_width
                    center_y = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")
            
            img_dest = os.path.join(images_dir, img_filename)
            img.save(img_dest)
    
    def export_voc_format(self, export_dir):
        annotations_dir = os.path.join(export_dir, "Annotations")
        os.makedirs(annotations_dir, exist_ok=True)
        
        images_dir = os.path.join(export_dir, "JPEGImages")
        os.makedirs(images_dir, exist_ok=True)
        
        imagesets_dir = os.path.join(export_dir, "ImageSets", "Main")
        os.makedirs(imagesets_dir, exist_ok=True)
        
        with open(os.path.join(export_dir, "classes.txt"), 'w') as f:
            for class_name in self.classes:
                f.write(f"{class_name}\n")
        
        image_filenames = []
        
        for img_path, annotations in self.annotations.items():
            if not annotations:
                continue
                
            img = Image.open(img_path)
            img_width, img_height = img.size
            img_filename = os.path.basename(img_path)
            img_name = os.path.splitext(img_filename)[0]
            image_filenames.append(img_name)
            
            xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
                <annotation>
                    <folder>JPEGImages</folder>
                    <filename>{img_filename}</filename>
                    <path>{img_path}</path>
                    <source>
                        <database>Unknown</database>
                    </source>
                    <size>
                        <width>{img_width}</width>
                        <height>{img_height}</height>
                        <depth>3</depth>
                    </size>
                    <segmented>0</segmented>
            """
                        
            for ann in annotations:
                bbox = ann["bbox"]
                class_name = ann["class"]
                x1, y1, x2, y2 = bbox
                
                xml_content += f"""    <object>
                        <name>{class_name}</name>
                        <pose>Unspecified</pose>
                        <truncated>0</truncated>
                        <difficult>0</difficult>
                        <bndbox>
                            <xmin>{x1}</xmin>
                            <ymin>{y1}</ymin>
                            <xmax>{x2}</xmax>
                            <ymax>{y2}</ymax>
                        </bndbox>
                    </object>
                """
            
            xml_content += "</annotation>"
            
            xml_path = os.path.join(annotations_dir, img_name + ".xml")
            with open(xml_path, 'w') as f:
                f.write(xml_content)
            
            img_dest = os.path.join(images_dir, img_filename)
            img.save(img_dest)
        
        with open(os.path.join(imagesets_dir, "trainval.txt"), 'w') as f:
            for img_name in image_filenames:
                f.write(f"{img_name}\n")
    
    def delete_current_image(self):
        if self.image_files and 0 <= self.current_image_index < len(self.image_files):
            deleted_path = self.image_files.pop(self.current_image_index)
            
            if deleted_path in self.annotations:
                del self.annotations[deleted_path]
            
            if self.current_image_index >= len(self.image_files):
                self.current_image_index = len(self.image_files) - 1
            
            self.populate_tree(self.current_directory)
            
            if self.image_files:
                self.current_image_index = min(self.current_image_index, len(self.image_files) - 1)
                self.load_image(self.image_files[self.current_image_index])
                self.image_tree.selection_set(self.image_tree.get_children()[self.current_image_index])
            else:
                self.current_image_path = None
                self.canvas.delete("all")
                self.refresh_annotations_display()
    
    def zoom(self, event):
        if event.delta > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor /= 1.1
        
        self.zoom_factor = max(0.1, min(self.zoom_factor, 5.0))
        
        self.redraw_image()
    
    def redraw_image(self):
        if self.current_image_path:
            try:
                image = Image.open(self.current_image_path)
                resized_image = image.resize(
                    (int(image.width * self.zoom_factor), 
                     int(image.height * self.zoom_factor)),
                    Image.LANCZOS
                )
                self.current_image_tk = ImageTk.PhotoImage(resized_image)
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, image=self.current_image_tk, anchor=tk.NW, tags="base_image")
                self.refresh_annotations_display()
                image.close()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to redraw image: {e}")

def main():
    root = tk.Tk()
    app = ImageAnnotationTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()