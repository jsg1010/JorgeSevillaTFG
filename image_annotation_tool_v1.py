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
        self.annotations = {}  # {image_path: [{'bbox': [x1, y1, x2, y2], 'class': class_name}, ...]}
        self.classes = []  # Default classes
        self.drawing = False
        self.start_x, self.start_y = 0, 0
        self.rect_id = None
        self.temp_rect = None
        self.selected_annotation_index = -1
        self.yolo_model = None
        self.model_loaded = False
        self.original_image_size = None
        self.zoom_factor = 1.0

        # Create the main layout
        self.create_menu()
        self.create_toolbar()
        self.create_main_frame()
        self.create_sidebar()
        self.create_status_bar()
        
        # Bind keyboard shortcuts
        self.root.bind("<Delete>", self.delete_selected_annotation)
        self.root.bind("<Left>", lambda event: self.prev_image())
        self.root.bind("<Right>", lambda event: self.next_image())

        self.undo_stack = []  # Stores previous states for undo
        self.redo_stack = []  # Stores future states for redo

        # Bind mouse events for moving/resizing bboxes
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        self.canvas.bind("<MouseWheel>", self.zoom)  # Windows and MacOS
        self.canvas.bind("<Button-4>", self.zoom)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.zoom)  # Linux scroll down

        # Bind keyboard shortcuts for undo/redo
        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Control-y>", self.redo)
        
    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_command(label="Open Directory", command=self.open_directory)
        file_menu.add_separator()
        file_menu.add_command(label="Save Annotations", command=self.save_annotations)
        file_menu.add_separator()

        # Export submenu
        export_menu = tk.Menu(file_menu, tearoff=0)
        export_menu.add_command(label="COCO Format", command=lambda: self.export_annotations("coco"))
        export_menu.add_command(label="YOLO Format", command=lambda: self.export_annotations("yolo"))
        export_menu.add_command(label="Pascal VOC Format", command=lambda: self.export_annotations("voc"))
        file_menu.add_cascade(label="Export", menu=export_menu)
        
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Model menu
        model_menu = tk.Menu(menubar, tearoff=0)
        model_menu.add_command(label="Load YOLO Model", command=self.load_yolo_model)
        model_menu.add_command(label="Run Detection", command=self.run_detection)
        menubar.add_cascade(label="Model", menu=model_menu)
        
        self.root.config(menu=menubar)
    
    def create_toolbar(self):
        toolbar_frame = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Navigation buttons
        prev_btn = ttk.Button(toolbar_frame, text="Previous", command=self.prev_image)
        prev_btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        next_btn = ttk.Button(toolbar_frame, text="Next", command=self.next_image)
        next_btn.pack(side=tk.LEFT, padx=2, pady=2)

        delete_btn = ttk.Button(toolbar_frame, text="Delete Image", command=self.delete_current_image)
        delete_btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)
        
        # Annotation buttons
        delete_btn = ttk.Button(toolbar_frame, text="Delete Annotation", command=self.delete_selected_annotation)
        delete_btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)
        
        # YOLO buttons
        self.model_btn = ttk.Button(toolbar_frame, text="Load YOLO Model", command=self.load_yolo_model)
        self.model_btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.detect_btn = ttk.Button(toolbar_frame, text="Run Detection", command=self.run_detection, state=tk.DISABLED)
        self.detect_btn.pack(side=tk.LEFT, padx=2, pady=2)
    
    def create_main_frame(self):
        # Main frame containing the canvas for image display
        self.main_frame = tk.Frame(self.root, bd=2, relief=tk.SUNKEN)
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas for displaying the image and annotations
        self.canvas = tk.Canvas(self.main_frame, bg="gray90", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        h_scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Bind canvas events
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonPress-3>", self.on_right_click)  # Right-click for context menu
    
    def create_sidebar(self):
        # Sidebar frame for annotations list and classes
        self.sidebar = tk.Frame(self.root, width=250, bd=2, relief=tk.RAISED)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Classes section
        classes_frame = tk.LabelFrame(self.sidebar, text="Classes", padx=5, pady=5)
        classes_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.class_var = tk.StringVar(value=self.classes[0] if self.classes else "")
        self.class_combobox = ttk.Combobox(classes_frame, textvariable=self.class_var, values=self.classes)
        self.class_combobox.pack(fill=tk.X, padx=5, pady=5)
        
        add_class_btn = ttk.Button(classes_frame, text="Add Class", command=self.add_class)
        add_class_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Annotations list
        annotations_frame = tk.LabelFrame(self.sidebar, text="Annotations", padx=5, pady=5)
        annotations_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a treeview to display annotations
        self.annotation_tree = ttk.Treeview(annotations_frame, columns=("Class", "Coordinates"), show="headings")
        self.annotation_tree.heading("Class", text="Class")
        self.annotation_tree.heading("Coordinates", text="Coordinates")
        self.annotation_tree.column("Class", width=100)
        self.annotation_tree.column("Coordinates", width=120)
        self.annotation_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bind treeview selection event
        self.annotation_tree.bind("<<TreeviewSelect>>", self.on_annotation_select)
        
        # Buttons for annotations
        btn_frame = tk.Frame(annotations_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        delete_btn = ttk.Button(btn_frame, text="Delete", command=self.delete_selected_annotation)
        delete_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        edit_btn = ttk.Button(btn_frame, text="Edit", command=self.edit_selected_annotation)
        edit_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
    
    def create_status_bar(self):
        # Status bar at the bottom
        self.status_bar = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_text = tk.StringVar()
        self.status_text.set("Ready")
        status_label = tk.Label(self.status_bar, textvariable=self.status_text, anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.X, padx=5)
        
        self.coords_text = tk.StringVar()
        coords_label = tk.Label(self.status_bar, textvariable=self.coords_text, anchor=tk.E)
        coords_label.pack(side=tk.RIGHT, padx=5)
        
        self.image_info = tk.StringVar()
        image_info_label = tk.Label(self.status_bar, textvariable=self.image_info, anchor=tk.CENTER)
        image_info_label.pack(side=tk.RIGHT, padx=5)
    
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.image_files = [file_path]
            self.current_image_index = 0
            self.load_current_image()
    
    def open_directory(self):
        directory = filedialog.askdirectory(title="Select Directory with Images")
        
        if directory:
            self.image_files = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                self.image_files.extend(glob.glob(os.path.join(directory, ext)))
            
            if self.image_files:
                self.image_files.sort()
                self.current_image_index = 0
                self.load_current_image()
            else:
                messagebox.showinfo("Info", "No images found in the selected directory.")
    
    def load_current_image(self):
        if 0 <= self.current_image_index < len(self.image_files):
            self.current_image_path = self.image_files[self.current_image_index]
            
            # Load image with PIL
            self.current_image = Image.open(self.current_image_path)
            
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Calculate scaling
            img_width, img_height = self.current_image.size
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            self.zoom_factor = min(scale_w, scale_h)
            self.original_image_size = (img_width, img_height)

            # Resize image to fit canvas
            resized_image = self.current_image.resize(
                (int(img_width * self.zoom_factor), 
                int(img_height * self.zoom_factor)),
                Image.LANCZOS
            )
            
            self.current_image_tk = ImageTk.PhotoImage(resized_image)
            
            # Clear the canvas completely
            self.canvas.delete("all")
            
            # Draw the base image
            self.canvas.create_image(0, 0, image=self.current_image_tk, anchor=tk.NW, tags="base_image")
            
            # Also load with OpenCV for YOLO processing
            self.current_image_cv = cv2.imread(self.current_image_path)
            
            # Update status bar
            self.image_info.set(f"Image {self.current_image_index + 1}/{len(self.image_files)} | {img_width}x{img_height}")
            self.status_text.set(f"Loaded: {os.path.basename(self.current_image_path)}")
            
            # Reset selection state
            self.selected_annotation_index = -1
            
            # Load existing annotations for this image 
            self.refresh_annotations_display()
    
    def prev_image(self):
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
    
    def next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()

    #############################
    def save_state(self):
        """Save the current state for undo functionality."""
        import copy
        self.undo_stack.append(copy.deepcopy(self.annotations))
        self.redo_stack.clear()  # Clear redo stack when making a new change
    
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
        """Splits the dataset into train/val/test sets."""
        from sklearn.model_selection import train_test_split
        train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1  # Default split
        
        img_paths = list(self.annotations.keys())
        train_imgs, temp_imgs = train_test_split(img_paths, train_size=train_ratio)
        val_imgs, test_imgs = train_test_split(temp_imgs, train_size=val_ratio / (val_ratio + test_ratio))
        
        return {"train": train_imgs, "val": val_imgs, "test": test_imgs}

    def on_press(self, event):
        # Convert from window coordinates to canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Convert to image coordinates (accounting for zoom)
        image_x = canvas_x / self.zoom_factor
        image_y = canvas_y / self.zoom_factor
        
        # Reset selected annotation
        self.selected_bbox = None
        
        # Check if we're clicking on an existing bbox
        if self.current_image_path in self.annotations:
            for i, ann in enumerate(self.annotations[self.current_image_path]):
                x1, y1, x2, y2 = ann['bbox']
                if x1 <= image_x <= x2 and y1 <= image_y <= y2:
                    self.selected_bbox = i
                    self.selected_annotation_index = i
                    self.refresh_annotations_display()  # Highlight selected box
                    break
        
        # Start position (in image coordinates)
        self.start_x, self.start_y = image_x, image_y
        
        # If not on an existing bbox, start drawing a new one
        if self.selected_bbox is None:
            self.drawing = True
            # Create temporary rectangle for drawing - using canvas coordinates
            self.temp_rect = self.canvas.create_rectangle(
                image_x * self.zoom_factor, image_y * self.zoom_factor, 
                image_x * self.zoom_factor, image_y * self.zoom_factor,
                outline="red", width=2, 
                tags=("temp_annotation")
            )
        
        # Update coordinates display
        self.coords_text.set(f"({int(image_x)}, {int(image_y)})")

    def on_drag(self, event):
        # Get canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Convert to image coordinates
        image_x = canvas_x / self.zoom_factor
        image_y = canvas_y / self.zoom_factor
        
        if self.selected_bbox is not None:
            # We're moving an existing bbox
            ann = self.annotations[self.current_image_path][self.selected_bbox]
            x1, y1, x2, y2 = ann['bbox']
            
            # Calculate movement delta in image coordinates
            dx = image_x - self.start_x
            dy = image_y - self.start_y
            
            # Update bbox with new position
            ann['bbox'] = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
            
            # Update start position for next drag event
            self.start_x = image_x
            self.start_y = image_y
            
            # Refresh display to show updated position
            self.refresh_annotations_display()
            
        elif self.drawing and self.temp_rect:
            # We're drawing a new bbox - update the rectangle IN CANVAS COORDINATES
            self.canvas.coords(
                self.temp_rect, 
                self.start_x * self.zoom_factor, self.start_y * self.zoom_factor, 
                image_x * self.zoom_factor, image_y * self.zoom_factor
            )
            # Update coordinates display
            self.coords_text.set(f"({int(self.start_x)}, {int(self.start_y)}) to ({int(image_x)}, {int(image_y)})")

    def on_release(self, event):
        # Get canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Convert to image coordinates
        image_x = canvas_x / self.zoom_factor
        image_y = canvas_y / self.zoom_factor
        
        if self.selected_bbox is not None:
            # Finished moving a bbox
            self.save_state()  # Save for undo/redo
            self.selected_bbox = None
            
        elif self.drawing and self.temp_rect and self.current_image_path:
            # Finishing drawing a new bbox
            if abs(image_x - self.start_x) > 5 and abs(image_y - self.start_y) > 5:
                # Get the class for this annotation
                class_name = self.class_var.get()
                if not class_name:
                    class_name = self.classes[0] if self.classes else "default"
                
                # Normalize coordinates: x1, y1, x2, y2
                x1, y1 = min(self.start_x, image_x), min(self.start_y, image_y)
                x2, y2 = max(self.start_x, image_x), max(self.start_y, image_y)
                
                # Add the annotation (in image coordinates)
                if self.current_image_path not in self.annotations:
                    self.annotations[self.current_image_path] = []
                
                self.annotations[self.current_image_path].append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class': class_name
                })
                
                self.save_state()  # Save for undo/redo
                
                # Update the annotations list
                self.refresh_annotations_display()
                self.status_text.set(f"Added {class_name} annotation")
            
            # Always delete the temporary rectangle
            self.canvas.delete("temp_annotation")
        
        self.temp_rect = None
        self.drawing = False

    def delete_selected_annotation(self, event=None):
        if self.selected_annotation_index >= 0 and self.current_image_path in self.annotations:
            if self.selected_annotation_index < len(self.annotations[self.current_image_path]):
                # Save state before modifying
                self.save_state()
                
                # Remove the annotation from the data structure
                del self.annotations[self.current_image_path][self.selected_annotation_index]
                
                # Reset selection
                self.selected_annotation_index = -1
                
                # Refresh display - this will remove the annotation from view
                self.refresh_annotations_display()
                
                self.status_text.set("Annotation deleted")

    def on_right_click(self, event):
        # Check if right-click is on an annotation to select it
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        selected = False
        if self.current_image_path in self.annotations:
            for i, ann in enumerate(self.annotations[self.current_image_path]):
                x1, y1, x2, y2 = ann['bbox']
                if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                    # Select this annotation
                    self.selected_annotation_index = i
                    selected = True
                    
                    # Create context menu
                    context_menu = tk.Menu(self.root, tearoff=0)
                    context_menu.add_command(label=f"Edit {ann['class']}", command=self.edit_selected_annotation)
                    context_menu.add_command(label="Delete", command=self.delete_selected_annotation)
                    context_menu.post(event.x_root, event.y_root)
                    
                    # Highlight the selected annotation
                    self.refresh_annotations_display()
                    break
        
        if not selected:
            self.selected_annotation_index = -1
            self.refresh_annotations_display()
    
    def draw_annotation(self, index):
        if self.current_image_path in self.annotations and 0 <= index < len(self.annotations[self.current_image_path]):
            ann = self.annotations[self.current_image_path][index]
            # x1, y1, x2, y2 = ann['bbox']
            x1, y1, x2, y2 = [coord * self.zoom_factor for coord in ann['bbox']]
            class_name = ann['class']
            color = "green" if index == self.selected_annotation_index else "blue"
            width = 3 if index == self.selected_annotation_index else 2
            
            # Draw the rectangle - note the "annotation" tag
            rect_id = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color, width=width,
                tags=("annotation", f"ann_{index}")  # Both tags - annotation and specific index
            )
            
            # Draw the class label - also tagged as annotation
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
        # Remove all previous annotation objects from canvas
        # Make sure to only delete objects tagged as annotations, not the base image
        for item in self.canvas.find_withtag("annotation"):
            self.canvas.delete(item)
        
        # Clear the treeview
        for item in self.annotation_tree.get_children():
            self.annotation_tree.delete(item)
        
        # Draw all annotations for the current image
        if self.current_image_path in self.annotations:
            for i, ann in enumerate(self.annotations[self.current_image_path]):
                self.draw_annotation(i)
                
                # Add to the treeview
                x1, y1, x2, y2 = ann['bbox']
                self.annotation_tree.insert("", tk.END, values=(ann['class'], f"({x1},{y1},{x2},{y2})"))
                
    def on_annotation_select(self, event):
        selected_items = self.annotation_tree.selection()
        if selected_items:
            # Get the index of the selected item
            index = self.annotation_tree.index(selected_items[0])
            self.selected_annotation_index = index
            self.refresh_annotations_display()
    
    def delete_selected_annotation(self, event=None):
        if self.selected_annotation_index >= 0 and self.current_image_path in self.annotations:
            if self.selected_annotation_index < len(self.annotations[self.current_image_path]):
                del self.annotations[self.current_image_path][self.selected_annotation_index]
                self.selected_annotation_index = -1
                self.refresh_annotations_display()
                self.status_text.set("Annotation deleted")
    
    def edit_selected_annotation(self):
        if self.selected_annotation_index >= 0 and self.current_image_path in self.annotations:
            ann = self.annotations[self.current_image_path][self.selected_annotation_index]
            
            # Create a dialog to edit the class
            class_name = simpledialog.askstring("Edit Annotation", "Class name:", initialvalue=ann['class'])
            
            if class_name:
                ann['class'] = class_name
                
                if class_name not in self.classes:
                    self.classes.append(class_name)
                    self.class_combobox['values'] = self.classes
                
                self.refresh_annotations_display()
                self.status_text.set(f"Updated annotation to {class_name}")
    
    def add_class(self):
        class_name = simpledialog.askstring("Add Class", "Enter new class name:")
        if class_name and class_name not in self.classes:
            self.classes.append(class_name)
            self.class_combobox['values'] = self.classes
            self.class_var.set(class_name)
            self.status_text.set(f"Added class: {class_name}")
    
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
                # Convert image paths to relative paths if possible
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
                
                self.status_text.set(f"Annotations saved to {os.path.basename(save_path)}")
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
            else:
                # If no model is provided, use pretrained best
                self.yolo_model = YOLO("train/crack_detection/yolo11l_exp1/weights/best.pt")
            
            self.model_loaded = True
            
            # Update classes from the model
            self.classes = list(self.yolo_model.names.values())
            # Update UI
            self.class_combobox['values'] = self.classes  # Update dropdown values
            self.class_var.set(self.classes[0] if self.classes else "")  # Set first class as default

            self.detect_btn.configure(state=tk.NORMAL)
            self.model_btn.configure(text="Model Loaded")
            self.status_text.set("YOLO model loaded successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")
    
    def run_detection(self):
        if not self.model_loaded or not self.current_image_cv is not None:
            messagebox.showinfo("Info", "Please load a YOLO model and an image first.")
            return
        
        try:
            # Run inference
            results = self.yolo_model(self.current_image_cv)
            
            # Check if results are in pandas format (YOLOv5)
            if hasattr(results, "pandas"):
                detections = results.pandas().xyxy[0]  # YOLOv5 output
                # *!* Only required if want to change tags 
                class_names = self.yolo_model.names
                detections["name"] = detections["class"].map(class_names)  # Map class index to name

            else:
                results = results[0]  # YOLOv8 returns a list
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
            
            print(detections)

            # Clear existing annotations for this image
            if self.current_image_path in self.annotations:
                self.annotations[self.current_image_path] = []
            else:
                self.annotations[self.current_image_path] = []
            
            # Add new annotations from YOLO
            for _, det in detections.iterrows():
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                class_name = det['name']
                
                # Add class if not already in list
                if class_name not in self.classes:
                    self.classes.append(class_name)
                    self.class_combobox['values'] = self.classes
                
                # Add the annotation
                self.annotations[self.current_image_path].append({
                    'bbox': [x1, y1, x2, y2],
                    'class': class_name
                })
            
            # Update display
            self.refresh_annotations_display()
            self.status_text.set(f"YOLO detection complete. Found {len(detections)} objects.")
            
            # Update class combobox
            self.class_combobox['values'] = self.classes
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {e}")
    
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
            
            self.status_text.set(f"Annotations exported in {format_type.upper()} format to {export_dir}")
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
        
        # Add categories
        for i, class_name in enumerate(self.classes, 1):
            coco_data["categories"].append({
                "id": i,
                "name": class_name,
                "supercategory": "none"
            })
        
        # Add images and annotations
        image_id = 1
        annotation_id = 1
        
        for img_path, annotations in self.annotations.items():
            # Get image dimensions
            img = Image.open(img_path)
            width, height = img.size
            
            # Add image
            img_filename = os.path.basename(img_path)
            coco_data["images"].append({
                "id": image_id,
                "file_name": img_filename,
                "width": width,
                "height": height,
                "license": 1,
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Add annotations
            for ann in annotations:
                bbox = ann["bbox"]
                # COCO format: [x, y, width, height]
                coco_bbox = [
                    bbox[0],  # x
                    bbox[1],  # y
                    bbox[2] - bbox[0],  # width
                    bbox[3] - bbox[1]   # height
                ]
                
                # Get category id
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
        
        # Save COCO JSON file
        with open(os.path.join(export_dir, "annotations.json"), 'w') as f:
            json.dump(coco_data, f, indent=2)
    
    def export_yolo_format(self, export_dir):
        # Create labels directory
        labels_dir = os.path.join(export_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        
        # Create images directory
        images_dir = os.path.join(export_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Save class names
        with open(os.path.join(export_dir, "classes.txt"), 'w') as f:
            for class_name in self.classes:
                f.write(f"{class_name}\n")
        
        # Process each image
        for img_path, annotations in self.annotations.items():
            # Get image dimensions
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # Prepare YOLO format labels file
            img_filename = os.path.basename(img_path)
            label_filename = os.path.splitext(img_filename)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_filename)
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    bbox = ann["bbox"]
                    class_id = self.classes.index(ann["class"])
                    
                    # Convert to YOLO format: class_id, center_x, center_y, width, height (normalized)
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) / 2 / img_width
                    center_y = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")
            
            # Copy the image to the images directory
            img_dest = os.path.join(images_dir, img_filename)
            img.save(img_dest)
    
    def export_voc_format(self, export_dir):
        # Create annotations directory
        annotations_dir = os.path.join(export_dir, "Annotations")
        os.makedirs(annotations_dir, exist_ok=True)
        
        # Create images directory
        images_dir = os.path.join(export_dir, "JPEGImages")
        os.makedirs(images_dir, exist_ok=True)
        
        # Create ImageSets directory
        imagesets_dir = os.path.join(export_dir, "ImageSets", "Main")
        os.makedirs(imagesets_dir, exist_ok=True)
        
        # Save class names
        with open(os.path.join(export_dir, "classes.txt"), 'w') as f:
            for class_name in self.classes:
                f.write(f"{class_name}\n")
        
        # List of image filenames for ImageSets
        image_filenames = []
        
        # Process each image
        for img_path, annotations in self.annotations.items():
            if not annotations:
                continue
                
            # Get image dimensions and filename
            img = Image.open(img_path)
            img_width, img_height = img.size
            img_filename = os.path.basename(img_path)
            img_name = os.path.splitext(img_filename)[0]
            image_filenames.append(img_name)
            
            # Create XML file
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
            
            # Save XML file
            xml_path = os.path.join(annotations_dir, img_name + ".xml")
            with open(xml_path, 'w') as f:
                f.write(xml_content)
            
            # Copy the image to the JPEGImages directory
            img_dest = os.path.join(images_dir, img_filename)
            img.save(img_dest)
        
        # Create ImageSets files
        with open(os.path.join(imagesets_dir, "trainval.txt"), 'w') as f:
            for img_name in image_filenames:
                f.write(f"{img_name}\n")

    def delete_current_image(self):
        if self.image_files and 0 <= self.current_image_index < len(self.image_files):
            # Remove the current image from the list
            deleted_path = self.image_files.pop(self.current_image_index)
            
            # Remove annotations for this image if they exist
            if deleted_path in self.annotations:
                del self.annotations[deleted_path]
            
            # Adjust current index if needed
            if self.current_image_index >= len(self.image_files):
                self.current_image_index = len(self.image_files) - 1
            
            # Load next or previous image
            if self.image_files:
                self.load_current_image()
            else:
                # Clear canvas if no images left
                self.canvas.delete("all")
                self.status_text.set("No images remaining")

    def zoom(self, event):
        # Determine zoom direction
        if event.delta > 0:
            self.zoom_factor *= 1.1  # Zoom in
        else:
            self.zoom_factor /= 1.1  # Zoom out
        
        # Limit zoom range
        self.zoom_factor = max(0.1, min(self.zoom_factor, 5.0))
        
        # Redraw the image with new zoom
        self.redraw_image()
    
    def redraw_image(self):
        if self.current_image:
            # Clear the canvas
            self.canvas.delete("all")
            
            # Resize image
            resized_image = self.current_image.resize(
                (int(self.current_image.width * self.zoom_factor), 
                int(self.current_image.height * self.zoom_factor)),
                Image.LANCZOS
            )
            
            # Convert to PhotoImage
            self.current_image_tk = ImageTk.PhotoImage(resized_image)
            
            # Redraw image
            self.canvas.create_image(0, 0, image=self.current_image_tk, anchor=tk.NW, tags="base_image")
            
            # Redraw annotations
            self.refresh_annotations_display()

def main():
    root = tk.Tk()
    app = ImageAnnotationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()