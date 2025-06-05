"""
Core application logic for the Image Annotation Tool with YOLO Integration.

This module defines the ImageAnnotationTool class, which manages the main window,
coordinates between UI components, and handles high-level application logic.
"""

import os
import tkinter as tk
from tkinter import messagebox
from ui_components import create_menu, create_toolbar, create_left_sidebar, create_main_frame, create_right_sidebar
from annotation_utils import (
    on_press, on_drag, on_release, draw_annotation, refresh_annotations_display,
    on_annotation_select, delete_selected_annotation, edit_selected_annotation, add_class
)
from yolo_utils import load_default_model, load_yolo_model, run_detection, run_detection_on_all
from export_utils import export_coco_format, export_yolo_format, export_voc_format  # Updated imports
from utils import load_preferences, save_preferences, split_dataset

class ImageAnnotationTool:
    def __init__(self, root):
        """
        Initialize the ImageAnnotationTool with the main window and its components.
        
        Args:
            root (tk.Tk): The main Tkinter window.
        """
        self.root = root
        self.root.title("Image Annotation Tool with YOLO Integration")
        self.root.geometry("1200x800")
        
        # Initialize core variables
        self.current_image_path = None
        self.current_image = None
        self.current_image_cv = None
        self.image_files = []  # List of active image file paths
        self.current_image_index = -1
        self.annotations = {}  # Dictionary to store annotations {image_path: [annotations]}
        self.classes = []  # List of class names
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
        self.deleted_images = []  # List to track deleted images
        
        # Preferences for frame extraction and dataset splits
        self.preferences_file = "preferences.json"
        self.preferences = {
            "frame_extraction_interval": 10,
            "default_model_path": self.default_model_path,
            "train_split": 70,
            "val_split": 20,
            "test_split": 10
        }
        load_preferences(self)

        # Create the UI layout
        create_menu(self)
        create_toolbar(self)
        create_left_sidebar(self)
        create_main_frame(self)
        create_right_sidebar(self)
        
        # Bind keyboard shortcuts
        self.root.bind("<Delete>", lambda event: delete_selected_annotation(self))
        self.root.bind("<Left>", lambda event: self.prev_image())
        self.root.bind("<Right>", lambda event: self.next_image())

        # Undo/redo stacks
        self.undo_stack = []
        self.redo_stack = []

        # Bind mouse events for annotation interactions
        self.canvas.bind("<ButtonPress-1>", lambda event: on_press(self, event))
        self.canvas.bind("<B1-Motion>", lambda event: on_drag(self, event))
        self.canvas.bind("<ButtonRelease-1>", lambda event: on_release(self, event))
        
        # Bind zoom events
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-4>", self.zoom)
        self.canvas.bind("<Button-5>", self.zoom)

        # Bind undo/redo shortcuts
        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Control-y>", self.redo)

        # Bind window close event to save preferences
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Load the default YOLO model
        load_default_model(self)
    
    def on_closing(self):
        """Handle window closing by saving preferences and destroying the window."""
        save_preferences(self)
        self.root.destroy()
    
    def open_directory(self):
        """Open a directory containing images and populate the Treeview."""
        directory = tk.filedialog.askdirectory(title="Select Directory with Images")
        
        if directory:
            image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
            has_images = False
            try:
                for item in os.listdir(directory):
                    if item.lower().endswith(image_extensions):
                        has_images = True
                        break
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to access directory: {e}")
                return
            
            if not has_images:
                tk.messagebox.showinfo("Info", "The selected directory does not contain any images.")
                return
            
            self.current_directory = directory
            self.image_files = []
            self.current_image_index = -1
            self.current_image_path = None
            self.deleted_images = []  # Reset deleted images list
            self.canvas.delete("all")
            refresh_annotations_display(self)
            self.populate_tree(directory)
    
    def load_video(self):
        """Load a video file, extract frames, and save them to a specified directory."""
        import cv2
        import threading

        video_path = tk.filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not video_path:
            return
        
        output_folder = tk.filedialog.askdirectory(title="Select Output Folder for Frames")
        if not output_folder:
            return
        
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Extracting Frames")
        progress_window.geometry("300x100")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Center the progress window
        x = (self.root.winfo_screenwidth() - progress_window.winfo_width()) // 2
        y = (self.root.winfo_screenheight() - progress_window.winfo_height()) // 2
        progress_window.geometry(f"+{x}+{y}")
        
        label = tk.ttk.Label(progress_window, text="Extracting frames, please wait...")
        label.pack(pady=10)
        
        progress_bar = tk.ttk.Progressbar(progress_window, orient=tk.HORIZONTAL, length=200, mode='determinate')
        progress_bar.pack(pady=10)
        
        def extract_frames_task():
            """Extract frames from the video at specified intervals."""
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    self.root.after(0, lambda: tk.messagebox.showerror("Error", "Could not open video."))
                    self.root.after(0, progress_window.destroy)
                    return
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_seconds = total_frames / fps if fps > 0 else 0
                interval_seconds = self.preferences["frame_extraction_interval"]
                
                os.makedirs(output_folder, exist_ok=True)
                
                frame_count = 0
                image_count = 0
                last_position = 0
                
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    current_position = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Convert to seconds
                    if current_position >= (last_position + interval_seconds) or frame_count == 0:
                        image_path = os.path.join(output_folder, f"frame_{image_count:04d}.jpg")
                        cv2.imwrite(image_path, frame)
                        image_count += 1
                        last_position = current_position
                    
                    frame_count += 1
                    
                    progress = (frame_count / total_frames) * 100
                    self.root.after(0, lambda p=progress: progress_bar.config(value=p))
                    self.root.update_idletasks()
                
                cap.release()
                
                self.root.after(0, progress_window.destroy)
                self.root.after(0, lambda: self.open_directory_after_extraction(output_folder))
            
            except Exception as e:
                self.root.after(0, lambda: tk.messagebox.showerror("Error", f"Frame extraction failed: {e}"))
                self.root.after(0, progress_window.destroy)
        
        threading.Thread(target=extract_frames_task, daemon=True).start()
    
    def open_directory_after_extraction(self, directory):
        """Open the directory where video frames were extracted and populate the Treeview."""
        self.current_directory = directory
        self.image_files = []
        self.current_image_index = -1
        self.current_image_path = None
        self.deleted_images = []  # Reset deleted images list
        self.canvas.delete("all")
        refresh_annotations_display(self)
        self.populate_tree(directory)
    
    def populate_tree(self, directory):
        """
        Populate the Treeview with images from the directory, marking annotated and deleted images.
        
        Args:
            directory (str): Path to the directory containing images.
        """
        current_index = self.current_image_index
        selected_path = self.image_files[current_index] if 0 <= current_index < len(self.image_files) else None
        
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
            tk.messagebox.showerror("Error", f"Failed to access directory: {e}")
            return
        
        # Create a list of all image paths, including deleted ones
        all_image_paths = []
        for item in image_items:
            full_path = os.path.join(directory, item)
            all_image_paths.append(full_path)
        
        # Filter out deleted images for self.image_files
        self.image_files = [path for path in all_image_paths if path not in self.deleted_images]
        
        # Display all images in the Treeview with [A] for annotated and [D] for deleted
        index = 1
        for item in image_items:
            full_path = os.path.join(directory, item)
            has_annotations = full_path in self.annotations and len(self.annotations[full_path]) > 0
            is_deleted = full_path in self.deleted_images
            
            # Build the display text with indicators
            indicators = []
            if has_annotations:
                indicators.append("[A]")
            if is_deleted:
                indicators.append("[D]")
            display_text = f"{index}. {item} {' '.join(indicators)}"
            
            tree_item = self.image_tree.insert("", "end", text=display_text, tags=("image",))
            
            if full_path == selected_path:
                self.image_tree.selection_set(tree_item)
                self.current_image_index = index - 1
            
            if not is_deleted:
                index += 1
        
        if not self.image_files and current_index >= 0:
            self.current_image_index = -1
            self.current_image_path = None
            self.canvas.delete("all")
            refresh_annotations_display(self)
        elif self.image_files and (current_index < 0 or current_index >= len(self.image_files)):
            self.current_image_index = 0
            self.image_tree.selection_set(self.image_tree.get_children()[0])
            self.load_image(self.image_files[self.current_image_index])
    
    def update_treeview_item(self):
        """
        Update the Treeview item for the current image to reflect its annotation and deletion status.
        """
        if self.current_image_path is None or self.current_image_index < 0:
            return
        
        # Find the Treeview item corresponding to the current image
        all_items = self.image_tree.get_children()
        tree_index = -1
        visible_index = 0
        
        for i, item in enumerate(all_items):
            item_text = self.image_tree.item(item, "text")
            if "[D]" not in item_text:
                if visible_index == self.current_image_index:
                    tree_index = i
                    break
                visible_index += 1
        
        if tree_index == -1:
            return
        
        item = all_items[tree_index]
        full_path = self.image_files[self.current_image_index]
        item_name = os.path.basename(full_path)
        
        # Determine annotation and deletion status
        has_annotations = full_path in self.annotations and len(self.annotations[full_path]) > 0
        is_deleted = full_path in self.deleted_images
        
        # Build the updated display text with indicators
        indicators = []
        if has_annotations:
            indicators.append("[A]")
        if is_deleted:
            indicators.append("[D]")
        
        # Preserve the index in the display text (e.g., "1. image.jpg")
        current_text = self.image_tree.item(item, "text")
        index = current_text.split(".")[0]
        display_text = f"{index}. {item_name} {' '.join(indicators)}"
        
        # Update the Treeview item
        self.image_tree.item(item, text=display_text)
    
    def on_tree_select(self, event):
        """Handle selection of an image in the Treeview and load the corresponding image."""
        selected_item = self.image_tree.selection()
        if not selected_item:
            return
        
        item = selected_item[0]
        tags = self.image_tree.item(item, "tags")
        
        if "image" in tags:
            # Calculate the adjusted index considering deleted images
            all_items = self.image_tree.get_children()
            tree_index = self.image_tree.index(item)
            adjusted_index = 0
            for i in range(tree_index):
                item_text = self.image_tree.item(all_items[i], "text")
                if "[D]" not in item_text:  # Count only non-deleted images
                    adjusted_index += 1
            if "[D]" not in self.image_tree.item(item, "text"):
                self.current_image_index = adjusted_index
                self.load_image(self.image_files[self.current_image_index])
    
    def load_image(self, image_path):
        """
        Load and display an image on the canvas, adjusting for the current zoom level.
        
        Args:
            image_path (str): Path to the image file.
        """
        from PIL import Image, ImageTk
        import cv2

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
            refresh_annotations_display(self)
            
            self.current_image.close()
            self.current_image = None
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load image: {e}")
            self.current_image_path = None
            self.canvas.delete("all")
    
    def prev_image(self):
        """Navigate to the previous image in the list, skipping deleted images."""
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image(self.image_files[self.current_image_index])
            # Update Treeview selection, skipping deleted images
            all_items = self.image_tree.get_children()
            visible_index = 0
            for i, item in enumerate(all_items):
                item_text = self.image_tree.item(item, "text")
                if "[D]" not in item_text:
                    if visible_index == self.current_image_index:
                        self.image_tree.selection_set(item)
                        break
                    visible_index += 1
    
    def next_image(self):
        """Navigate to the next image in the list, skipping deleted images."""
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image(self.image_files[self.current_image_index])
            # Update Treeview selection, skipping deleted images
            all_items = self.image_tree.get_children()
            visible_index = 0
            for i, item in enumerate(all_items):
                item_text = self.image_tree.item(item, "text")
                if "[D]" not in item_text:
                    if visible_index == self.current_image_index:
                        self.image_tree.selection_set(item)
                        break
                    visible_index += 1
    
    def save_state(self):
        """Save the current state of annotations for undo functionality."""
        import copy
        self.undo_stack.append(copy.deepcopy(self.annotations))
        self.redo_stack.clear()
    
    def undo(self, event=None):
        """Undo the last annotation change."""
        if self.undo_stack:
            self.redo_stack.append(self.annotations)
            self.annotations = self.undo_stack.pop()
            refresh_annotations_display(self)
            self.populate_tree(self.current_directory)
    
    def redo(self, event=None):
        """Redo the last undone annotation change."""
        if self.redo_stack:
            self.undo_stack.append(self.annotations)
            self.annotations = self.redo_stack.pop()
            refresh_annotations_display(self)
            self.populate_tree(self.current_directory)
    
    def delete_current_image(self):
        """Mark the current image as deleted and update the UI."""
        if self.image_files and 0 <= self.current_image_index < len(self.image_files):
            deleted_path = self.image_files[self.current_image_index]
            
            # Add the image to the deleted list
            if deleted_path not in self.deleted_images:
                self.deleted_images.append(deleted_path)
            
            # Remove the image from the active list
            self.image_files.pop(self.current_image_index)
            
            # Remove associated annotations
            if deleted_path in self.annotations:
                del self.annotations[deleted_path]
            
            if self.current_image_index >= len(self.image_files):
                self.current_image_index = len(self.image_files) - 1
            
            # Update the Treeview to reflect the deletion
            self.populate_tree(self.current_directory)
            
            if self.image_files:
                self.current_image_index = min(self.current_image_index, len(self.image_files) - 1)
                self.load_image(self.image_files[self.current_image_index])
                # Update Treeview selection
                all_items = self.image_tree.get_children()
                visible_index = 0
                for i, item in enumerate(all_items):
                    item_text = self.image_tree.item(item, "text")
                    if "[D]" not in item_text:
                        if visible_index == self.current_image_index:
                            self.image_tree.selection_set(item)
                            break
                        visible_index += 1
            else:
                self.current_image_path = None
                self.canvas.delete("all")
                refresh_annotations_display(self)
    
    def zoom(self, event):
        """
        Zoom in or out on the current image using the mouse wheel.
        
        Args:
            event: Tkinter event object.
        """
        if event.delta > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor /= 1.1
        
        self.zoom_factor = max(0.1, min(self.zoom_factor, 5.0))
        
        self.redraw_image()
    
    def redraw_image(self):
        """Redraw the current image on the canvas with the updated zoom factor."""
        from PIL import Image, ImageTk

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
                refresh_annotations_display(self)
                image.close()
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to redraw image: {e}")

    def export_annotations(self, format_type):
        """
        Export annotations in the specified format (COCO, YOLO, or VOC).
        
        Args:
            format_type (str): The format to export to ('coco', 'yolo', or 'voc').
        """
        if not self.annotations:
            tk.messagebox.showinfo("Info", "No annotations to export.")
            return
        
        export_dir = tk.filedialog.askdirectory(title=f"Select Directory to Export {format_type.upper()} Format")
        if not export_dir:
            return
        
        try:
            if format_type == "coco":
                export_coco_format(self, export_dir)
            elif format_type == "yolo":
                export_yolo_format(self, export_dir)
            elif format_type == "voc":
                export_voc_format(self, export_dir)
            
            tk.messagebox.showinfo("Success", f"Annotations exported in {format_type.upper()} format.")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Export failed: {e}")