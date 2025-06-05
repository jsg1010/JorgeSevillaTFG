"""
UI component creation for the Image Annotation Tool.

This module contains functions to create the menu, toolbar, sidebars, and main frame of the application.
"""

import tkinter as tk
from tkinter import ttk
from annotation_utils import add_class, on_annotation_select, delete_selected_annotation, edit_selected_annotation
import tkinter.filedialog

from yolo_utils import load_yolo_model, run_detection, run_detection_on_all  # Added explicit import for filedialog

def create_menu(app):
    """Create the top menu bar with File and Preferences options."""
    menubar = tk.Menu(app.root)
    
    # File menu
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Open Directory", command=app.open_directory)
    file_menu.add_command(label="Load Video", command=app.load_video)
    file_menu.add_separator()

    # Export submenu
    export_menu = tk.Menu(file_menu, tearoff=0)
    export_menu.add_command(label="COCO Format", command=lambda: app.export_annotations("coco"))
    export_menu.add_command(label="YOLO Format", command=lambda: app.export_annotations("yolo"))
    export_menu.add_command(label="Pascal VOC Format", command=lambda: app.export_annotations("voc"))
    file_menu.add_cascade(label="Export", menu=export_menu)
    
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=app.root.quit)
    menubar.add_cascade(label="File", menu=file_menu)
    
    # Preferences menu
    preferences_menu = tk.Menu(menubar, tearoff=0)
    preferences_menu.add_command(label="Settings", command=lambda: open_preferences(app))
    menubar.add_cascade(label="Preferences", menu=preferences_menu)
    
    app.root.config(menu=menubar)

def open_preferences(app):
    """Open a preferences window to configure settings like frame extraction interval and dataset splits."""
    pref_window = tk.Toplevel(app.root)
    pref_window.title("Preferences")
    pref_window.geometry("400x300")
    pref_window.transient(app.root)
    pref_window.grab_set()

    # Center the window
    pref_window.update_idletasks()
    x = (app.root.winfo_screenwidth() - pref_window.winfo_width()) // 2
    y = (app.root.winfo_screenheight() - pref_window.winfo_height()) // 2
    pref_window.geometry(f"+{x}+{y}")

    # Frame extraction interval
    tk.Label(pref_window, text="Frame Extraction Interval (seconds):").pack(pady=5)
    interval_var = tk.IntVar(value=app.preferences["frame_extraction_interval"])
    interval_entry = ttk.Entry(pref_window, textvariable=interval_var)
    interval_entry.pack(pady=5)

    # Default YOLO model path
    tk.Label(pref_window, text="Default YOLO Model Path:").pack(pady=5)
    model_frame = tk.Frame(pref_window)
    model_frame.pack(pady=5, fill=tk.X, padx=5)
    model_var = tk.StringVar(value=app.preferences["default_model_path"])
    model_entry = ttk.Entry(model_frame, textvariable=model_var)
    model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    ttk.Button(model_frame, text="Browse", command=lambda: select_model_path(model_var)).pack(side=tk.LEFT)

    # Dataset split percentages
    tk.Label(pref_window, text="Dataset Split Percentages (must sum to 100):").pack(pady=5)
    split_frame = tk.Frame(pref_window)
    split_frame.pack(pady=5, fill=tk.X, padx=5)

    tk.Label(split_frame, text="Train (%):").pack(side=tk.LEFT)
    train_var = tk.IntVar(value=app.preferences["train_split"])
    train_entry = ttk.Entry(split_frame, textvariable=train_var, width=5)
    train_entry.pack(side=tk.LEFT, padx=5)

    tk.Label(split_frame, text="Val (%):").pack(side=tk.LEFT)
    val_var = tk.IntVar(value=app.preferences["val_split"])
    val_entry = ttk.Entry(split_frame, textvariable=val_var, width=5)
    val_entry.pack(side=tk.LEFT, padx=5)

    tk.Label(split_frame, text="Test (%):").pack(side=tk.LEFT)
    test_var = tk.IntVar(value=app.preferences["test_split"])
    test_entry = ttk.Entry(split_frame, textvariable=test_var, width=5)
    test_entry.pack(side=tk.LEFT, padx=5)

    # Save and Cancel buttons
    btn_frame = tk.Frame(pref_window)
    btn_frame.pack(pady=10)
    ttk.Button(btn_frame, text="Save", command=lambda: save_preferences_from_window(
        app, pref_window, interval_var.get(), model_var.get(), train_var.get(), val_var.get(), test_var.get())).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Cancel", command=pref_window.destroy).pack(side=tk.LEFT, padx=5)

def select_model_path(model_var):
    """
    Open a file dialog to select a YOLO model file and update the model path variable.
    
    Args:
        model_var (tk.StringVar): Variable to store the selected model path.
    """
    model_path = tkinter.filedialog.askopenfilename(
        title="Select Default YOLO Model",
        filetypes=[("YOLO Model", "*.pt *.pth")]
    )
    if model_path:
        model_var.set(model_path)

def save_preferences_from_window(app, window, interval, model_path, train_split, val_split, test_split):
    """
    Save preferences from the preferences window after validating inputs.
    
    Args:
        app: The ImageAnnotationTool instance.
        window (tk.Toplevel): The preferences window.
        interval (int): Frame extraction interval in seconds.
        model_path (str): Path to the default YOLO model.
        train_split (int): Percentage for training split.
        val_split (int): Percentage for validation split.
        test_split (int): Percentage for test split.
    """
    import os
    from yolo_utils import load_default_model

    try:
        if interval <= 0:
            tk.messagebox.showerror("Error", "Frame extraction interval must be greater than 0.")
            return
        
        if model_path and not os.path.exists(model_path):
            tk.messagebox.showerror("Error", "Selected model file does not exist.")
            return
        
        total_split = train_split + val_split + test_split
        if total_split != 100:
            tk.messagebox.showerror("Error", f"Dataset split percentages must sum to 100. Current sum: {total_split}")
            return
        if train_split < 0 or val_split < 0 or test_split < 0:
            tk.messagebox.showerror("Error", "Dataset split percentages cannot be negative.")
            return

        app.preferences["frame_extraction_interval"] = interval
        app.preferences["default_model_path"] = model_path
        app.preferences["train_split"] = train_split
        app.preferences["val_split"] = val_split
        app.preferences["test_split"] = test_split
        app.default_model_path = model_path
        
        if model_path:
            load_default_model(app)
        
        window.destroy()
        tk.messagebox.showinfo("Success", "Preferences saved successfully.")
    except Exception as e:
        tk.messagebox.showerror("Error", f"Failed to save preferences: {e}")

def create_toolbar(app):
    """Create the toolbar with buttons for navigation and YOLO detection."""
    toolbar_frame = tk.Frame(app.root, bd=1, relief=tk.RAISED)
    toolbar_frame.pack(side=tk.TOP, fill=tk.X)
    
    prev_btn = ttk.Button(toolbar_frame, text="Previous", command=app.prev_image)
    prev_btn.pack(side=tk.LEFT, padx=2, pady=2)
    
    next_btn = ttk.Button(toolbar_frame, text="Next", command=app.next_image)
    next_btn.pack(side=tk.LEFT, padx=2, pady=2)

    delete_btn = ttk.Button(toolbar_frame, text="Delete Image", command=app.delete_current_image)
    delete_btn.pack(side=tk.LEFT, padx=2, pady=2)
    
    ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)
    
    app.model_btn = ttk.Button(toolbar_frame, text="Load YOLO Model", command=lambda: load_yolo_model(app))
    app.model_btn.pack(side=tk.LEFT, padx=2, pady=2)
    
    app.detect_btn = ttk.Button(toolbar_frame, text="Run Detection", command=lambda: run_detection(app), state=tk.DISABLED)
    app.detect_btn.pack(side=tk.LEFT, padx=2, pady=2)
    
    app.detect_all_btn = ttk.Button(toolbar_frame, text="Run Detection on ALL", command=lambda: run_detection_on_all(app), state=tk.DISABLED)
    app.detect_all_btn.pack(side=tk.LEFT, padx=2, pady=2)

def create_left_sidebar(app):
    """Create the left sidebar with a Treeview to display the list of images."""
    app.left_sidebar = tk.Frame(app.root, width=250, bd=2, relief=tk.RAISED)
    app.left_sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
    
    images_frame = tk.LabelFrame(app.left_sidebar, text="Images", padx=5, pady=5)
    images_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    app.image_tree = ttk.Treeview(images_frame, show="tree")
    app.image_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    app.image_tree.bind("<<TreeviewSelect>>", app.on_tree_select)

def create_main_frame(app):
    """Create the main frame with a canvas for displaying and annotating images."""
    app.main_frame = tk.Frame(app.root, bd=2, relief=tk.SUNKEN)
    app.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    app.canvas = tk.Canvas(app.main_frame, bg="gray90", cursor="crosshair")
    app.canvas.pack(fill=tk.BOTH, expand=True)
    
    h_scrollbar = ttk.Scrollbar(app.main_frame, orient=tk.HORIZONTAL, command=app.canvas.xview)
    h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    
    v_scrollbar = ttk.Scrollbar(app.main_frame, orient=tk.VERTICAL, command=app.canvas.yview)
    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    app.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

def create_right_sidebar(app):
    """Create the right sidebar with class selection and annotation list."""
    app.sidebar = tk.Frame(app.root, width=250, bd=2, relief=tk.RAISED)
    app.sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
    
    # Classes section
    classes_frame = tk.LabelFrame(app.sidebar, text="Classes", padx=5, pady=5)
    classes_frame.pack(fill=tk.X, padx=5, pady=5)
    
    app.class_var = tk.StringVar(value=app.classes[0] if app.classes else "")
    app.class_combobox = ttk.Combobox(classes_frame, textvariable=app.class_var, values=app.classes)
    app.class_combobox.pack(fill=tk.X, padx=5, pady=5)
    
    add_class_btn = ttk.Button(classes_frame, text="Add Class", command=lambda: add_class(app))
    add_class_btn.pack(fill=tk.X, padx=5, pady=5)
    
    # Annotations section
    annotations_frame = tk.LabelFrame(app.sidebar, text="Annotations", padx=5, pady=5)
    annotations_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    app.annotation_tree = ttk.Treeview(annotations_frame, columns=("Class", "Coordinates"), show="headings")
    app.annotation_tree.heading("Class", text="Class")
    app.annotation_tree.heading("Coordinates", text="Coordinates")
    app.annotation_tree.column("Class", width=100)
    app.annotation_tree.column("Coordinates", width=120)
    app.annotation_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    app.annotation_tree.bind("<<TreeviewSelect>>", lambda event: on_annotation_select(app, event))
    
    btn_frame = tk.Frame(annotations_frame)
    btn_frame.pack(fill=tk.X, padx=5, pady=5)
    
    delete_btn = ttk.Button(btn_frame, text="Delete", command=lambda: delete_selected_annotation(app))
    delete_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
    
    edit_btn = ttk.Button(btn_frame, text="Edit", command=lambda: edit_selected_annotation(app))
    edit_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)