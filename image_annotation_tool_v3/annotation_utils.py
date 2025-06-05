"""
Utilities for handling annotations in the Image Annotation Tool.

This module contains functions for drawing, editing, deleting, and managing annotations.
"""
import tkinter as tk

def on_press(app, event):
    """
    Handle mouse press event to start drawing a new bounding box or select an existing one.
    
    Args:
        app: The ImageAnnotationTool instance.
        event: Tkinter event object.
    """
    canvas_x = app.canvas.canvasx(event.x)
    canvas_y = app.canvas.canvasy(event.y)
    
    image_x = canvas_x / app.zoom_factor
    image_y = canvas_y / app.zoom_factor
    
    app.selected_bbox = None
    
    if app.current_image_path in app.annotations:
        for i, ann in enumerate(app.annotations[app.current_image_path]):
            x1, y1, x2, y2 = ann['bbox']
            if x1 <= image_x <= x2 and y1 <= image_y <= y2:
                app.selected_bbox = i
                app.selected_annotation_index = i
                refresh_annotations_display(app)
                break
    
    app.start_x, app.start_y = image_x, image_y
    
    if app.selected_bbox is None:
        app.drawing = True
        app.temp_rect = app.canvas.create_rectangle(
            image_x * app.zoom_factor, image_y * app.zoom_factor, 
            image_x * app.zoom_factor, image_y * app.zoom_factor,
            outline="red", width=2, 
            tags=("temp_annotation")
        )

def on_drag(app, event):
    """
    Handle mouse drag event to resize a bounding box or move an existing one.
    
    Args:
        app: The ImageAnnotationTool instance.
        event: Tkinter event object.
    """
    canvas_x = app.canvas.canvasx(event.x)
    canvas_y = app.canvas.canvasy(event.y)
    
    image_x = canvas_x / app.zoom_factor
    image_y = canvas_y / app.zoom_factor
    
    if app.selected_bbox is not None:
        ann = app.annotations[app.current_image_path][app.selected_bbox]
        x1, y1, x2, y2 = ann['bbox']
        
        dx = image_x - app.start_x
        dy = image_y - app.start_y
        
        ann['bbox'] = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
        
        app.start_x = image_x
        app.start_y = image_y
        
        refresh_annotations_display(app)
        
    elif app.drawing and app.temp_rect:
        app.canvas.coords(
            app.temp_rect, 
            app.start_x * app.zoom_factor, app.start_y * app.zoom_factor, 
            image_x * app.zoom_factor, image_y * app.zoom_factor
        )

def on_release(app, event):
    """
    Handle mouse release event to finalize drawing a new bounding box.
    
    Args:
        app: The ImageAnnotationTool instance.
        event: Tkinter event object.
    """
    canvas_x = app.canvas.canvasx(event.x)
    canvas_y = app.canvas.canvasy(event.y)
    
    image_x = canvas_x / app.zoom_factor
    image_y = canvas_y / app.zoom_factor
    
    if app.selected_bbox is not None:
        app.save_state()
        app.selected_bbox = None
        
    elif app.drawing and app.temp_rect and app.current_image_path:
        if abs(image_x - app.start_x) > 5 and abs(image_y - app.start_y) > 5:
            class_name = app.class_var.get()
            if not class_name:
                class_name = app.classes[0] if app.classes else "default"
            
            x1, y1 = min(app.start_x, image_x), min(app.start_y, image_y)
            x2, y2 = max(app.start_x, image_x), max(app.start_y, image_y)
            
            if app.current_image_path not in app.annotations:
                app.annotations[app.current_image_path] = []
            
            app.annotations[app.current_image_path].append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'class': class_name
            })
            
            app.save_state()
            
            refresh_annotations_display(app)
            app.update_treeview_item()  # Update [A] indicator for the current image
        
        app.canvas.delete("temp_annotation")
    
    app.temp_rect = None
    app.drawing = False

def draw_annotation(app, index):
    """
    Draw a bounding box and its label on the canvas for a given annotation.
    
    Args:
        app: The ImageAnnotationTool instance.
        index (int): Index of the annotation in the current image's annotations list.
    
    Returns:
        int: ID of the drawn rectangle on the canvas.
    """
    if app.current_image_path in app.annotations and 0 <= index < len(app.annotations[app.current_image_path]):
        ann = app.annotations[app.current_image_path][index]
        x1, y1, x2, y2 = [coord * app.zoom_factor for coord in ann['bbox']]
        class_name = ann['class']
        color = "green" if index == app.selected_annotation_index else "blue"
        width = 3 if index == app.selected_annotation_index else 2
        
        rect_id = app.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline=color, width=width,
            tags=("annotation", f"ann_{index}")
        )
        
        label_bg = app.canvas.create_rectangle(
            x1, y1-20, x1+len(class_name)*8+10, y1,
            fill=color, outline=color,
            tags=("annotation", f"ann_{index}")
        )
        
        label = app.canvas.create_text(
            x1+5, y1-10,
            text=class_name,
            fill="white",
            font=("Arial", 10, "bold"),
            anchor=tk.W,
            tags=("annotation", f"ann_{index}")
        )
        
        return rect_id

def refresh_annotations_display(app):
    """Refresh the canvas and annotation Treeview to display current annotations."""
    for item in app.canvas.find_withtag("annotation"):
        app.canvas.delete(item)
    
    for item in app.annotation_tree.get_children():
        app.annotation_tree.delete(item)
    
    if app.current_image_path in app.annotations:
        for i, ann in enumerate(app.annotations[app.current_image_path]):
            draw_annotation(app, i)
            
            x1, y1, x2, y2 = ann['bbox']
            app.annotation_tree.insert("", tk.END, values=(ann['class'], f"({x1},{y1},{x2},{y2})"))

def on_annotation_select(app, event):
    """Handle selection of an annotation in the Treeview and highlight it on the canvas."""
    selected_items = app.annotation_tree.selection()
    if selected_items:
        index = app.annotation_tree.index(selected_items[0])
        app.selected_annotation_index = index
        refresh_annotations_display(app)

def delete_selected_annotation(app, event=None):
    """Delete the selected annotation from the current image without changing the image."""
    if app.selected_annotation_index >= 0 and app.current_image_path in app.annotations:
        if app.selected_annotation_index < len(app.annotations[app.current_image_path]):
            app.save_state()
            
            del app.annotations[app.current_image_path][app.selected_annotation_index]
            
            app.selected_annotation_index = -1
            
            refresh_annotations_display(app)
            app.update_treeview_item()  # Update [A] indicator for the current image

def edit_selected_annotation(app):
    """Edit the class name of the selected annotation."""
    if app.selected_annotation_index >= 0 and app.current_image_path in app.annotations:
        ann = app.annotations[app.current_image_path][app.selected_annotation_index]
        
        class_name = tk.simpledialog.askstring("Edit Annotation", "Class name:", initialvalue=ann['class'])
        
        if class_name:
            ann['class'] = class_name
            
            if class_name not in app.classes:
                app.classes.append(class_name)
                app.class_combobox['values'] = app.classes
            
            refresh_annotations_display(app)
            app.update_treeview_item()  # Update [A] indicator for the current image

def add_class(app):
    """Add a new class name to the list of available classes."""
    class_name = tk.simpledialog.askstring("Add Class", "Enter new class name:")
    if class_name and class_name not in app.classes:
        app.classes.append(class_name)
        app.class_combobox['values'] = app.classes
        app.class_var.set(class_name)