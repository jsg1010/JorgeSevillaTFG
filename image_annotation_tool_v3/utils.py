"""
General utility functions for the Image Annotation Tool.

This module contains functions for managing preferences and dataset splitting.
"""

import json
import os
import tkinter as tk

def load_preferences(app):
    """Load user preferences from a JSON file."""
    try:
        if os.path.exists(app.preferences_file):
            with open(app.preferences_file, 'r') as f:
                loaded_preferences = json.load(f)
                app.preferences.update(loaded_preferences)
                app.default_model_path = app.preferences["default_model_path"]
    except Exception as e:
        tk.messagebox.showwarning("Warning", f"Failed to load preferences: {e}")

def save_preferences(app):
    """Save user preferences to a JSON file."""
    try:
        with open(app.preferences_file, 'w') as f:
            json.dump(app.preferences, f, indent=2)
    except Exception as e:
        tk.messagebox.showerror("Error", f"Failed to save preferences: {e}")

def split_dataset(app):
    """
    Split the dataset into train, validation, and test sets using scikit-learn.
    
    Args:
        app: The ImageAnnotationTool instance.
    
    Returns:
        dict: A dictionary with 'train', 'val', and 'test' keys containing lists of image paths.
    """
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        tk.messagebox.showerror("Error", "The 'scikit-learn' module is not installed. Please install it with:\npip install scikit-learn")
        return {"train": [], "val": [], "test": []}
    
    train_ratio = app.preferences["train_split"] / 100
    val_ratio = app.preferences["val_split"] / 100
    test_ratio = app.preferences["test_split"] / 100
    
    img_paths = list(app.annotations.keys())
    if not img_paths:
        return {"train": [], "val": [], "test": []}
    
    train_imgs, temp_imgs = train_test_split(img_paths, train_size=train_ratio)
    
    if val_ratio + test_ratio > 0:
        val_relative_ratio = val_ratio / (val_ratio + test_ratio)
        val_imgs, test_imgs = train_test_split(temp_imgs, train_size=val_relative_ratio)
    else:
        val_imgs, test_imgs = [], temp_imgs
    
    return {"train": train_imgs, "val": val_imgs, "test": test_imgs}