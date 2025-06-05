"""
Main entry point for the Image Annotation Tool with YOLO Integration.

This script initializes and runs the application for creating datasets to train YOLO models
for road defect detection.

Usage:
    python main.py
"""

import tkinter as tk
from app import ImageAnnotationTool

def main():
    """Initialize and start the Tkinter application."""
    root = tk.Tk()
    app = ImageAnnotationTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()