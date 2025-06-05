"""
Utilities for exporting annotations in various formats from the Image Annotation Tool.

This module contains helper functions to export annotations in COCO, YOLO, and Pascal VOC formats.
"""

import os
import json
from datetime import datetime
from PIL import Image

def export_coco_format(app, export_dir):
    """
    Export annotations in COCO format to the specified directory.
    
    Args:
        app: The ImageAnnotationTool instance.
        export_dir (str): Directory to export the annotations to.
    """
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
    
    for i, class_name in enumerate(app.classes, 1):
        coco_data["categories"].append({
            "id": i,
            "name": class_name,
            "supercategory": "none"
        })
    
    image_id = 1
    annotation_id = 1
    
    for img_path, annotations in app.annotations.items():
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
            
            category_id = app.classes.index(ann["class"]) + 1
            
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

def export_yolo_format(app, export_dir):
    """
    Export annotations in YOLO format to the specified directory with train/val/test splits.
    
    Args:
        app: The ImageAnnotationTool instance.
        export_dir (str): Directory to export the annotations to.
    """
    from utils import split_dataset

    # Create directories for train, val, and test splits
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(export_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(export_dir, "labels", split), exist_ok=True)
    
    # Save class names
    with open(os.path.join(export_dir, "classes.txt"), 'w') as f:
        for class_name in app.classes:
            f.write(f"{class_name}\n")
    
    # Split the dataset
    dataset_splits = split_dataset(app)
    
    # Process each split
    for split in splits:
        for img_path in dataset_splits[split]:
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            img_filename = os.path.basename(img_path)
            label_filename = os.path.splitext(img_filename)[0] + ".txt"
            
            # Save the image
            img_dest = os.path.join(export_dir, "images", split, img_filename)
            img.save(img_dest)
            
            # Create and save the label file
            label_path = os.path.join(export_dir, "labels", split, label_filename)
            with open(label_path, 'w') as f:
                for ann in app.annotations[img_path]:
                    bbox = ann["bbox"]
                    class_id = app.classes.index(ann["class"])
                    
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) / 2 / img_width
                    center_y = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")

def export_voc_format(app, export_dir):
    """
    Export annotations in Pascal VOC format to the specified directory.
    
    Args:
        app: The ImageAnnotationTool instance.
        export_dir (str): Directory to export the annotations to.
    """
    annotations_dir = os.path.join(export_dir, "Annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    
    images_dir = os.path.join(export_dir, "JPEGImages")
    os.makedirs(images_dir, exist_ok=True)
    
    imagesets_dir = os.path.join(export_dir, "ImageSets", "Main")
    os.makedirs(imagesets_dir, exist_ok=True)
    
    with open(os.path.join(export_dir, "classes.txt"), 'w') as f:
        for class_name in app.classes:
            f.write(f"{class_name}\n")
    
    image_filenames = []
    
    for img_path, annotations in app.annotations.items():
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