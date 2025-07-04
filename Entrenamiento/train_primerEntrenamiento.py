from ultralytics import YOLO

# Load a better model (YOLOv8m is more accurate than YOLOv8n).
# model = YOLO('yolov8m.pt')
model = YOLO('pretrained.pt')

# Training.
if __name__ == '__main__':
    results = model.train(
        data='data.yaml',  # Path to dataset configuration.
        imgsz=640,  # Image size.
        epochs=50,  # More epochs for better convergence.
        batch=16,  # Adjust based on your GPU.
        device='cuda',  # Use GPU if available.
        workers=4,  # Speed up data loading.
        optimizer='AdamW',  # More stable than SGD for some cases.
        lr0=0.001,  # Initial learning rate.
        cos_lr=True,  # Cosine learning rate schedule.
        weight_decay=0.0005,  # Regularization.
        augment=True,  # Enable data augmentation.
        patience=10,  # Stop training if no improvement.
        save=True,  # Save best model.
        save_period=10,  # Save checkpoints every 5 epochs.
        project='crack_detection',  # Custom project name.
        name='yolo11l_exp1',  # Experiment name.
    )

    # Validate the model on the validation set.
    metrics = model.val()

    # Export the model for deployment.
    model.export(format='onnx')  # Export as ONNX for further inference optimization.
