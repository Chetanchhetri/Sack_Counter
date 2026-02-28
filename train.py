from ultralytics import YOLO
import os

# 1. Load the foundation model
model = YOLO('yolov8n.pt') 

# 2. Train and save directly to the current folder
# 'project' sets the root folder, 'name' sets the subfolder name
model.train(
    data='data.yaml', 
    epochs=20, 
    imgsz=640,
    project='.',          # Saves in the current directory instead of 'runs/'
    name='custom_sack_model', # Your model will be in 'custom_sack_model/weights/best.pt'
    exist_ok=True         
)
