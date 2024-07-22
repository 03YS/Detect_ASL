import torch
import joblib
import numpy as np
import cv2
import argparse
import time
import cnn_models

# Load label binarizer
lb = joblib.load('../outputs/lb.pkl')

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = cnn_models.CustomCNN().to(device)
model.load_state_dict(torch.load('../outputs/model.pth', map_location=device))
print(model)
print('Model loaded')

def hand_area(img):
    """Extract and resize the hand area from the frame."""
    hand = img[100:324, 100:324]
    hand = cv2.resize(hand, (224, 224))
    return hand

# Open video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Error while trying to open camera. Please check again...')
    exit()

# Get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define codec and create VideoWriter object
out = cv2.VideoWriter('../outputs/asl.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Read until end of video
while cap.isOpened():
    # Capture each frame of the video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get the hand area on the video capture screen
    cv2.rectangle(frame, (100, 100), (324, 324), (20, 34, 255), 2)
    hand = hand_area(frame)
    
    # Preprocess and make prediction
    image = np.transpose(hand, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float).to(device)
    image = image.unsqueeze(0)
    
    outputs = model(image)
    _, preds = torch.max(outputs.data, 1)
    
    # Display the result on the frame
    cv2.putText(frame, lb.classes_[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow('image', frame)
    out.write(frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release VideoCapture and VideoWriter
cap.release()
out.release()

# Close all frames and video windows
cv2.destroyAllWindows()
