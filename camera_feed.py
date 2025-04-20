import cv2
import imutils
import numpy as np
import torch
import torch.nn as nn
import time
import os

# Gesture labels
GESTURE_LABELS = ['Blank', 'OK', 'Thumbs Up', 'Thumbs Down', 'Punch', 'High Five']
MODEL_PATH = "hand_gesture_model.pth"

# Model Definition
class HandGestureModel(nn.Module):
    def __init__(self):
        super(HandGestureModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(64 * 25 * 30, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = x.view(-1, 64 * 25 * 30)
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

# Load PyTorch model
def load_model():
    model = HandGestureModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# Predict from frame with confidence check
def predict_from_frame(thresh_img, model):
    img = cv2.resize(thresh_img, (100, 120)).astype("float32") / 255.0
    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_idx].item()
    
    if confidence < 0.6:
        return "Blank"
    
    return GESTURE_LABELS[predicted_idx]

# Background model
bg = None
def run_avg(image, accumWeight=0.5):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
    cv2.accumulateWeighted(image, bg, accumWeight)

def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None
    segmented = max(cnts, key=cv2.contourArea)
    return thresholded, segmented

# Main live camera prediction
def live_gesture_prediction():
    print("[INFO] Starting camera...")
    model = load_model()
    cap = cv2.VideoCapture(0)

    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0
    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()

        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 30:
            run_avg(gray)
            if num_frames == 1:
                print("[STATUS] Calibrating background...")
            elif num_frames == 29:
                print("[STATUS] Calibration complete.")
        else:
            hand = segment(gray)
            if hand is not None:
                thresholded, segmented = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                
                contour_area = cv2.contourArea(segmented)
                if contour_area < 1500:
                    prediction = "Blank"
                else:
                    prediction = predict_from_frame(thresholded, model)

                cv2.putText(clone, prediction, (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Thresholded", thresholded)

        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        # FPS calculation
        fps = 1 / (time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(clone, f"FPS: {int(fps)}", (560, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        num_frames += 1
        cv2.imshow("Real-Time Gesture Recognition", clone)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session ended.")

if __name__ == "__main__":
    live_gesture_prediction()
