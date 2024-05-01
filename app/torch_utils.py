import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
import random
import pandas as pd
import glob
import matplotlib.pyplot as plt
import dlib
import traceback
# define model class

from torch import nn
from torchvision import models


class MyModel(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(MyModel, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)  # Residual Network CNN
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))


model = MyModel(2)

device = torch.device('cpu')
PATH = "D:/docs/7th sem/major project 1/VS_Code/model(1_seq_10_7epoch).pth"
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()





#create only face

def frame_extract(path):
    vidObj = cv2.VideoCapture(str(path))  # Ensure that path is passed as a string
    success = True
    frame_count = 0  # Add this line to count frames
    while success:
        success, image = vidObj.read()
        if success:
            frame_count += 1  # Increment frame count
            yield image

    print(f"Number of frames captured: {frame_count}")


def create_face_video(input_video_path, output_video_path):
    frames = []
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (112, 112))

    # Initialize face detector from dlib
    face_detector = dlib.get_frontal_face_detector()

    for idx, frame in enumerate(frame_extract(input_video_path)):
        if idx <= 150:
            frames.append(frame)
            if len(frames) == 4:
                # Convert frames to grayscale for face detection
                gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

                # Detect faces using dlib face detector
                faces = [face_detector(gray_frame) for gray_frame in gray_frames]

                for i, face in enumerate(faces):
                    if face:
                        top, right, bottom, left = (face[0].top(), face[0].right(), face[0].bottom(), face[0].left())
                        try:
                            out.write(cv2.resize(frames[i][top:bottom, left:right, :], (112, 112)))
                        except:
                            pass
                frames = []

    out.release()


#preprocess
def get_data_transforms(im_size=112):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transforms


def preprocess(random_video_path):
    video_frames = []
    cap = cv2.VideoCapture(random_video_path)

    # Check if the video capture is successful
    if not cap.isOpened():
        print(f"Error: Unable to open video file: {random_video_path}")
        return None

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            if frame is not None and frame.size != 0:  # Check for non-empty frames
                video_frames.append(frame)

        # Check if any frames were captured
        if not video_frames:
            print("Error: No frames were captured from the video.")
            return None

        print(f"Number of frames captured: {len(video_frames)}")

        # Preprocess the video frames
        data_transform = get_data_transforms()
        processed_frames = [data_transform(frame) for frame in video_frames]

        # Check if the processed frames list is empty
        if not processed_frames:
            print("Error: No frames were successfully processed.")
            return None

        # Check the shape of the first processed frame
        print(f"Shape of the first processed frame: {processed_frames[0].shape}")

        # Stack the processed frames into a video tensor
        video_tensor = torch.stack(processed_frames)

        # Ensure that the sequence_length parameter matches the one used during training
        sequence_length = 10
        video_tensor = video_tensor[:sequence_length]

        # Check the shape of the resulting video tensor
        print(f"Shape of the video tensor: {video_tensor.shape}")

        return video_tensor

    finally:
        cap.release()

    






def predict(video_tensor):
    # Run inference
    if torch.cuda.is_available():
        video_tensor = video_tensor.cuda()

    predicted_labels = []

    with torch.no_grad():
        try:
            for frame_tensor in video_tensor:
                # Unsqueeze to add the batch and sequence dimensions
                frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(1)
                _, predictions = model(frame_tensor)
                _, predicted_label = torch.max(predictions, 1)
                predicted_label = predicted_label.item()

                # Accumulate predicted labels
                predicted_labels.append(predicted_label)

        except Exception as e:
            traceback.print_exc()
            print(f"Error during prediction: {str(e)}")
            return []  # Return an empty list if there's an issue during prediction

    return predicted_labels
 


def output_predict(predicted_labels):
    if not predicted_labels:
        return "No predictions available."

    # Calculate the number of 'REAL' and 'FAKE' predictions
    num_real = predicted_labels.count(1)
    num_fake = len(predicted_labels) - num_real

    # Check if the denominator is zero
    if len(predicted_labels) == 0:
        return "Error: Division by zero."

    # Calculate accuracies for 'REAL' and 'FAKE'
    real_accuracy = (num_real / len(predicted_labels)) * 100.0
    fake_accuracy = (num_fake / len(predicted_labels)) * 100.0

    # Determine the overall predicted label
    if real_accuracy > fake_accuracy:
        result = f"THE VIDEO IS REAL    Accuracy: {real_accuracy:.2f}%"
    else:
        result = f"THE VIDEO IS FAKE    Accuracy: {fake_accuracy:.2f}%"

    return result


    





 