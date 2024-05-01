import cv2
import torch
from flask import Flask, render_template, request, jsonify
from torch_utils import frame_extract, create_face_video, get_data_transforms, preprocess, predict, output_predict



input_video_path = "D:/docs/7th sem/major project 1/VS_Code/00107.mp4"
output_video_path = "D:/docs/7th sem/major project 1/VS_Code/onlyface3.mp4"
create_face_video(input_video_path, output_video_path)
video_tensor = preprocess(output_video_path)
predicted_labels = predict(video_tensor)
output_predict(predicted_labels)