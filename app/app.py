import cv2
import torch
from flask import Flask, render_template, request, jsonify
from torch_utils import MyModel,frame_extract, create_face_video, get_data_transforms, preprocess, predict, output_predict
import traceback
import os
# define model class

from torch import nn
from torchvision import models


app = Flask(__name__)

ALLOWED_EXTENSIONS = {'MP4','MPEG-4','mp4','webm','avi','mov','wmv'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS


model = MyModel(2)
device = torch.device('cpu')
model.to(device)
PATH = "D:/docs/7th sem/major project 1/VS_Code/model(1_seq_10_7epoch).pth"
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()





@app.route('/', methods=['GET'])
def webpage():
    return render_template("index.html")


@app.route('/detection', methods=['POST'])
def predict_view():
    if request.method == 'POST':
        fileInput = request.files.get('fileInput')
        basepath= os.path.dirname(__file__)        
        input_video_path = os.path.join(basepath,'video_input',(fileInput.filename))
        print(f"Saving file to: {input_video_path}")
        fileInput.save(input_video_path)
        if fileInput is None or fileInput.filename=="":
            return jsonify({'error': 'no file'})
        if not allowed_file(fileInput.filename):
            return jsonify({'error': 'format is not supported'})
        try:
            

            output_video_name = 'output_face_video.avi'
            output_video_path = os.path.join(basepath, 'video_output', output_video_name)
            create_face_video(input_video_path, output_video_path)
            video_tensor = preprocess(output_video_path)
            predicted_labels = predict(video_tensor)
            output_result=output_predict(predicted_labels)
            return render_template('index.html', result=output_result)
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'error during prediction: {str(e)}'})

    return render_template('index.html')


app.run(debug=True)
