import cv2
import numpy as np
import logging
import os
from flask import Flask, request, send_from_directory, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
from werkzeug.utils import secure_filename


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app, support_credentials=True)

basedir = os.path.abspath(os.path.dirname(__file__))

app.config['UPLOAD_FOLDER'] = 'backend/uploads'
app.config['OUTPUT_FOLDER'] = os.path.join(basedir, 'outputs')
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")




@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        logging.error("No video part in the upload")
        return "No file part", 400

    file = request.files['video']
    if file.filename == '':
        logging.error("No file selected for upload")
        return "No selected file", 400

    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        logging.info(f"Video saved at {video_path}")
        output_image_path = process_video(video_path, fps=30)
        if output_image_path:
            image_url = os.path.basename(output_image_path)
            socketio.emit('imageReady', {'imageUrl': image_url})
            logging.info(f"Image ready at {output_image_path}")
            return jsonify(message="File uploaded and processing started."), 200
        else:
            socketio.emit('processingError', {'error': "Error processing video"})
            logging.error("Error processing video")
            return jsonify(error="Error in video processing"), 500

def process_video(video_path, fps, window_duration=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Failed to open video")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_diffs = []
    ret, prev_frame = cap.read()
    current_frame = 1  
    if not ret:
        logging.error("Failed to read the first frame of the video")
        return None

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        diff = cv2.absdiff(curr_frame, prev_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        frame_diffs.append(np.sum(gray_diff))
        prev_frame = curr_frame
        
        progress = (current_frame / total_frames) * 100
        socketio.emit('processingProgress', {'progress': progress})
        current_frame += 1
    cap.release()

    window_size = int(window_duration * fps)
    max_window_score = 0
    max_window_start = 0

    for i in range(len(frame_diffs) - window_size + 1):
        window_score = sum(frame_diffs[i:i+window_size])
        if window_score > max_window_score:
            max_window_score = window_score
            max_window_start = i

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max_window_start)

    ret, accum_image = cap.read()
    accum_image = accum_image.astype(np.float32)

    for i in range(1, window_size):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.accumulateWeighted(frame.astype(np.float32), accum_image, alpha=0.1)

    output_image = cv2.convertScaleAbs(accum_image)
    output_path = os.path.join('backend','outputs','long_exposure.png')
    cv2.imwrite(output_path, output_image)
    cap.release()

    return output_path

@app.route('/outputs/<filename>')
def download_file(filename):
    actual_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    print(f"Trying to send file from: {actual_path}")  # Debugging line
    try:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
    except Exception as e:
        print(f"Failed to send file due to: {e}")  # Debugging line
        return f"File not found: {filename}", 404

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['OUTPUT_FOLDER']):
        os.makedirs(app.config['OUTPUT_FOLDER'])
    socketio.run(app, port=3000)
