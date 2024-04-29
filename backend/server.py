import eventlet
eventlet.monkey_patch()
import cv2
import numpy as np
import logging
import os
from flask import Flask, request, send_from_directory, jsonify, abort
from flask_socketio import SocketIO
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

basedir = os.path.abspath(os.path.dirname(__file__))

app.config['UPLOAD_FOLDER'] = 'backend/uploads'
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'outputs')
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")




@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify(error="No video part in the upload"), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    use_window = request.form.get('useWindow', 'false').lower() == 'true'
    fps = int(request.form.get('fps', 30))  
    window_size = request.form.get('windowSize')

    if use_window and not window_size:
        return jsonify(error="Window size is required when useWindow is true"), 400

    window_size = int(window_size) if window_size else None 

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)

    output_filename = process_video(video_path, fps, window_size, use_window)
    if output_filename:
        image_url = os.path.basename(output_filename)
        full_path = os.path.join(app.config['OUTPUT_FOLDER'], image_url)
        if os.path.isfile(full_path):
            socketio.emit('imageReady', {'imageUrl': image_url})
            return jsonify(message="File uploaded and processing started.", imageUrl=image_url), 200
        else:
            logging.error("File does not exist after writing: " + full_path)
    else:
        socketio.emit('processingError', {'error': "Error processing video"})
        return jsonify(error="Error in video processing"), 500

    
def process_video(video_path, fps, window_duration, use_window):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Failed to open video")
        return None

    try:
        if use_window:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_diffs = []
            ret, prev_frame = cap.read()
            if not ret:
                logging.error("Failed to read the first frame of the video")
                return None

            for current_frame in range(1, total_frames):
                ret, curr_frame = cap.read()
                if not ret:
                    break
                diff = cv2.absdiff(curr_frame, prev_frame)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                frame_diffs.append(np.sum(gray_diff))
                prev_frame = curr_frame
                
                progress = (current_frame / total_frames) * 100
                socketio.emit('processingProgress', {'progress': progress})

            window_size = int(window_duration * fps)
            max_window_score = 0
            max_window_start = 0
            for i in range(len(frame_diffs) - window_size + 1):
                window_score = sum(frame_diffs[i:i + window_size])
                if window_score > max_window_score:
                    max_window_score = window_score
                    max_window_start = i

            cap.set(cv2.CAP_PROP_POS_FRAMES, max_window_start)
            ret, accum_image = cap.read()
            if ret:
                accum_image = accum_image.astype(np.float32)
                for i in range(1, window_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.accumulateWeighted(frame.astype(np.float32), accum_image, alpha=0.1)
        else:
            ret, accum_image = cap.read()
            if not ret:
                logging.error("Failed to read the first frame for full video processing")
                return None
            accum_image = accum_image.astype(np.float32)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.accumulateWeighted(frame.astype(np.float32), accum_image, alpha=0.1)

        output_image = cv2.convertScaleAbs(accum_image)
        unique_filename = f"long_exposure_{uuid.uuid4()}.png"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], unique_filename)
        cv2.imwrite(output_path, output_image)
        cap.release()
        
    finally:
        cap.release()

    return unique_filename

@app.route('/outputs/<filename>')
def download_file(filename):
    actual_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    print(f"Trying to send file from: {actual_path}")

    if not os.path.isfile(actual_path):
        print(f"File not found: {actual_path}")
        abort(404) 
    
    try:
        response = send_from_directory(app.config['OUTPUT_FOLDER'], filename)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate' 
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except Exception as e:
        print(f"Failed to send file due to: {e}")
        abort(500)  

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['OUTPUT_FOLDER']):
        os.makedirs(app.config['OUTPUT_FOLDER'])
    socketio.run(app, port=3000)
