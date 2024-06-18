import json

from flask import Flask, request, render_template, send_file, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import glob
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['FRAMES_FOLDER'] = 'static/frames'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Create necessary directories if they do not exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['FRAMES_FOLDER'], app.config['OUTPUT_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def clear_frames_directory():
    files = glob.glob(os.path.join(app.config['FRAMES_FOLDER'], '*'))
    for f in files:
        os.remove(f)

def extract_frames(video_path, start_time, end_time, fps):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * original_fps)
    end_frame = int(end_time * original_fps)
    frame_interval = max(1, int(original_fps / fps))  # Ensure frame_interval is at least 1
    current_frame = 0

    frame_paths = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if start_frame <= current_frame <= end_frame and current_frame % frame_interval == 0:
            frame_filename = f"frame_{current_frame}.jpg"
            frame_path = os.path.join(app.config['FRAMES_FOLDER'], frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(url_for('static', filename=f'frames/{frame_filename}', _external=True))
        current_frame += 1
    cap.release()
    return frame_paths

def create_long_exposure_image(selected_frames):
    exposure_image = None
    valid_frame_count = 0

    for frame_url in selected_frames:
        frame_path = frame_url.replace(request.url_root, '').replace('/static/', 'static/')
        frame_full_path = os.path.join(app.root_path, frame_path)
        if not os.path.exists(frame_full_path):
            print(f"Frame file does not exist: {frame_full_path}")
            continue
        frame = cv2.imread(frame_full_path)
        if frame is None:
            print(f"Error reading frame: {frame_full_path}")
            continue
        frame = frame.astype(np.float32)
        if exposure_image is None:
            exposure_image = np.zeros_like(frame)
        exposure_image += frame
        valid_frame_count += 1

    if exposure_image is not None and valid_frame_count > 0:
        exposure_image /= valid_frame_count
        exposure_image = np.uint8(exposure_image)
        timestamp = int(time.time())  # Generate a unique timestamp
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'long_exposure_{timestamp}.jpg')
        cv2.imwrite(output_path, exposure_image)
        return output_path
    else:
        raise ValueError("No valid frames to process.")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        clear_frames_directory()  # Clear frames directory before processing a new video

        video = request.files['video']
        start_time = float(request.form['start_time'])
        end_time = float(request.form['end_time'])
        fps = float(request.form['fps'])

        filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(video_path)

        frame_paths = extract_frames(video_path, start_time, end_time, fps)
        return render_template('select_frames.html', frames=frame_paths)

    return render_template('upload.html')

@app.route('/process_frames', methods=['POST'])
def process_frames():
    try:
        selected_frames_json = request.form.get('selectedFrames')
        selected_frames = json.loads(selected_frames_json)
        if not selected_frames:
            return "No frames selected for processing.", 400
        output_image_path = create_long_exposure_image(selected_frames)
        return send_file(output_image_path, as_attachment=True)
    except Exception as e:
        print(f"Error: {e}")
        return f"An error occurred during processing: {e}. Please check the server logs for more details.", 500

if __name__ == '__main__':
    app.run(debug=True)