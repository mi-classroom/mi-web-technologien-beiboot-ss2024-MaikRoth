import json
from flask import Flask, request, render_template, send_file, url_for, Response
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import glob
import time
from concurrent.futures import ThreadPoolExecutor
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure upload, frames, and output folders
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static/uploads')
app.config['FRAMES_FOLDER'] = os.path.join(app.root_path, 'static/frames')
app.config['OUTPUT_FOLDER'] = os.path.join(app.root_path, 'outputs')
app.config['SERVER_NAME'] = '127.0.0.1:5000'

# Set up logging
logging.basicConfig(level=logging.INFO)

# Global variable to store progress
progress = 0

# Create the necessary directories if they don't exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['FRAMES_FOLDER'], app.config['OUTPUT_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created directory: {folder}")

# Initialize thread pool for concurrent execution
executor = ThreadPoolExecutor(max_workers=4)

# Set up logging
logging.basicConfig(level=logging.INFO)

def clear_frames_directory():
    """
    Clear all files in the frames directory.
    """
    try:
        files = glob.glob(os.path.join(app.config['FRAMES_FOLDER'], '*'))
        for f in files:
            os.remove(f)
        logging.info(f"Cleared frames directory: {app.config['FRAMES_FOLDER']}")
    except Exception as e:
        logging.error(f"Error clearing frames directory: {e}")

def extract_frames(video_path, start_time, end_time, fps, root_url):
    """
    Extract frames from the video between start_time and end_time at the specified fps.

    Args:
        video_path (str): Path to the video file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        fps (float): Frames per second to extract.
        root_url (str): Root URL for frame paths.

    Returns:
        list: List of URLs for the extracted frames.
    """
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * original_fps)
    end_frame = int(end_time * original_fps)
    frame_interval = max(1, int(original_fps / fps))  # Calculate frame interval

    current_frame = 0
    frame_paths = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if start_frame <= current_frame <= end_frame and current_frame % frame_interval == 0:
                frame_filename = f"frame_{current_frame}.jpg"
                frame_path = os.path.join(app.config['FRAMES_FOLDER'], frame_filename)
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])  # Save frame with JPEG quality of 85
                frame_url = f"{root_url}static/frames/{frame_filename}"
                frame_paths.append(frame_url)
            current_frame += 1
    except Exception as e:
        logging.error(f"Error extracting frames: {e}")
    finally:
        cap.release()

    logging.info(f"Extracted frames from {video_path}")
    return frame_paths

def create_long_exposure_image(selected_frames, highlighted_frames):
    """
    Create a long exposure image from the selected frames, optionally highlighting specific frames.

    Args:
        selected_frames (list): List of URLs of selected frames.
        highlighted_frames (list): List of URLs of highlighted frames.

    Returns:
        str: Path to the created long exposure image.

    Raises:
        ValueError: If no valid frames are provided for processing.
    """
    exposure_image = None
    valid_frame_count = 0
    exposure_factor = 1.5  # Exposure factor for blending frames

    for frame_url in selected_frames:
        frame_path = frame_url.replace(request.url_root, '').replace('/static/', 'static/')
        frame_full_path = os.path.join(app.root_path, frame_path)
        if not os.path.exists(frame_full_path):
            logging.warning(f"Frame file does not exist: {frame_full_path}")
            continue
        frame = cv2.imread(frame_full_path).astype(np.float32)
        if exposure_image is None:
            exposure_image = np.zeros_like(frame)
        exposure_image += frame * exposure_factor
        valid_frame_count += 1

    if exposure_image is not None and valid_frame_count > 0:
        exposure_image /= valid_frame_count

        if highlighted_frames:
            for highlighted_frame in highlighted_frames:
                highlighted_path = highlighted_frame.replace(request.url_root, '').replace('/static/', 'static/')
                highlighted_full_path = os.path.join(app.root_path, highlighted_path)
                if os.path.exists(highlighted_full_path):
                    highlighted_img = cv2.imread(highlighted_full_path).astype(np.float32)
                    exposure_image = highlight_object(exposure_image, highlighted_img)

        exposure_image = np.clip(exposure_image, 0, 255).astype(np.uint8)
        timestamp = int(time.time())
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'long_exposure_{timestamp}.jpg')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logging.info(f"Saving long exposure image to: {output_path}")
        cv2.imwrite(output_path, exposure_image)
        return output_path
    else:
        raise ValueError("No valid frames to process.")

def highlight_object(exposure_image, highlighted_img):
    """
    Highlight the main object in the highlighted image within the exposure image.

    Args:
        exposure_image (np.ndarray): The long exposure image.
        highlighted_img (np.ndarray): The highlighted frame image.

    Returns:
        np.ndarray: The updated exposure image with the highlighted object.
    """
    back_sub = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=50, detectShadows=False)
    fg_mask = back_sub.apply(highlighted_img)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(highlighted_img, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        mask = mask[:, :, 0] / 255.0

        alpha = 0.3  # Transparency factor for highlighted object
        for c in range(3):
            exposure_image[:, :, c] = (1 - mask * alpha) * exposure_image[:, :, c] + mask * alpha * highlighted_img[:, :, c]
    return np.clip(exposure_image, 0, 255)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    Handle file upload and frame extraction.

    Returns:
        Rendered template or redirection based on the request method.
    """
    if request.method == 'POST':
        clear_frames_directory()  # Clear the frames directory before processing
        video = request.files['video']
        
        # Validate and parse form data
        try:
            start_time = float(request.form['start_time'])
            end_time = float(request.form['end_time'])
            fps = float(request.form['fps'])
        except (ValueError, KeyError) as e:
            logging.error(f"Error parsing form data: {e}")
            return "Invalid form data.", 400

        # Validate the video file
        if not video or not video.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            logging.error("Invalid video file format.")
            return "Invalid video file format. Only .mp4, .avi, .mov, .mkv are allowed.", 400

        filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(video_path)

        logging.info(f"Uploaded video to {video_path}")

        root_url = request.url_root
        future = executor.submit(extract_frames, video_path, start_time, end_time, fps, root_url)
        frame_paths = future.result()

        return render_template('select_frames.html', frames=frame_paths)

    return render_template('upload.html')

@app.route('/process_frames', methods=['POST'])
def process_frames():
    """
    Process selected frames to create a long exposure image.

    Returns:
        The created long exposure image file or error message.
    """
    try:
        selected_frames_json = request.form.get('selectedFrames')
        highlighted_frames_json = request.form.get('highlightedFrames')

        if not selected_frames_json:
            logging.error("No frames selected for processing.")
            return "No frames selected for processing.", 400

        selected_frames = json.loads(selected_frames_json)
        highlighted_frames = json.loads(highlighted_frames_json) if highlighted_frames_json else []

        if not selected_frames:
            logging.error("No frames selected for processing.")
            return "No frames selected for processing.", 400
        
        output_image_path = create_long_exposure_image(selected_frames, highlighted_frames)
        return send_file(output_image_path, as_attachment=True)
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        return f"An error occurred during processing: {e}. Please check the server logs for more details.", 500
    except Exception as e:
        logging.error(f"Error: {e}")
        return f"An error occurred during processing: {e}. Please check the server logs for more details.", 500


def generate_preview(selected_frames, highlighted_frames):
    global progress
    exposure_image = None
    valid_frame_count = 0
    exposure_factor = 1.5 

    def convert_url_to_path(url):
        return os.path.join(app.root_path, url.replace(request.url_root, '').replace('/static/', 'static/'))

    selected_frame_paths = [convert_url_to_path(frame_url) for frame_url in selected_frames]
    frames_to_process = selected_frame_paths  

    total_frames = len(frames_to_process)

    if total_frames == 0:
        logging.error("No frames available for processing.")
        raise ValueError("No frames available for processing.")

    try:
        for i, frame_path in enumerate(frames_to_process):
            if not os.path.exists(frame_path):
                logging.error(f"Frame path does not exist: {frame_path}")
                continue

            frame = cv2.imread(frame_path)
            if frame is None:
                logging.error(f"Failed to read frame: {frame_path}")
                continue

            frame = frame.astype(np.float32)

            if exposure_image is None:
                exposure_image = np.zeros_like(frame)

            exposure_image += frame * exposure_factor
            valid_frame_count += 1

            progress = int((i + 1) / total_frames * 100)

        if valid_frame_count > 0:
            exposure_image /= valid_frame_count

        if highlighted_frames:
            for highlighted_frame in highlighted_frames:
                highlighted_path = convert_url_to_path(highlighted_frame)
                if os.path.exists(highlighted_path):
                    highlighted_img = cv2.imread(highlighted_path).astype(np.float32)
                    exposure_image = highlight_object(exposure_image, highlighted_img)

        exposure_image = np.clip(exposure_image, 0, 255).astype(np.uint8)

        timestamp = int(time.time())
        output_folder = app.config['OUTPUT_FOLDER']
        output_path = os.path.join(output_folder, f'preview_{timestamp}.jpg')

        os.makedirs(output_folder, exist_ok=True)
        logging.info(f"Saving preview image to: {output_path}")

        cv2.imwrite(output_path, exposure_image) 
        
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Failed to create the preview image at {output_path}")

        return output_path

    except Exception as e:
        logging.error(f"Error generating preview: {e}")
        raise

@app.route('/process_preview', methods=['POST'])
def process_preview():
    """
    Generate a preview of the long exposure image based on selected frames.

    Returns:
        The generated preview image or an error message.
    """
    try:
        data = request.get_json()
        selected_frames = data.get('selectedFrames')
        highlighted_frames = data.get('highlightedFrames', [])

        if not selected_frames:
            logging.error("No frames selected for preview.")
            return "No frames selected for preview.", 400

        # Reset progress
        global progress
        progress = 0

        # Generate the long exposure image for preview
        output_image_path = generate_preview(selected_frames, highlighted_frames)

        # Check if file exists before sending it
        if os.path.exists(output_image_path):
            return send_file(output_image_path, mimetype='image/jpeg')
        else:
            logging.error(f"File not found: {output_image_path}")
            return "Error: Preview image could not be generated.", 500

    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        return f"An error occurred during preview generation: {e}.", 500
    except Exception as e:
        logging.error(f"Error generating preview: {e}")
        return f"An error occurred during preview generation: {e}.", 500


@app.route('/progress')
def progress_stream():
    """
    Stream the progress percentage to the client.
    """
    def generate():
        global progress
        while progress < 100:
            time.sleep(0.5)  # Adjust the interval as needed
            yield f"data:{progress}\n\n"
        yield "data:100\n\n"

    return Response(generate(), mimetype='text/event-stream')
# Run the app in debug mode
if __name__ == '__main__':
    app.run(debug=True)
