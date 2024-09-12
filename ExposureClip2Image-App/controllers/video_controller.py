from flask import Blueprint, render_template, request, send_file, Response
import logging
import json
import time
import os
from typing import List

def create_video_controller(video_service, executor):
    video_controller = Blueprint('video_controller', __name__)
    generated_preview_path = None
    progress = {'value': 0} 

    @video_controller.route('/', methods=['GET', 'POST'])
    def upload_file():
        """
        Handle file upload and frame extraction.

        Returns:
            Rendered template or redirection based on the request method.
        """
        if request.method == 'POST':
            video_service.processor.clear_frames_directory() 
            video = request.files['video']
            
            try:
                start_time, end_time, fps = video_service.parse_form_data(request.form)
            except ValueError as e:
                return str(e), 400

            try:
                video_path = video_service.save_video(video)
            except ValueError as e:
                return str(e), 400

            root_url = request.url_root
            future = executor.submit(video_service.process_video, video_path, start_time, end_time, fps, root_url)
            frame_paths = future.result()

            return render_template('select_frames.html', frames=frame_paths)

        return render_template('upload.html')

    @video_controller.route('/process_frames', methods=['POST'])
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

            output_image_path = video_service.create_long_exposure(selected_frames_json, highlighted_frames_json)
            return send_file(output_image_path, as_attachment=True)
        except ValueError as e:
            logging.error(f"Error: {e}")
            return f"An error occurred during processing: {e}.", 500
        except Exception as e:
            logging.error(f"Error: {e}")
            return f"An error occurred during processing: {e}.", 500

    @video_controller.route('/process_preview', methods=['POST'])
    def process_preview():
        """
        Generate a preview of the long exposure image based on selected frames.
        Returns:
            The generated preview image or an error message.
        """
        nonlocal generated_preview_path
        try:
            data = request.get_json()
            selected_frames = data.get('selectedFrames')
            highlighted_frames = data.get('highlightedFrames', [])
            selected_filter = data.get('filter', 'none') 

            if not selected_frames:
                logging.error("No frames selected for preview.")
                return "No frames selected for preview.", 400

            progress['value'] = 0

            output_image_path = video_service.generate_preview(selected_frames, highlighted_frames, selected_filter, progress)
            generated_preview_path = output_image_path  

            if output_image_path and os.path.exists(output_image_path):
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

    @video_controller.route('/progress')
    def progress_stream():
        """
        Stream the progress percentage to the client.
        """
        def generate():
            while progress['value'] < 100:
                time.sleep(0.5) 
                yield f"data:{progress['value']}\n\n"
            yield "data:100\n\n"

        return Response(generate(), mimetype='text/event-stream')

    return video_controller
