from flask import Blueprint, render_template, request, send_file, Response
import logging
import json
import time
import os
from typing import List
from concurrent.futures import ThreadPoolExecutor


def create_video_controller(video_service, executor : ThreadPoolExecutor):
    # Create a new Blueprint for video-related routes
    video_controller = Blueprint('video_controller', __name__)
    generated_preview_path = None  # Variable to store the path of the generated preview image
    progress = {'value': 0}  # Dictionary to track the progress of preview generation

    @video_controller.route('/', methods=['GET', 'POST'])
    def upload_file():
        """
        Handle file upload and frame extraction.
        Returns:
            Rendered template or redirection based on the request method.
        """
        if request.method == 'POST':
            # Clear any existing frames in the frames directory
            video_service.processor.clear_frames_directory() 
            # Get the uploaded video file from the form
            video = request.files['video']

            try:
                # Parse form data to extract start time, end time, and fps
                start_time, end_time, fps = video_service.parse_form_data(request.form)
            except ValueError as e:
                # Return an error message if form data is invalid
                return str(e), 400

            try:
                # Save the uploaded video file
                video_path = video_service.save_video(video)
            except ValueError as e:
                # Return an error message if video saving fails
                return str(e), 400

            # Get the root URL of the request
            root_url = request.url_root
            # Submit a task to process the video in a separate thread
            future = executor.submit(video_service.process_video, video_path, start_time, end_time, fps, root_url)
            # Get the list of extracted frame paths once the task is complete
            frame_paths = future.result()

            # Render the template for selecting frames
            return render_template('select_frames.html', frames=frame_paths)

        # Render the upload template for GET requests
        return render_template('upload.html')

    @video_controller.route('/process_frames', methods=['POST'])
    def process_frames():
        """
        Process selected frames to create a long exposure image with an optional filter.
        Returns:
            The created long exposure image file with the filter applied or an error message.
        """
        try:
            # Get the selected and highlighted frames from the form data
            selected_frames_json = request.form.get('selectedFrames')
            highlighted_frames_json = request.form.get('highlightedFrames')
            selected_filter = request.form.get('filter', 'none')  # Get the selected filter, defaulting to 'none'

            if not selected_frames_json:
                # Log an error and return a message if no frames are selected
                logging.error("No frames selected for processing.")
                return "No frames selected for processing.", 400

            # Create a long exposure image using the selected frames
            long_exposure_image_path = video_service.create_long_exposure(selected_frames_json, highlighted_frames_json)

            # Apply the selected filter to the long exposure image
            filtered_image_path = video_service.processor.apply_filter_to_image(long_exposure_image_path, selected_filter)

            # Send the filtered image file as an attachment
            return send_file(filtered_image_path, as_attachment=True)
        except ValueError as e:
            # Log and return an error message if a ValueError occurs
            logging.error(f"Error: {e}")
            return f"An error occurred during processing: {e}.", 500
        except Exception as e:
            # Log and return an error message if any other exception occurs
            logging.error(f"Error: {e}")
            return f"An error occurred during processing: {e}.", 500


    @video_controller.route('/process_preview', methods=['POST'])
    def process_preview():
        """
        Generate a preview of the long exposure image based on selected frames.
        Returns:
            The generated preview image or an error message.
        """
        nonlocal generated_preview_path  # Access the outer scope variable to store the generated preview path
        try:
            # Parse the JSON data from the request
            data = request.get_json()
            selected_frames = data.get('selectedFrames')
            highlighted_frames = data.get('highlightedFrames', [])
            selected_filter = data.get('filter', 'none')  # Get the selected filter, defaulting to 'none'

            if not selected_frames:
                # Log an error and return a message if no frames are selected for preview
                logging.error("No frames selected for preview.")
                return "No frames selected for preview.", 400

            # Reset the progress tracker
            progress['value'] = 0

            # Generate a preview image using the selected frames and filter
            output_image_path = video_service.generate_preview(selected_frames, highlighted_frames, selected_filter, progress)
            generated_preview_path = output_image_path  # Store the path of the generated preview image

            if output_image_path and os.path.exists(output_image_path):
                # Send the preview image file if it was successfully generated
                return send_file(output_image_path, mimetype='image/jpeg')
            else:
                # Log and return an error if the preview image could not be generated
                logging.error(f"File not found: {output_image_path}")
                return "Error: Preview image could not be generated.", 500

        except json.JSONDecodeError as e:
            # Log and return an error message if there is a JSON parsing error
            logging.error(f"JSON decode error: {e}")
            return f"An error occurred during preview generation: {e}.", 500
        except Exception as e:
            # Log and return an error message if any other exception occurs
            logging.error(f"Error generating preview: {e}")
            return f"An error occurred during preview generation: {e}.", 500

    @video_controller.route('/progress')
    def progress_stream():
        """
        Stream the progress percentage to the client.
        """
        def generate():
            # Stream the progress value to the client, updating every 0.5 seconds
            while progress['value'] < 100:
                time.sleep(0.5) 
                yield f"data:{progress['value']}\n\n"
            # Ensure the client receives 100% progress at the end
            yield "data:100\n\n"

        # Return the generated progress stream as a Server-Sent Event (SSE)
        return Response(generate(), mimetype='text/event-stream')

    # Return the created Blueprint to be registered with the Flask app
    return video_controller
