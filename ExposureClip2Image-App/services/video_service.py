from models.video_processor import VideoProcessor
from typing import List, Tuple
import json
import logging
from werkzeug.datastructures import FileStorage

class VideoService:
    def __init__(self, app):
        # Initialize the VideoService with the Flask app instance
        self.app = app
        # Create an instance of VideoProcessor using the app configuration
        self.processor = VideoProcessor(app)

    def parse_form_data(self, form_data) -> Tuple[float, float, float]:
        """
        Parse form data to extract start time, end time, and frames per second (fps).

        Args:
            form_data: The form data submitted by the user.

        Returns:
            Tuple containing start_time, end_time, and fps as floats.

        Raises:
            ValueError: If the form data is invalid or missing required fields.
        """
        try:
            # Extract start_time, end_time, and fps from the form data and convert them to floats
            start_time = float(form_data['start_time'])
            end_time = float(form_data['end_time'])
            fps = float(form_data['fps'])
            return start_time, end_time, fps
        except (ValueError, KeyError) as e:
            # Log an error if parsing fails and raise a ValueError
            logging.error(f"Error parsing form data: {e}")
            raise ValueError("Invalid form data.")

    def save_video(self, video_file: FileStorage) -> str:
        """
        Save the uploaded video file to the server.

        Args:
            video_file (Filestorage): The uploaded video file.

        Returns:
            str: The file path where the video is saved.

        Raises:
            ValueError: If the video file is invalid or has an unsupported format.
        """
        from werkzeug.utils import secure_filename
        import os

        # Check if the video file is provided and has a valid extension
        if not video_file or not video_file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            logging.error("Invalid video file format.")
            raise ValueError("Invalid video file format. Only .mp4, .avi, .mov, .mkv are allowed.")

        # Secure the filename to prevent directory traversal attacks
        filename = secure_filename(video_file.filename)
        # Construct the full path to save the video
        video_path = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
        # Save the video file to the specified path
        video_file.save(video_path)

        logging.info(f"Uploaded video to {video_path}")
        return video_path

    def process_video(self, video_path: str, start_time: float, end_time: float, fps: float, root_url: str) -> List[str]:
        """
        Process the video by extracting frames within a specified time range and at a specified frame rate.

        Args:
            video_path (str): The path to the video file.
            start_time (float): The start time in seconds.
            end_time (float): The end time in seconds.
            fps (float): Frames per second to extract.
            root_url (str): The root URL for constructing frame URLs.

        Returns:
            List[str]: A list of URLs for the extracted frames.
        """
        # Use the VideoProcessor to extract frames from the video
        return self.processor.extract_frames(video_path, start_time, end_time, fps, root_url)

    def create_long_exposure(self, selected_frames_json: str, highlighted_frames_json: str) -> str:
        """
        Create a long exposure image from selected frames, optionally highlighting specific frames.

        Args:
            selected_frames_json (str): JSON string of selected frame URLs.
            highlighted_frames_json (str): JSON string of highlighted frame URLs.

        Returns:
            str: The path to the created long exposure image.

        Raises:
            ValueError: If no frames are selected for processing.
            json.JSONDecodeError: If there is an error parsing the JSON strings.
        """
        try:
            # Parse the JSON strings to get lists of frame URLs
            selected_frames = json.loads(selected_frames_json)
            highlighted_frames = json.loads(highlighted_frames_json) if highlighted_frames_json else []
            if not selected_frames:
                # Log an error if no frames are selected and raise a ValueError
                logging.error("No frames selected for processing.")
                raise ValueError("No frames selected for processing.")
            # Use the VideoProcessor to create the long exposure image
            return self.processor.create_long_exposure_image(selected_frames, highlighted_frames)
        except json.JSONDecodeError as e:
            # Log a JSON decode error if parsing fails
            logging.error(f"JSON decode error: {e}")
            raise

    def generate_preview(self, selected_frames: List[str], highlighted_frames: List[str], filter_name: str = "none", progress=None) -> str:
        """
        Generate a preview image from selected frames with optional highlighting and filtering.

        Args:
            selected_frames (List[str]): List of selected frame URLs.
            highlighted_frames (List[str]): List of highlighted frame URLs.
            filter_name (str, optional): The name of the filter to apply. Defaults to "none".
            progress (optional): Progress tracker object. Defaults to None.

        Returns:
            str: The path to the generated preview image.
        """
        # Use the VideoProcessor to generate the preview image with the specified parameters
        return self.processor.generate_preview(selected_frames, highlighted_frames, filter_name, progress)
