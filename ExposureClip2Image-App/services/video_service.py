from models.video_processor import VideoProcessor
from typing import List, Tuple
import json
import logging

class VideoService:
    def __init__(self, app):
        self.app = app
        self.processor = VideoProcessor(app)

    def parse_form_data(self, form_data) -> Tuple[float, float, float]:
        try:
            start_time = float(form_data['start_time'])
            end_time = float(form_data['end_time'])
            fps = float(form_data['fps'])
            return start_time, end_time, fps
        except (ValueError, KeyError) as e:
            logging.error(f"Error parsing form data: {e}")
            raise ValueError("Invalid form data.")

    def save_video(self, video_file) -> str:
        from werkzeug.utils import secure_filename
        import os

        if not video_file or not video_file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            logging.error("Invalid video file format.")
            raise ValueError("Invalid video file format. Only .mp4, .avi, .mov, .mkv are allowed.")

        filename = secure_filename(video_file.filename)
        video_path = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)

        logging.info(f"Uploaded video to {video_path}")
        return video_path

    def process_video(self, video_path: str, start_time: float, end_time: float, fps: float, root_url: str) -> List[str]:
        return self.processor.extract_frames(video_path, start_time, end_time, fps, root_url)

    def create_long_exposure(self, selected_frames_json: str, highlighted_frames_json: str) -> str:
        try:
            selected_frames = json.loads(selected_frames_json)
            highlighted_frames = json.loads(highlighted_frames_json) if highlighted_frames_json else []
            if not selected_frames:
                logging.error("No frames selected for processing.")
                raise ValueError("No frames selected for processing.")
            return self.processor.create_long_exposure_image(selected_frames, highlighted_frames)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            raise

    def generate_preview(self, selected_frames: List[str], highlighted_frames: List[str], filter_name: str = "none", progress=None) -> str:
        return self.processor.generate_preview(selected_frames, highlighted_frames, filter_name, progress)
