from flask import Flask
from concurrent.futures import ThreadPoolExecutor
from utils import file_helper
import logging
import os

def create_app() -> Flask:
    # Initialize the Flask application
    app = Flask(__name__)

    # Configure paths for various directories used by the app
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static/uploads')
    app.config['FRAMES_FOLDER'] = os.path.join(app.root_path, 'static/frames')
    app.config['OUTPUT_FOLDER'] = os.path.join(app.root_path, 'outputs')
    app.config['SERVER_NAME'] = '127.0.0.1:5000'

    # Ensure that the necessary directories exist
    file_helper.ensure_directories_exist([
        os.path.join(app.root_path, 'static/uploads'),
        os.path.join(app.root_path, 'static/frames'),
        os.path.join(app.root_path, 'outputs')
    ])

    # Set up basic logging configuration
    logging.basicConfig(level=logging.INFO)
    
    # Create a ThreadPoolExecutor for handling asynchronous tasks
    executor = ThreadPoolExecutor(max_workers=4)

    # Import and initialize the VideoService with the app configuration
    from services.video_service import VideoService
    video_service = VideoService(app)

    # Import and create the video controller, passing in the VideoService and executor
    from controllers.video_controller import create_video_controller
    video_controller = create_video_controller(video_service, executor)

    # Register the video controller blueprint with the app
    app.register_blueprint(video_controller)

    return app

if __name__ == '__main__':
    # Create the Flask app and run it in debug mode
    app = create_app()
    app.run(debug=True)
