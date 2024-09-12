from flask import Flask
from concurrent.futures import ThreadPoolExecutor
import logging
import os

def create_app():
    app = Flask(__name__)

    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static/uploads')
    app.config['FRAMES_FOLDER'] = os.path.join(app.root_path, 'static/frames')
    app.config['OUTPUT_FOLDER'] = os.path.join(app.root_path, 'outputs')
    app.config['SERVER_NAME'] = '127.0.0.1:5000'

    logging.basicConfig(level=logging.INFO)

    for folder in [app.config['UPLOAD_FOLDER'], app.config['FRAMES_FOLDER'], app.config['OUTPUT_FOLDER']]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")

    executor = ThreadPoolExecutor(max_workers=4)

    from services.video_service import VideoService
    video_service = VideoService(app)

    from controllers.video_controller import create_video_controller
    video_controller = create_video_controller(video_service, executor)

    app.register_blueprint(video_controller)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
