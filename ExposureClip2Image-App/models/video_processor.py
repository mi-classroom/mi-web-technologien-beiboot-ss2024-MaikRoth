import cv2
import numpy as np
import os
import glob
import logging
import time
from flask import request
from typing import List, Optional

class VideoProcessor:
    def __init__(self, app):
        self.app = app

    def clear_frames_directory(self) -> None:
        """
        Clear all files in the frames directory.
        """
        try:
            files = glob.glob(os.path.join(self.app.config['FRAMES_FOLDER'], '*'))
            for f in files:
                os.remove(f)
            logging.info(f"Cleared frames directory: {self.app.config['FRAMES_FOLDER']}")
        except Exception as e:
            logging.error(f"Error clearing frames directory: {e}")

    def extract_frames(self, video_path: str, start_time: float, end_time: float, fps: float, root_url: str) -> List[str]:
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
        frame_interval = max(1, int(original_fps / fps)) 

        current_frame = 0
        frame_paths = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if start_frame <= current_frame <= end_frame and current_frame % frame_interval == 0:
                    frame_filename = f"frame_{current_frame}.jpg"
                    frame_path = os.path.join(self.app.config['FRAMES_FOLDER'], frame_filename)
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85]) 
                    frame_url = f"{root_url}static/frames/{frame_filename}"
                    frame_paths.append(frame_url)
                current_frame += 1
        except Exception as e:
            logging.error(f"Error extracting frames: {e}")
        finally:
            cap.release()

        logging.info(f"Extracted frames from {video_path}")
        return frame_paths

    def create_long_exposure_image(self, selected_frames: List[str], highlighted_frames: List[str]) -> str:
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
        exposure_factor = 1.5 

        for frame_url in selected_frames:
            frame_path = frame_url.replace(request.url_root, '').replace('/static/', 'static/')
            frame_full_path = os.path.join(self.app.root_path, frame_path)
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
                    highlighted_full_path = os.path.join(self.app.root_path, highlighted_path)
                    if os.path.exists(highlighted_full_path):
                        highlighted_img = cv2.imread(highlighted_full_path).astype(np.float32)
                        exposure_image = self.highlight_object(exposure_image, highlighted_img)

            exposure_image = np.clip(exposure_image, 0, 255).astype(np.uint8)
            timestamp = int(time.time())
            output_path = os.path.join(self.app.config['OUTPUT_FOLDER'], f'long_exposure_{timestamp}.jpg')

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            logging.info(f"Saving long exposure image to: {output_path}")
            cv2.imwrite(output_path, exposure_image)
            return output_path
        else:
            raise ValueError("No valid frames to process.")

    def highlight_object(self, exposure_image: np.ndarray, highlighted_img: np.ndarray) -> np.ndarray:
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

            alpha = 0.3  
            for c in range(3):
                exposure_image[:, :, c] = (1 - mask * alpha) * exposure_image[:, :, c] + mask * alpha * highlighted_img[:, :, c]
        return np.clip(exposure_image, 0, 255)

    def apply_filter(self, image: np.ndarray, filter_name: str) -> np.ndarray:
        if filter_name == "grayscale":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif filter_name == "sepia":
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            return cv2.transform(image, sepia_filter)
        elif filter_name == "invert":
            image_8bit = np.clip(image, 0, 255).astype(np.uint8)
            return cv2.bitwise_not(image_8bit)
        elif filter_name == "blur":
            return cv2.GaussianBlur(image, (15, 15), 0)
        elif filter_name == "sharpen":
            kernel = np.array([[0, -1, 0],
                               [-1, 5,-1],
                               [0, -1, 0]])
            return cv2.filter2D(image, -1, kernel)
        elif filter_name == "emboss":
            kernel = np.array([[2, 0, 0],
                               [0, -1, 0],
                               [0, 0, -1]])
            return cv2.filter2D(image, -1, kernel) + 128
        elif filter_name == "edge_detection":
            return cv2.Canny(image.astype(np.uint8), 100, 200)
        elif filter_name == "brightness":
            return cv2.convertScaleAbs(image, alpha=1, beta=50) 
        elif filter_name == "contrast":
            return cv2.convertScaleAbs(image, alpha=2.0, beta=0)  
        elif filter_name == "saturation":
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_image[..., 1] = cv2.add(hsv_image[..., 1], 50)  
            return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        elif filter_name == "posterize":
            image_8bit = np.clip(image, 0, 255).astype(np.uint8)
            return cv2.convertScaleAbs(image_8bit, alpha=0.5, beta=0)  
        elif filter_name == "solarize":
            image_8bit = np.clip(image, 0, 255).astype(np.uint8)
            return cv2.threshold(image_8bit, 128, 255, cv2.THRESH_BINARY_INV)[1] + image_8bit
        elif filter_name == "hdr":
            hdr = cv2.detailEnhance(image, sigma_s=12, sigma_r=0.15)
            return cv2.convertScaleAbs(hdr)
        elif filter_name == "sketch":
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            inv_gray_image = 255 - gray_image
            blur_image = cv2.GaussianBlur(inv_gray_image, (21, 21), 0)
            sketch_image = cv2.divide(gray_image, 255 - blur_image, scale=256)
            return cv2.cvtColor(sketch_image, cv2.COLOR_GRAY2BGR)
        else:
            return image 

    def generate_preview(self, selected_frames: List[str], highlighted_frames: List[str], filter_name: str = "none", progress=None) -> str:
        exposure_image = None
        valid_frame_count = 0
        exposure_factor = 1.5  

        def convert_url_to_path(url: str) -> str:
            return os.path.join(self.app.root_path, url.replace(request.url_root, '').replace('/static/', 'static/'))

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
                if progress is not None:
                    progress['value'] = int(((i + 1) / total_frames) * 100)
            if valid_frame_count > 0:
                exposure_image /= valid_frame_count

            if highlighted_frames:
                for highlighted_frame in highlighted_frames:
                    highlighted_path = convert_url_to_path(highlighted_frame)
                    if os.path.exists(highlighted_path):
                        highlighted_img = cv2.imread(highlighted_path).astype(np.float32)
                        exposure_image = self.highlight_object(exposure_image, highlighted_img)

            exposure_image = self.apply_filter(exposure_image, filter_name)

            if filter_name != "grayscale": 
                exposure_image = np.clip(exposure_image, 0, 255).astype(np.uint8)

            timestamp = int(time.time())
            output_folder = self.app.config['OUTPUT_FOLDER']
            output_path = os.path.join(output_folder, f'preview_{timestamp}.jpg')

            os.makedirs(output_folder, exist_ok=True)
            logging.info(f"Saving preview image to: {output_path}")

            if filter_name == "grayscale":
                cv2.imwrite(output_path, exposure_image)  
            else:
                cv2.imwrite(output_path, exposure_image)

            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Failed to create the preview image at {output_path}")

            return output_path

        except Exception as e:
            logging.error(f"Error generating preview: {e}")
            raise
