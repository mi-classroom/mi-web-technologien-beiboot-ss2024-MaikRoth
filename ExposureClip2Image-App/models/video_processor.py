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
        # Initialize the VideoProcessor with the Flask app instance
        self.app = app

    def clear_frames_directory(self) -> None:
        """
        Clear all files in the frames directory.
        """
        try:
            # Get all files in the frames directory
            files = glob.glob(os.path.join(self.app.config['FRAMES_FOLDER'], '*'))
            # Remove each file found
            for f in files:
                os.remove(f)
            logging.info(f"Cleared frames directory: {self.app.config['FRAMES_FOLDER']}")
        except Exception as e:
            # Log an error if something goes wrong
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
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        # Get the video's original frames per second
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        # Calculate the start and end frame numbers based on the times
        start_frame = int(start_time * original_fps)
        end_frame = int(end_time * original_fps)
        # Determine the interval between frames to extract
        frame_interval = max(1, int(original_fps / fps)) 

        current_frame = 0  # Initialize the current frame counter
        frame_paths = []   # List to store the paths of extracted frames

        try:
            while True:
                # Read the next frame from the video
                ret, frame = cap.read()
                if not ret:
                    # Break the loop if there are no more frames
                    break
                if start_frame <= current_frame <= end_frame and current_frame % frame_interval == 0:
                    # If the current frame is within the desired range and matches the interval
                    frame_filename = f"frame_{current_frame}.jpg"
                    frame_path = os.path.join(self.app.config['FRAMES_FOLDER'], frame_filename)
                    # Save the frame as a JPEG image with 85% quality
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85]) 
                    # Create the URL for accessing the frame
                    frame_url = f"{root_url}static/frames/{frame_filename}"
                    frame_paths.append(frame_url)
                # Increment the frame counter
                current_frame += 1
        except Exception as e:
            # Log any errors that occur during frame extraction
            logging.error(f"Error extracting frames: {e}")
        finally:
            # Release the video capture object
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
        exposure_image = None       # Initialize the exposure image
        valid_frame_count = 0       # Counter for valid frames processed
        exposure_factor = 1.5       # Factor to adjust exposure effect

        for frame_url in selected_frames:
            # Convert the frame URL to a local file path
            frame_path = frame_url.replace(request.url_root, '').replace('/static/', 'static/')
            frame_full_path = os.path.join(self.app.root_path, frame_path)
            if not os.path.exists(frame_full_path):
                # Log a warning if the frame file does not exist
                logging.warning(f"Frame file does not exist: {frame_full_path}")
                continue
            # Read the frame image and convert to float32 for calculations
            frame = cv2.imread(frame_full_path).astype(np.float32)
            if exposure_image is None:
                # Initialize the exposure image with zeros matching the frame size
                exposure_image = np.zeros_like(frame)
            # Add the frame to the exposure image with the exposure factor
            exposure_image += frame * exposure_factor
            valid_frame_count += 1

        if exposure_image is not None and valid_frame_count > 0:
            # Average the exposure image based on the number of valid frames
            exposure_image /= valid_frame_count

            if highlighted_frames:
                # If there are frames to highlight, process them
                for highlighted_frame in highlighted_frames:
                    # Convert the highlighted frame URL to a local file path
                    highlighted_path = highlighted_frame.replace(request.url_root, '').replace('/static/', 'static/')
                    highlighted_full_path = os.path.join(self.app.root_path, highlighted_path)
                    if os.path.exists(highlighted_full_path):
                        # Read the highlighted frame image
                        highlighted_img = cv2.imread(highlighted_full_path).astype(np.float32)
                        # Apply highlighting to the exposure image
                        exposure_image = self.highlight_object(exposure_image, highlighted_img)

            # Clip the pixel values to the valid range and convert to uint8
            exposure_image = np.clip(exposure_image, 0, 255).astype(np.uint8)
            # Generate a timestamp for the output filename
            timestamp = int(time.time())
            # Define the output path for the long exposure image
            output_path = os.path.join(self.app.config['OUTPUT_FOLDER'], f'long_exposure_{timestamp}.jpg')

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            logging.info(f"Saving long exposure image to: {output_path}")
            # Save the final exposure image
            cv2.imwrite(output_path, exposure_image)
            return output_path
        else:
            # Raise an error if no valid frames were processed
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
        # Create a background subtractor for detecting moving objects
        back_sub = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=50, detectShadows=False)
        # Apply the background subtractor to get the foreground mask
        fg_mask = back_sub.apply(highlighted_img)
        # Find contours in the foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the largest contour, assuming it's the main object
            largest_contour = max(contours, key=cv2.contourArea)
            # Create a mask for the largest contour
            mask = np.zeros_like(highlighted_img, dtype=np.uint8)
            cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            # Convert the mask to a single channel and normalize to [0,1]
            mask = mask[:, :, 0] / 255.0

            alpha = 0.3  # Transparency factor for blending
            # Blend the highlighted object into the exposure image
            for c in range(3):
                exposure_image[:, :, c] = (1 - mask * alpha) * exposure_image[:, :, c] + mask * alpha * highlighted_img[:, :, c]
        # Ensure the pixel values are within the valid range
        return np.clip(exposure_image, 0, 255)

    def apply_filter(self, image: np.ndarray, filter_name: str) -> np.ndarray:
        # Apply a specified filter to the image
        if filter_name == "grayscale":
            # Convert the image to grayscale
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif filter_name == "sepia":
            # Apply a sepia filter using a transformation matrix
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            return cv2.transform(image, sepia_filter)
        elif filter_name == "invert":
            # Invert the colors of the image
            image_8bit = np.clip(image, 0, 255).astype(np.uint8)
            return cv2.bitwise_not(image_8bit)
        elif filter_name == "blur":
            # Apply a Gaussian blur to the image
            return cv2.GaussianBlur(image, (15, 15), 0)
        elif filter_name == "sharpen":
            # Sharpen the image using a kernel
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            return cv2.filter2D(image, -1, kernel)
        elif filter_name == "emboss":
            # Apply an emboss effect using a kernel
            kernel = np.array([[2, 0, 0],
                               [0, -1, 0],
                               [0, 0, -1]])
            return cv2.filter2D(image, -1, kernel) + 128
        elif filter_name == "edge_detection":
            # Perform edge detection using the Canny algorithm
            return cv2.Canny(image.astype(np.uint8), 100, 200)
        elif filter_name == "brightness":
            # Increase the brightness of the image
            return cv2.convertScaleAbs(image, alpha=1, beta=50) 
        elif filter_name == "contrast":
            # Increase the contrast of the image
            return cv2.convertScaleAbs(image, alpha=2.0, beta=0)  
        elif filter_name == "saturation":
            # Increase the saturation in the HSV color space
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_image[..., 1] = cv2.add(hsv_image[..., 1], 50)  
            return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        elif filter_name == "posterize":
            # Reduce the number of colors in the image
            image_8bit = np.clip(image, 0, 255).astype(np.uint8)
            return cv2.convertScaleAbs(image_8bit, alpha=0.5, beta=0)  
        elif filter_name == "solarize":
            # Apply a solarize effect to the image
            image_8bit = np.clip(image, 0, 255).astype(np.uint8)
            return cv2.threshold(image_8bit, 128, 255, cv2.THRESH_BINARY_INV)[1] + image_8bit
        elif filter_name == "hdr":
            # Enhance the image details using HDR effect
            hdr = cv2.detailEnhance(image, sigma_s=12, sigma_r=0.15)
            return cv2.convertScaleAbs(hdr)
        elif filter_name == "sketch":
            # Create a sketch-like effect
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            inv_gray_image = 255 - gray_image
            blur_image = cv2.GaussianBlur(inv_gray_image, (21, 21), 0)
            sketch_image = cv2.divide(gray_image, 255 - blur_image, scale=256)
            return cv2.cvtColor(sketch_image, cv2.COLOR_GRAY2BGR)
        else:
            # Return the original image if no valid filter is specified
            return image 
        
    def apply_filter_to_image(self, image_path: str, filter_name: str) -> str:
        """
        Apply a specified filter to an existing image and save the result.

        Args:
            image_path (str): The path to the image to which the filter will be applied.
            filter_name (str): The name of the filter to apply.

        Returns:
            str: The path to the filtered image.
        """
        # Load the image from the specified path
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Apply the specified filter using the existing apply_filter method
        filtered_image = self.apply_filter(image, filter_name)

        # Generate a new filename for the filtered image
        filtered_image_path = os.path.join(self.app.config['OUTPUT_FOLDER'], f"filtered_{os.path.basename(image_path)}")

        # Save the filtered image
        cv2.imwrite(filtered_image_path, filtered_image)

        return filtered_image_path


    def generate_preview(self, selected_frames: List[str], highlighted_frames: List[str], filter_name: str = "none", progress=None) -> str:
        # Generate a preview image from selected frames with optional highlighting and filtering
        exposure_image = None       # Initialize the exposure image
        valid_frame_count = 0       # Counter for valid frames processed
        exposure_factor = 1.5       # Factor to adjust exposure effect

        def convert_url_to_path(url: str) -> str:
            # Convert a frame URL to a local file path
            return os.path.join(self.app.root_path, url.replace(request.url_root, '').replace('/static/', 'static/'))

        # Convert selected frame URLs to local file paths
        selected_frame_paths = [convert_url_to_path(frame_url) for frame_url in selected_frames]
        frames_to_process = selected_frame_paths 

        total_frames = len(frames_to_process)  # Total number of frames to process

        if total_frames == 0:
            # Raise an error if no frames are available
            logging.error("No frames available for processing.")
            raise ValueError("No frames available for processing.")

        try:
            for i, frame_path in enumerate(frames_to_process):
                if not os.path.exists(frame_path):
                    # Log an error if the frame path does not exist
                    logging.error(f"Frame path does not exist: {frame_path}")
                    continue

                # Read the frame image
                frame = cv2.imread(frame_path)
                if frame is None:
                    # Log an error if the frame cannot be read
                    logging.error(f"Failed to read frame: {frame_path}")
                    continue

                # Convert the frame to float32 for calculations
                frame = frame.astype(np.float32)

                if exposure_image is None:
                    # Initialize the exposure image with zeros matching the frame size
                    exposure_image = np.zeros_like(frame)

                # Add the frame to the exposure image with the exposure factor
                exposure_image += frame * exposure_factor
                valid_frame_count += 1

                if progress is not None:
                    # Update the progress if a progress tracker is provided
                    progress['value'] = int(((i + 1) / total_frames) * 100)

            if valid_frame_count > 0:
                # Average the exposure image based on the number of valid frames
                exposure_image /= valid_frame_count

            if highlighted_frames:
                # If there are frames to highlight, process them
                for highlighted_frame in highlighted_frames:
                    highlighted_path = convert_url_to_path(highlighted_frame)
                    if os.path.exists(highlighted_path):
                        # Read the highlighted frame image
                        highlighted_img = cv2.imread(highlighted_path).astype(np.float32)
                        # Apply highlighting to the exposure image
                        exposure_image = self.highlight_object(exposure_image, highlighted_img)

            # Apply the specified filter to the exposure image
            exposure_image = self.apply_filter(exposure_image, filter_name)

            if filter_name != "grayscale": 
                # Clip the pixel values to the valid range and convert to uint8
                exposure_image = np.clip(exposure_image, 0, 255).astype(np.uint8)

            # Generate a timestamp for the output filename
            timestamp = int(time.time())
            # Define the output path for the preview image
            output_folder = self.app.config['OUTPUT_FOLDER']
            output_path = os.path.join(output_folder, f'preview_{timestamp}.jpg')

            # Ensure the output directory exists
            os.makedirs(output_folder, exist_ok=True)
            logging.info(f"Saving preview image to: {output_path}")

            # Save the final preview image
            cv2.imwrite(output_path, exposure_image)

            if not os.path.exists(output_path):
                # Raise an error if the output file was not created
                raise FileNotFoundError(f"Failed to create the preview image at {output_path}")

            return output_path

        except Exception as e:
            # Log any exceptions that occur during processing
            logging.error(f"Error generating preview: {e}")
            raise
