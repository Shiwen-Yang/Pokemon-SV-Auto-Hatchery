from __future__ import annotations
import cv2
import os
import pytesseract
import time
import numpy as np
import pandas as pd
import threading
from typing import Optional, Tuple

CSV_FILENAME = r"src/roi_data.csv"
roi_data = pd.read_csv(CSV_FILENAME)

def make_video_capture(device: int = 0, width: int = 720, height: int = 480, fps: int = 30):
    """
    Initializes and configures a video capture stream using OpenCV.

    This function sets up a video capture device with the specified resolution, 
    frame rate, and codec settings. It uses the V4L2 backend and forces the YUYV codec.

    Parameters:
    ----------
    device : int, optional
        Index of the video capture device (default is 0).
    width : int, optional
        Desired width of the video frames in pixels (default is 720).
    height : int, optional
        Desired height of the video frames in pixels (default is 480).
    fps : int, optional
        Desired frames per second of the capture stream (default is 30).

    Returns:
    -------
    cap : cv2.VideoCapture
        An OpenCV VideoCapture object that can be used to read frames.

    Notes:
    -----
    - The actual resolution, FPS, and codec may differ depending on hardware support.
    - Prints the applied settings to help with debugging and device configuration.
    """
    
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)  # Use Linux backend

    # Set YUYV codec FIRST
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Set FPS (must match supported intervals)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Validate if settings were applied
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4)])

    print(f"Width: {actual_width}, Height: {actual_height}, FPS: {actual_fps}, FOURCC: {fourcc_str}")

    return cap


def capture_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    """
    Captures a single frame from the video capture device.

    Parameters:
    ----------
        cap: An open cv2.VideoCapture object.

    Returns:
    -------
        The captured frame as a numpy array, or None if capturing fails.
    """
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        return None
    return frame

def show_frame(frame: np.ndarray, window_name: str = 'Frame', wait_time: int = 1) -> None:
    """
    Displays a frame in a window.

    Parameters:
    ----------
        frame: The image/frame to display.
        window_name: Name of the window (default: 'Frame')
        wait_time: Time in milliseconds to wait (default: 1)
    """
    cv2.imshow(window_name, frame)
    cv2.waitKey(wait_time)

def save_snapshot(frame: np.ndarray, filename: str) -> None:
    """
    Saves the current frame to a file.

    Parameters:
    ----------
        frame: The frame to save.
        filename: The filename or path to save the snapshot.
    """
    cv2.imwrite(filename, frame)
    
def save_roi_to_csv(name, roi):
    """
    Appends a Region of Interest (ROI) entry to a CSV file.

    This function logs the name and coordinates of an ROI into a CSV file. 
    If the file does not exist, it is created with headers. Otherwise, the 
    new entry is appended without headers.

    Parameters:
    ----------
    name : str
        A unique identifier or label for the ROI.
    roi : tuple or list
        A 4-element tuple or list representing the ROI in the format 
        (X, Y, Width, Height).

    Notes:
    -----
    - The output file path is defined by the global constant `CSV_FILENAME`.
    - Each row in the CSV file corresponds to one ROI entry with columns: 
      ["Name", "X", "Y", "Width", "Height"].
    """
    df = pd.DataFrame([[name, roi[0], roi[1], roi[2], roi[3]]], 
                      columns=["Name", "X", "Y", "Width", "Height"])
    
    if not os.path.exists(CSV_FILENAME):
        df.to_csv(CSV_FILENAME, index=False)  # Create new file with headers
    else:
        df.to_csv(CSV_FILENAME, mode='a', header=False, index=False)  
        
        
def get_roi_by_name(frame, roi_name, roi_data, coordinate = False):
    
    """
    Extracts a Region of Interest (ROI) from the given frame using its name.

    This function looks up the coordinates of a named ROI from a DataFrame and 
    crops the corresponding region from the provided image frame.

    Parameters:
    ----------
    frame : numpy.ndarray
        The image frame (typically from cv2.VideoCapture) to extract the ROI from.
    roi_name : str
        The name/label of the ROI to extract.
    roi_data : pandas.DataFrame
        A DataFrame containing ROI entries with columns ["Name", "X", "Y", "Width", "Height"].
    coordinate : bool, optional
        If True, the function also returns the coordinates of the ROI in the format (x, y, w, h).

    Returns:
    -------
    roi_frame : numpy.ndarray or None
        The cropped image corresponding to the ROI, or None if the ROI was not found.
    coordinates : tuple, optional
        The (x, y, w, h) coordinates of the ROI, only returned if `coordinate=True`.

    Notes:
    -----
    - If the specified ROI name is not found in `roi_data`, the function prints a warning and returns None.
    - Ensure that `roi_data` is loaded correctly and includes the required columns.
    """
    
    required_cols = {"Name", "X", "Y", "Width", "Height"}
    if not required_cols.issubset(roi_data.columns):
        raise ValueError(f"roi_data must contain columns: {required_cols}")
    
    # Find the row with the matching ROI name
    roi_row = roi_data[roi_data["Name"] == roi_name]

    if roi_row.empty:
        print(f"ROI '{roi_name}' not found.")
        return None

    # Extract ROI coordinates
    x, y, w, h = int(roi_row["X"].values[0]), int(roi_row["Y"].values[0]), \
                    int(roi_row["Width"].values[0]), int(roi_row["Height"].values[0])

    # Crop the ROI from the frame
    roi_frame = frame[y:y+h, x:x+w]

    if coordinate:
        return roi_frame, (x, y, w, h)
    
    return roi_frame

def binarize_img(image):
    """
    Converts an image to a clean binarized version using grayscale conversion, 
    resizing, Gaussian blur, Otsu thresholding, and median filtering.

    This preprocessing pipeline is designed to enhance features in the image for OCR
    by reducing noise and improving contrast.

    Steps:
    ------
    1. Convert to grayscale.
    2. Resize with scaling factors (fx=8, fy=6) using cubic interpolation.
    3. Apply Gaussian blur to reduce noise.
    4. Apply Otsu's thresholding to binarize.
    5. Apply median blur for additional denoising.

    Parameters:
    ----------
    image : numpy.ndarray
        Input image in BGR format (as read by OpenCV).

    Returns:
    -------
    binarized_image : numpy.ndarray
        The final binarized (black & white) image as a 2D array.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=8, fy=6, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (9,9), 0)  # Reduce noise
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Binarization
    gray = cv2.medianBlur(gray, 3)  # Further denoise
    return(gray)

def get_text(image, erode = True, threshold=60, config='--oem 1 --psm 6'):
    
    """
    Performs OCR (Optical Character Recognition) on an image and returns extracted text 
    only if the average confidence score meets or exceeds a specified threshold.

    The function uses Tesseract OCR via pytesseract and includes preprocessing steps 
    (binarization and optional erosion) to improve recognition accuracy.

    Parameters:
    ----------
    image : numpy.ndarray
        The input image (can be grayscale or BGR color image).
    erode : bool, optional
        Whether to apply morphological erosion to reduce noise and improve text separation (default is True).
    threshold : float, optional
        Minimum average confidence score required to accept the OCR result (default is 60).
    config : str, optional
        Custom configuration string for Tesseract OCR engine (default is '--oem 1 --psm 6').
        - `--oem` specifies the OCR Engine Mode (1 = LSTM neural net only).
        - `--psm` specifies the Page Segmentation Mode (6 = Assume a single uniform block of text).

    Returns:
    -------
    result_text : str
        The recognized text from the image, or an empty string if confidence is below threshold.
    avg_confidence : float
        The average confidence score of the recognized text.

    Notes:
    -----
    - Uses LSTM-based OCR engine (`--oem 1`) with paragraph-level text segmentation (`--psm 6`).
    - Text and confidence scores are extracted from Tesseract's detailed output (image_to_data).
    - Erosion is helpful for separating connected characters, especially when using resized or binarized images.
    """

    gray = binarize_img(image)

    # # Apply Adaptive Thresholding (recommended for OCR)
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    if erode:
        # #  Apply Erosion to remove small noise
        kernel = np.ones((6, 2), np.uint8)  # Adjust kernel size for different effects
        gray = cv2.erode(gray, kernel, iterations=1)  # Erodes white areas, separating merged text

    # Get OCR data with confidence scores
    data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
    
    # Extract words and their confidence scores
    words = data["text"]
    
    # confidences = [int(data["conf"][i]) for i in range(len(words)) if words[i].strip()]
    confidences = []
    for conf, word in zip(data["conf"], data["text"]):
        if word.strip():
            try:
                c = int(conf)
                if c >= 0:  # Filter out Tesseract's "-1" placeholders
                    confidences.append(c)
            except ValueError:
                continue
            
    if not confidences:  # If no valid words detected
        return "", 0.0
    
    # Compute the average confidence
    avg_confidence = sum(confidences) / len(confidences)
    
    # Return OCR text if confidence meets the threshold, else return empty string
    ocr_text = " ".join([words[i] for i in range(len(words)) if words[i].strip()])
    
    return ocr_text if avg_confidence >= threshold else "", avg_confidence

class FrameStore:
    """
    A thread-safe container to store the current frame, its HSV version,
    and the latest recognized conversation text.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.cf = None        # Current BGR frame
        self.cf_hsv = None    # HSV version of current frame
        self.convo = None     # OCR-detected conversation text

    def update_frame(self, frame):
        """Update the current frame and its HSV version."""
        with self._lock:
            self.cf = frame
            self.cf_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def update_convo(self, convo_text):
        """Update the conversation text."""
        with self._lock:
            self.convo = convo_text

    def get_frame(self):
        """Safely retrieve the current frame and HSV frame."""
        with self._lock:
            return self.cf, self.cf_hsv

    def get_convo(self):
        """Safely retrieve the conversation text."""
        with self._lock:
            return self.convo


def update_video_feed(cap, frame_store, roi_data=roi_data, ocr_threshold=70, display=True, stop_event = None):
    """
    Continuously captures and processes video frames, updates the FrameStore,
    and optionally displays the video stream. Also performs OCR to monitor conversation.

    Parameters:
    ----------
    cap : cv2.VideoCapture
        Video capture object.
    frame_store : FrameStore
        Thread-safe frame and OCR state container.
    roi_data : pd.DataFrame
        ROI DataFrame containing coordinates for "Conversation" ROI.
    ocr_threshold : int, optional
        Minimum average confidence required to update conversation text (default is 70).
    display : bool, optional
        If True, the video feed will be displayed in a window using OpenCV (default is True).
    """
    time_prev = time.time()

    while True:
        frame = capture_frame(cap)
        if stop_event and stop_event.is_set():
            break

        if frame is not None:
            # Update FrameStore
            frame_store.update_frame(frame)

            # OCR conversation extraction every 0.5 seconds
            time_current = time.time()
            if time_current - time_prev > 0.5:
                convo_roi = get_roi_by_name(frame, "Conversation", roi_data)
                current_convo, conf = get_text(convo_roi)

                prev_convo = frame_store.get_convo()
                if (prev_convo != current_convo and conf > ocr_threshold) or current_convo == "":
                    frame_store.update_convo(current_convo)

                time_prev = time_current

            # Show frame if display is enabled
            if display:
                cv2.imshow("Live Video Feed", frame)

        if display:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if display:
        cv2.destroyAllWindows()


    


def select_roi(frame):
    """
    Allows the user to select a Region of Interest (ROI) from the given frame and save it.

    The selected ROI is defined using a drag-box and is saved to a CSV file along with 
    a user-provided name. Useful for defining ROIs for future automation tasks.

    Parameters:
    ----------
    frame : numpy.ndarray
        The image frame to display and select ROI from.

    Returns:
    -------
    None

    Notes:
    -----
    - Uses OpenCV's `cv2.selectROI` for interactive ROI selection.
    - The ROI is saved using the `save_roi_to_csv` function.
    - Only valid ROIs (non-zero width and height) are saved.
    - Press ENTER or SPACE after selecting the ROI to confirm.
    """
    roi = cv2.selectROI("Video Stream", frame, fromCenter=False, showCrosshair=True)
    if roi[2] > 0 and roi[3] > 0:  # Ensure valid selection
        roi_name = input("Enter name for this ROI: ")
        save_roi_to_csv(roi_name, roi)
        print(f"ROI '{roi_name}' saved.")
    cv2.destroyWindow("ROI selector")
    
def wait_for_frame(frame, tol = 20):
    t = 0
    while frame is None and t < tol:
        print("waiting")
        time.sleep(0.5)
        t += 0.5/tol


if __name__ == "__main__":
    cap = make_video_capture(0, 720, 480, 30)

    # Create FrameStore instance
    frame_store = FrameStore()
    roi_selected = False
    roi = None

    # Load ROI data if needed
    try:
        roi_data = pd.read_csv(CSV_FILENAME)
    except FileNotFoundError:
        roi_data = pd.DataFrame(columns=["Name", "X", "Y", "Width", "Height"])
        print(f"ROI CSV file '{CSV_FILENAME}' not found. Starting with empty ROI data.")

    # Define mouse callback that uses FrameStore instead of global hsv
    def wrapped_get_hsv_value(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            _, hsv = frame_store.get_frame()
            if hsv is not None:
                pixel = hsv[y, x]
                print(f"HSV Value at ({x}, {y}): {pixel}")

    # Set the mouse callback (do it once now that window will exist)
    cv2.namedWindow("Video Stream")
    cv2.setMouseCallback("Video Stream", wrapped_get_hsv_value, None)

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Update FrameStore (BGR and HSV internally updated here)
        frame_store.update_frame(frame)

        # Show main video feed
        cv2.imshow("Video Stream", frame)

        # Show selected ROI if any
        if roi_selected:
            x, y, w, h = roi
            roi_frame = frame[y:y+h, x:x+w]
            cv2.imshow("Selected ROI", roi_frame)

        key = cv2.waitKey(1) & 0xFF

        # Press 'r' to select ROI
        if key == ord('r'):
            select_roi(frame)

        # Press 'c' to close the ROI window
        if key == ord('c') and roi_selected:
            cv2.destroyWindow("Selected ROI")
            roi_selected = False

        # Press 'q' to quit
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
