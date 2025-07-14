import time
import cv2
import sys
import os
import threading
import pandas as pd
import numpy as np

from nxbt import Buttons
from nxbt import Sticks

from src import video_processing as vp
from src import macro as mcr
from src import init_controller
from src import navigation as nvg
from src import traverse_coordinate as tc
from src.log_entry import add_to_log, summarize_and_clear_log




def object_detect_by_hsv_mask(slot_img, hsv_lower, hsv_upper, count_lower, count_upper):

    # Apply color thresholding
    mask = cv2.inRange(slot_img, hsv_lower, hsv_upper)
    masked_count = cv2.countNonZero(mask)
    
    return( masked_count < count_upper and masked_count > count_lower)


def is_slot_empty(slot_image, threshold=0.95):
    """
    Determines whether a Pokémon storage slot is empty based on HSV color statistics.

    This function compares the HSV values in a target region of the slot image to a reference
    region (which is always empty). It classifies the slot as empty if a high percentage of
    pixels in the center region fall within a dynamically calculated HSV range.

    Parameters
    ----------
    slot_image : numpy.ndarray
        The image of the slot, in HSV color space.
    threshold : float, optional
        The minimum ratio of pixels that must fall within the empty slot HSV range
        to classify the slot as empty (default is 0.95).

    Returns
    -------
    bool
        True if the slot is classified as empty, False otherwise.

    Notes
    -----
    - The HSV range for an empty slot is determined from the top portion of the image,
      assuming it always contains a consistent background.
    - The central region is analyzed for matching pixels using `cv2.inRange`.
    """
    
    # Get image dimensions
    h, w, channels = slot_image.shape

    # Define upper reference region (always empty part)
    upper_slice_h = slot_image[0: int(h // 8), int(w//8):int(7*w//8)]

    # Compute mean and standard deviation of the upper reference region
    mean_hsv = np.mean(upper_slice_h.reshape(-1, 3), axis=0)
    std_hsv = np.std(upper_slice_h.reshape(-1, 3), axis=0)


    # Define the HSV range for an empty slot (mean ± 5 * standard deviation)
    lower_hsv = np.clip(mean_hsv - 5 * std_hsv, 0, 255).astype(np.uint8)
    upper_hsv = np.clip(mean_hsv + 5 * std_hsv, 0, 255).astype(np.uint8)

    # Define middle sprite detection region
    middle_slice_v = slot_image[int(h // 8): h, int(3 * w / 8): int(5 * w / 8)]

    # Use inRange function to find pixels within the empty slot HSV range
    mask = cv2.inRange(middle_slice_v, lower_hsv, upper_hsv)

    # Calculate the percentage of pixels that match the empty slot range
    matching_pixel_ratio = np.count_nonzero(mask) / mask.size

    # Classify as empty if the matching pixel ratio is above the threshold
    return matching_pixel_ratio > threshold
    
    

def detect_slot(fs):
    """
    Detects the status of all PC box slots using HSV-based classification.

    This function analyzes the PC box region from the current frame in the FrameStore and determines
    the contents of each slot using empty-slot detection and egg-color filtering. The result is a
    matrix that represents the current PC box layout in terms of slot content.

    Parameters
    ----------
    fs : FrameStore
        Frame container used to retrieve the current video frame for analysis.

    Returns
    -------
    numpy.ndarray
        A 5x6 matrix representing the status of each PC box slot:
        - -1 : Slot contains an egg
        -  0 : Slot is empty
        -  1 : Slot contains a non-egg Pokémon (default assumption)

    Notes
    -----
    - The leftmost column (column 0) corresponds to the party slots, not part of the PC box.
    - The actual detected grid is 6 rows x 7 columns, but only a 5x6 region corresponds
      to the real PC box layout.
    - The returned matrix is composed by:
        - Extracting the bottom 5 rows of the first column (party slots),
        - Combining them with the top 5 rows of the remaining columns (PC box slots).
    - Egg detection is based on a specific HSV mask range and pixel intensity threshold.
    - Empty slot detection is performed via `is_slot_empty()`, using region-based HSV statistics.
    """

    r, c = 5, 6
    frame = fs.get_frame()[1]
    sub_frame_PC, xywh_PC = vp.get_roi_by_name(frame, "PC_Box", vp.roi_data, True)
    x, y, w, h = xywh_PC

    # Compute the width and height of each slot within PC_Box
    slot_w = w // c
    slot_h = h // r
    
    status = np.ones((r+1,c+1))  # Store Egg slot positions

    # Define the refined Egg HSV color range
    lower_egg = np.array([0, 0, 181])  
    upper_egg = np.array([176, 57, 255])  

    # Iterate through each slot (6x7 grid)
    for row in range(r+1):  
        for col in range(c+1):
            
            if col == 0:
                slot_x = (x // 1.65)
            else:
                # Get the slot's bounding box
                slot_x= x + (col-1) * slot_w
                
            slot_y =  y + row * slot_h
            # Crop the slot image
            slot_img = frame[int(slot_y):int(slot_y + slot_h), int(slot_x):int(slot_x + slot_w)]
            

            if is_slot_empty(slot_img):
                status[row, col] = 0
            if object_detect_by_hsv_mask(slot_img, lower_egg, upper_egg, 210, 245):
                status[row, col] = -1  # Egg detected
                
    # Extract the two submatrices
    submatrix_first_col = status[-5:, 0].reshape(-1, 1)  # Bottom 5 rows of first column
    submatrix_rest_cols = status[:5, 1:]  # Top 5 rows of the rest of the columns

    # Combine them horizontally
    combined_matrix = np.hstack((submatrix_first_col, submatrix_rest_cols))
    return combined_matrix



def get_status(fs):
    """
    Returns a column-wise summary of slot statuses in the current PC box layout.

    This function waits until the BOXES interface is active, then scans the PC box using
    HSV-based slot classification and counts the number of occupied, egg, and empty slots
    per column.

    Parameters
    ----------
    fs : FrameStore
        Frame container used to retrieve the current video frame for slot detection.

    Returns
    -------
    dict
        A dictionary summarizing slot counts **per column**:
        - "occupied": Number of non-egg Pokémon per column (value == 1)
        - "eggs": Number of egg slots per column (value == -1)
        - "empty": Number of empty slots per column (value == 0)

    Notes
    -----
    - Column-wise sums are intentional: the first column corresponds to party slots,
      which are filled in batches of 5 eggs.
    - Status detection is based on `detect_slot()` applied to the current HSV frame.
    - This information is typically used to decide how many eggs to move per batch
      during automated hatching.
    """
    while not nvg.BOXES.identifier(fs):
        # if the Boxes interface is not detected, we simply need to wait a bit most of the time
        time.sleep(2)
        if nvg.BOXES.identifier(fs):
            break
        # if waiting doesn't solve the problem, then we reinitialize
        else:
            initialize(fs, ctrler, nx)
            
    current_status = detect_slot(fs)
    count_1 = np.sum(current_status == 1, axis=0)
    count_neg1 = np.sum(current_status == -1, axis=0)
    count_0 = np.sum(current_status == 0, axis=0)
    
    return({'occupied': count_1, "eggs": count_neg1, "empty": count_0})



    
def initialize(fs, ctrler, nx, refresh = False):
    """
    Prepares the environment by navigating to the BOXES interface and resetting the selection cursor.

    This function ensures the current interface is set to BOXES and moves the selection
    to the top-left corner of the PC box grid (position [0, 0]).

    Parameters
    ----------
    fs : FrameStore
        Frame container used for interface detection and slot positioning.
    ctrler : str
        NXBT controller ID.
    nx : Nxbt instance
        NXBT controller interface used for sending macros.
    refresh: bool
        when True, go to the main_menu first regardless
    """
    
    nvg.go_to("BOXES", fs, ctrler, nx, refresh)
    tc.box_go_to([0,0], fs, ctrler, nx)

class Hatched_State:
    """
    Tracks the state of the current hatching process.

    Attributes
    ----------
    checked_boxes : int
        Number of boxes navigated through so far.
    termination_reason : str
        Message indicating why hatching was terminated, if any.
    to_be_hatched_party : int
        Number of eggs currently being hatched in party.
    total_hatched : int
        Total eggs hatched in this pipeline run.
    time_since_last_hatch : float
        Time (in seconds) since last egg hatched.
    """
    def __init__(self):
        self.checked_boxes = 0
        self.termination_reason = "still running"
        self.to_be_hatched_party = 0
        self.total_hatched = 0
        self.starting_time = time.time()
        self.time_stamp_last_hatch = time.time()
        self.restart_count = 0
    def record_hatch(self):
        self.to_be_hatched_party -= 1
        self.total_hatched += 1
        self.time_stamp_last_hatch = time.time()
    @property
    def time_since_last_hatch(self):
        return(time.time() - self.time_stamp_last_hatch)
        
        
def analyze_current_party_situation(fs, hatch_state, ctrler, nx):
    
    # first we make sure we are in the boxes interface with the cursor at (0,0)
    initialize(fs, ctrler, nx, True)
    
    status = get_status(fs)
    if status["empty"][0] == 5:
        party_status = "empty"
    elif status["occupied"][0] > 0 and status["eggs"][0] == 0:
        party_status = "occupied_no_eggs"
    elif status["eggs"][0] > 0:
        party_status = "eggs_present"
        hatch_state.to_be_hatched_party = status["eggs"][0]
    return(party_status)


def find_box_with_column_type(column_type, fs, hatch_state, ctrler, nx, max_scroll=32):
    """
    Searches for a box that has a column with either empty slots or eggs.

    Parameters
    ----------
    fs : FrameStore
        Frame container used to detect current box status.
    hatch_state : Hatched_State
        Tracks scroll count and state.
    ctrler : str
        NXBT controller ID.
    nx : Nxbt instance
        Controller interface to send inputs.
    column_type : str
        Either "empty" or "egg". Determines what kind of column to look for.
    max_scroll : int
        Maximum number of scrolls before giving up.

    Returns
    -------
    int or None
        Column index where the desired column was found,
        or None if not found.
    """
    status = get_status(fs)
    if column_type == "empty":
        direction = -1
        button = Buttons.L
        col_check = (status["empty"] == 5)[1:]
    elif column_type == "egg":
        direction = 1
        button = Buttons.R
        col_check = (status["eggs"] > 0)[1:]
    else:
        raise ValueError(f"Invalid column_type: {column_type}. Use 'empty' or 'egg'.")

    scroll_count = 0

    while not any(col_check) and scroll_count < max_scroll:
        nx.press_buttons(ctrler, button, 0.1, 0.2)
        hatch_state.checked_boxes += direction
        time.sleep(1)

        status = get_status(fs)
        if column_type == "empty":
            col_check = (status["empty"] == 5)[1:]
        else:
            col_check = (status["eggs"] > 0)[1:]

        scroll_count += 1

    if not any(col_check):
        hatch_state.termination_reason = f"No {column_type} column found after {scroll_count} scrolls"
        return None

    col_index = col_check.tolist().index(True)

    if column_type == "egg":
        count = get_status(fs)["eggs"][1:][col_index]
        hatch_state.to_be_hatched_party = count

    return col_index


def fix_occupied_no_eggs(fs, hatch_state, ctrler, nx):
    # In this case, the party is occupied by non-eggs. 
    # We need to find a box with an empty column, empty the party, and then search for a box with at least one egg
    add_to_log("Removing the non-egg occupants in the party.")
    # First we find a box with an empty column
    empty_col_at = find_box_with_column_type("empty", fs, hatch_state, ctrler, nx)
    # Load the party to the empty column
    nx.macro(ctrler, mcr.return_hatchlings_to_col_i(empty_col_at))
    
def fix_eggs_present(fs, hatch_state, ctrler, nx):
    # In this case, there are eggs in the party
    # We need to hatch the eggs first, and then the problem becomes "occupied by non-eggs"
    add_to_log("Hatching the eggs in the party and then removing the hatchlings.")
    hatch_eggs_one_batch(fs, hatch_state, ctrler, nx)
    initialize(fs, ctrler, nx, True)
    fix_occupied_no_eggs(fs, hatch_state, ctrler, nx)
    

def party_standardized(fs, hatch_state, ctrler, nx):
    # having an empty party will be the standard state
    current_state = analyze_current_party_situation(fs, hatch_state, ctrler, nx)
    add_to_log(f"Situation {current_state} detected.")
    while current_state != "empty" and hatch_state.termination_reason == "still running":
        
        if current_state == "occupied_no_eggs":
            fix_occupied_no_eggs(fs, hatch_state, ctrler, nx)
        elif current_state == "eggs_present":
            fix_eggs_present(fs, hatch_state, ctrler, nx)
            
        current_state = analyze_current_party_situation(fs, hatch_state, ctrler, nx)
        
        
    if hatch_state.termination_reason == "still running":
        add_to_log("Party successfully standardized.")
        return True
    else:
        return False
    
        
# if party_standardized(): then we load eggs to party
def load_eggs_to_party(fs, hatch_state, ctrler, nx):
    # First we make sure we are in the box interface
    initialize(fs, ctrler, nx)
    # After a successful standardization, we try to load the eggs
    location = find_box_with_column_type("egg", fs, hatch_state, ctrler, nx)
    
    if location is None:
        return
    
    expected_egg_count = hatch_state.to_be_hatched_party
    add_to_log(f"{expected_egg_count} eggs will be loaded to party.")
    
    nx.macro(ctrler, mcr.load_eggs_from_col_i(location))
    # Check the current party 
    current_party_eggs = (get_status(fs)["eggs"] == expected_egg_count)[0]
    
    retries = 0
    max_retries = 20
    while not current_party_eggs and retries < max_retries:
        
        initialize(fs, ctrler, nx, True)
        location = find_box_with_column_type("egg", fs, hatch_state, ctrler, nx)
        
        expected_egg_count = hatch_state.to_be_hatched_party
        
        if location is None:
            break
        # When eggs are already in the party, loading may fail visually, causing overlapping sprites.
        # This can cause get_status() to falsely report no eggs in the party.
        # initialize() resets the view so get_status() reflects the correct state.
        current_party_eggs = (get_status(fs)["eggs"] == expected_egg_count)[0]
        if current_party_eggs:
            break
        
        retries += 1
        add_to_log(f"Retry no.{retries}: detected wrong number of eggs in party.")
        
        time.sleep(0.5)
        nx.macro(ctrler, mcr.load_eggs_from_col_i(location))
        time.sleep(1)

    if retries == max_retries:
        hatch_state.termination_reason = "Failed to load eggs to party after multiple attempts"
    else:
        add_to_log(f"{hatch_state.to_be_hatched_party}/{expected_egg_count} eggs are loaded to the party.")
        

def hatch_eggs_one_batch(fs, hatch_state, ctrler, nx):
    
    # Exit the boxes
    nvg.go_to("OVERWORLD", fs, ctrler, nx)
    add_to_log("Moved to overworld.")
    while hatch_state.to_be_hatched_party > 0:
        nx.macro(ctrler, mcr.macro_run_around)
        
        if time.time() - hatch_state.time_stamp_last_hatch > 240:
            hatch_state.restart_count += 1
            hatch_state.time_stamp_last_hatch = time.time()
            add_to_log(f"Long hatching time detected, will be restarting the {hatch_state.restart_count}th time.")
            break

        if "Oh" in fs.get_convo():
            nx.press_buttons(ctrler, Buttons.A, 0.1, 0.9)
            
        if "hatched" in fs.get_convo(): 
            nx.press_buttons(ctrler, Buttons.A, 0.1, 0.9)
            interval = hatch_state.time_since_last_hatch
            hatch_state.record_hatch()
            msg = f"Egg {hatch_state.total_hatched} has been hatched. It's been {int(interval)}s since the previous egg hatched."
            add_to_log(msg)
        if hatch_state.to_be_hatched_party == 0:
            hatch_state.restart_count = 0

    
        
        

def hatch_pipeline(fs, hatch_state, ctrler, nx):
    """
    Runs the full egg hatching loop, cycling through boxes until all eggs are hatched
    or until hatch_state.termination_reason is triggered.

    Parameters
    ----------
    fs : FrameStore
    hatch_state : Hatched_State
    ctrler : str
    nx : Nxbt instance
    """
    max_boxes = 32

    while hatch_state.checked_boxes < max_boxes and hatch_state.termination_reason == "still running":
        
        is_standardized = party_standardized(fs, hatch_state, ctrler, nx)
        if not is_standardized:
            break

        load_eggs_to_party(fs, hatch_state, ctrler, nx)
        if hatch_state.termination_reason != "still running":
            break
        
        hatch_eggs_one_batch(fs, hatch_state, ctrler, nx)
        
        time_elapsed = time.time() - hatch_state.starting_time
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(time_elapsed))
        speed = 3600 * hatch_state.total_hatched / time_elapsed
        add_to_log(f"Current Speed: {round(speed, 2)} eggs/hr | Time Elapsed: {formatted_time}")
        
        if hatch_state.restart_count == 10:
            hatch_state.termination_reason = "Too many long hatch time restarts. "
            break

    add_to_log(f"Pipeline complete. Total eggs hatched: {hatch_state.total_hatched}")

def restart_script():
    add_to_log("Fatal Error occured. Restarting script...")
    python = sys.executable  # Path to current Python interpreter
    os.execv(python, [python] + sys.argv)
    

    
if __name__ == "__main__":
    
    stop_event = threading.Event()
    try:
        nx, ctrler = init_controller.nx_init()
        time.sleep(1)
        print("Controller Ready")

        cap = vp.make_video_capture()
        fs = vp.FrameStore()

        video_thread = threading.Thread(target = vp.update_video_feed, 
                                        args = (cap, fs, vp.roi_data), 
                                        kwargs={'ocr_threshold': 70, 'display': True, 'stop_event': stop_event},
                                        daemon=True)
        video_thread.start()
        time.sleep(1)
        summarize_and_clear_log()
        add_to_log("Start Hatching.")
        hatch_state = Hatched_State()
        
        hatch_pipeline(fs, hatch_state, ctrler, nx)
        # add_to_log(hatch_state.termination_reason)
        
    except Exception as e:
        add_to_log(f"[Fatal Error] {repr(e)}")  # Use repr() to get full trace-friendly string
        restart_script()

    finally:
        stop_event.set()
        if 'video_thread' in locals():  # Ensure the thread exists before joining
            video_thread.join()