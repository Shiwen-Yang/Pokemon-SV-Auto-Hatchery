"""
traverse_coordinate.py

Utility functions for navigating grid-based UI coordinates in Pokémon Scarlet and Violet.

This module provides tools to:
- Analyze the current cursor position in the PC Box grid.
- Calculate directional movement needed to reach a destination coordinate.
- Convert movement instructions into NXBT-compatible macros for automated controller input.

Functions
---------
analyze_selection_box() : Maps a bounding box (from OCR or highlight detection) to a grid position.
move_to() : Computes directional steps needed to move from one grid coordinate to another.
movements_to_nxbt_macro() : Converts movement steps into a formatted NXBT macro string.
box_go_to() : High-level utility that moves the selection box to a specific (row, col) destination.
"""

from src import video_processing as vp
from src import navigation as nvg
from src.log_entry import add_to_log
import time

def analyze_selection_box(xywh, fs):
    
    """
    Converts a bounding box coordinate into a grid position within the PC box UI.

    This function takes the absolute bounding box of the current selection and maps it
    to a (row, column) index based on the PC box layout (6 columns x 5 rows), using the
    "PC_Box" ROI as a reference.

    Parameters
    ----------
    xywh : tuple
        A tuple (x, y, w, h) representing the absolute coordinates of the selection box.
        Typically, this is the output from `get_current_selection_box()` when
        `subframe_name="Entire_Box"`.
    fs : FrameStore
        Frame container used to retrieve the current frame for ROI analysis.

    Returns
    -------
    list[int]
        A list [row, column] indicating the selection's grid position in the PC box.
    """

    position = [0,0]
    
    x,y,w,h = xywh
    x, y = x+w/2, y+h/2
    
    subframe, coord_PC_BOX = vp.get_roi_by_name(fs.get_frame()[0], "PC_Box", vp.roi_data, True)
    x_PC, y_PC, w_PC, h_PC = coord_PC_BOX
    
    box_slot_w = w_PC // 6
    box_slot_h = h_PC // 5

    
    if x > x_PC:
        position[1] = ((x - x_PC) // box_slot_w)+1
        
    position[0] = ((y - y_PC) // box_slot_h)
    
    return(list(map(int, position)))

def move_to(current_pos, target_pos):
    """
    Computes the number of up, down, left, right movements needed to reach the target position.

    Parameters
    ----------
        current_pos (tuple): Current position as (row, col).
        target_pos (tuple): Target position as (row, col).

    Returns
    -------
        dict: Movements required {"up": X, "down": Y, "left": Z, "right": W}
    """
    current_row, current_col = map(int, current_pos)
    target_row, target_col = map(int, target_pos)

    # Compute vertical movement
    if target_row > current_row:
        down = target_row - current_row
        up = 0
    else:
        up = current_row - target_row
        down = 0

    # Compute horizontal movement
    if target_col > current_col:
        right = target_col - current_col
        left = 0
    else:
        left = current_col - target_col
        right = 0

    return {"up": up, "down": down, "left": left, "right": right}

def movements_to_nxbt_macro(movements):
    """
    Converts movement dictionary into an NXBT macro.

    Parameters
    ----------
        movements (dict): Dictionary containing movements {'up': X, 'down': Y, 'left': Z, 'right': W}

    Returns
    -------
        str: NXBT macro string
    """
    macro_lines = []
    # Process UP movements
    for _ in range(movements["up"]):
        macro_lines.append("L_STICK@+000+100 0.1s")
        macro_lines.append("0.2s")

    # Process DOWN movements
    for _ in range(movements["down"]):
        macro_lines.append("L_STICK@+000-100 0.1s")
        macro_lines.append("0.2s")
        
    # Process LEFT movements
    for _ in range(movements["left"]):
        macro_lines.append("L_STICK@-100+000 0.1s")
        macro_lines.append("0.2s")

    # Process RIGHT movements
    for _ in range(movements["right"]):
        macro_lines.append("L_STICK@+100+000 0.1s")
        macro_lines.append("0.2s")


    # Join all lines into a formatted NXBT macro
    return "\n".join(macro_lines)

def box_go_to(destination, fs, ctrler, nx, retry = 40):
    """
    Moves the cursor to the specified destination in the PC Box Grid.

    Parameters
    ----------
    destination : tuple
        Target grid position as (row, col).
    fs : FrameStore
        Frame container used to detect current selection box.
    ctrler : str
        NXBT controller ID.
    nx : Nxbt instance
        NXBT controller interface.
    retry : int, optional
        Maximum number of attempts to detect current selection before proceeding (default is 40).

    Returns
    -------
    None
    """

    
    attempt = 0
    current_position = nvg.get_current_selection_box(fs, subframe_name = "Entire_Box")
    while attempt < retry and current_position is None:
        time.sleep(0.5)
        current_position = nvg.get_current_selection_box(fs, subframe_name = "Entire_Box")
        attempt += 1
    current_coordinate = analyze_selection_box(current_position, fs)

    while current_coordinate != destination:
        # add_to_log(f"Moving from {current_coordinate} → {destination}")
        instruction_dict = move_to(current_coordinate, destination)
        instruction_macro = movements_to_nxbt_macro(instruction_dict)
        nx.macro(ctrler, instruction_macro)
        
        time.sleep(1)
        
        current_position = nvg.get_current_selection_box(fs, subframe_name = "Entire_Box")
        current_coordinate = analyze_selection_box(current_position, fs)