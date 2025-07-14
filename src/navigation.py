"""
navigation.py

Menu navigation utilities for Pokémon Scarlet and Violet automation.

This module provides tools for identifying current in-game interfaces via OCR and color detection,
and navigating to specific menu locations using controller macros. It uses a graph-based structure
of `MenuOption` objects with associated identifiers and back-navigation actions.

Key Features
------------
- Detects current menu interface from video frames using ROI-based analysis.
- Navigates to target menus from current location via simulated button sequences.
- Supports directional movement macros in the main menu and beyond.
- Includes logic for identifying submenus like BOXES, PICNIC, CAMERA, OVERWORLD, etc.

Classes
-------
MenuOption : Represents a game menu interface and its relationships.

Functions
---------
find_current_interface() : Uses BFS to determine the current interface.
get_current_selection_box() : Detects highlight box position in menus.
outbound_main_menu() : Computes navigation macros from current menu state.
from_main_menu_to() : Navigates to a target menu option.
to_main_menu() : Returns from any menu/interface to the main menu.
go_to() : High-level wrapper to go to any named interface.

Constants
---------
menu_dict : Maps menu names to grid coordinates in the main menu.
"""


import cv2
import threading
import time
import numpy as np
# from src import init_controller
from src import video_processing as vp
from src import traverse_coordinate as tc
from collections import deque
from nxbt import Buttons
from nxbt import Sticks


class MenuOption:
    """
    Represents a menu interface in Pokémon SV.

    Parameters
    ----------
        name (str): Name of the menu.
        identifier (function): Function that returns True if the current frame is in this menu.
        next_interfaces (dict): Mapping of next menus {menu_name: (MenuOption, action)}.
        back_action (str): Button sequence to return to the parent menu.
    """
    def __init__(self, name, identifier, back_action):

        self.name = name  # Name of the menu
        self.identifier = identifier  # Function to check if we are in this menu
        self.next_interfaces = set()  # Set of connected MenuOption objects
        self.back_action = back_action  # Action to return to parent

    def get_next_interface(self):
        """Returns the next interface and the action required to navigate there."""
        return self.next_interfaces

    def add_connection(self, menu_option):
        """Adds a connection to another menu."""
        self.next_interfaces.add(menu_option)

    def navigate_back(self):
        """Returns the back navigation action."""
        return self.back_action
    def __hash__(self):
        return hash(self.name)  # or any unique attribute that identifies it

    def __eq__(self, other):
        return isinstance(other, MenuOption) and self.name == other.name



def in_main_menu(fs):
    main_menu_subframe = vp.get_roi_by_name(fs.get_frame()[1], "Main_Menu_Indicator", vp.roi_data)
    
    lower_red = np.array([0, 195, 250])   
    upper_red = np.array([5, 200, 255])
    
    mask = cv2.inRange(main_menu_subframe, lower_red, upper_red)
    mask_count = cv2.countNonZero(mask)
    ratio = mask_count/mask.size

    return(ratio < 0.245 and ratio > 0.23)
MAIN_MENU = MenuOption(
    name = "MAIN MENU",
    identifier = in_main_menu,
    back_action = None
)

def find_current_interface(fs, start_menu = MAIN_MENU):
    """
    Uses BFS to find the current game interface based on identifier functions.

    Parameters
    ----------
        frame (numpy.ndarray): The current game frame.
        start_menu (MenuOption): The root menu object to start BFS from (e.g., MAIN_MENU).

    Returns
    -------
        MenuOption or None: The class object representing the current interface, or None if no match is found.
    """
    queue = deque([start_menu])  # BFS queue
    visited = set()  # Track visited nodes to avoid redundant checks

    while queue:
        current_menu = queue.popleft()  # Get the next menu node

        if current_menu in visited:
            continue  # Skip if already checked
        visited.add(current_menu)

        # Check if this menu matches the current game state
        if current_menu.identifier(fs):
            return current_menu  # Found the correct interface

        # Add connected menus to the queue for BFS traversal
        queue.extend(current_menu.get_next_interface())

    return OTHER  # No matching interface found


def in_other():
    return(True)
OTHER = MenuOption(
    name = "OTHER",
    identifier = in_other,
    back_action=["B"]
)

def in_boxes(fs):
    img = vp.get_roi_by_name(fs.get_frame()[0], "Box_Indicator", vp.roi_data)
    box = vp.get_text(img)[0]
    return(box == "Party and Boxes")
BOXES = MenuOption(
    name = "BOXES",
    identifier = in_boxes,
    back_action = ["B"]
)

def in_picnic(fs):
    img = vp.get_roi_by_name(fs.get_frame()[0], "Picnic_Indicator", vp.roi_data)
    picnic = vp.get_text(img)[0]
    return(picnic == "Pack Up and Go")
PICNIC = MenuOption(
    name = "PICNIC",
    identifier = in_picnic,
    back_action = ["Y", "A"]
)

def in_poke_summary(fs):
    img = vp.get_roi_by_name(fs.get_frame()[0], "Pokemon_Summary_Indicator", vp.roi_data)
    summary = vp.get_text(img)[0]
    return("STATUS SUMMARY" in summary)
POKEMON_SUMMARY = MenuOption(
    name = "POKEMON_SUMMARY",
    identifier = in_poke_summary,
    back_action = ["B"]
)


def in_camera(fs):
    
    img = vp.get_roi_by_name(fs.get_frame()[1], "Camera_Indicator", vp.roi_data)
    lower_white = np.array([0, 0, 254])   
    upper_white = np.array([1, 1, 255])
    
    mask = cv2.inRange(img, lower_white, upper_white)
    mask_count = cv2.countNonZero(mask)
    ratio = mask_count/mask.size
    return(ratio > 0.51 and ratio < 0.53)

CAMERA = MenuOption(
    name = "CAMERA",
    identifier = in_camera,
    back_action = ["B"]
)


def in_map_pokedex_profile(fs):
    img = vp.get_roi_by_name(fs.get_frame()[1], "Map_Pokedex_Profile_Indicator", vp.roi_data)
    lower_white = np.array([0, 0, 254])   
    upper_white = np.array([1, 1, 255])
    
    mask = cv2.inRange(img, lower_white, upper_white)
    mask_count = cv2.countNonZero(mask)
    return((mask_count/mask.size) > 0.95)
MAP_POKEDEX_PROFILE = MenuOption(
    name = "MAP_POKEDEX_PROFILE",
    identifier = in_map_pokedex_profile,
    back_action = ["B"]
)


def in_overworld(fs):
    img_1 = vp.get_roi_by_name(fs.get_frame()[1], "Overworld_1_Indicator", vp.roi_data)
    img_2 = vp.get_roi_by_name(fs.get_frame()[1], "Overworld_2_Indicator", vp.roi_data)
    
    # Define color range for the selection (bright yellow)
    lower_green = np.array([71, 234, 254])   
    upper_green = np.array([75, 238, 255])
    
    mask_1 = cv2.inRange(img_1, lower_green, upper_green)
    mask_2 = cv2.inRange(img_2, lower_green, upper_green)
    
    mask_count = cv2.countNonZero(mask_1) + cv2.countNonZero(mask_2)
    return((mask_count/(mask_1.size + mask_2.size)) > 0.7)

OVERWORLD = MenuOption(
    name = "OVERWORLD",
    identifier = in_overworld,
    back_action = ["X"]
)



def get_current_selection_box(fs, roi_data=vp.roi_data, subframe_name = None):
    """
    Detects the selection area within a sub-frame of the main frame.

    This function extracts the "Entire_Box" region from the given frame, applies
    color filtering to detect a bright yellow selection, and determines its bounding
    box. The selection coordinates are adjusted to be relative to the original frame.

    Parameters
    ----------
        frame (np.ndarray, optional): The main frame from which the selection is detected.
                                      Defaults to the most recent captured frame (vp.cf).
        roi_data (pd.DataFrame, optional): The dataset containing pre-defined ROIs.
                                           Defaults to vp.roi_data.

    Returns
    -------
        tuple or None: A tuple `(x_abs, y_abs, w, h)` representing the absolute coordinates
                       and size of the detected selection area in the main frame.
                       Returns `None` if no selection is found.

    Example
    -------
        >>> selection_coords = get_box_selection()
        >>> if selection_coords:
        >>>     x, y, w, h = selection_coords
        >>>     print(f"Selection detected at ({x}, {y}), size: {w}x{h}")
    """
    frame = fs.get_frame()[1]
    while frame is None:
        print("Waiting for frames to come in")
        time.sleep(0.5)
        frame = fs.get_frame()[1]
    
    if subframe_name is not None:
        frame, sub_coordinate = vp.get_roi_by_name(frame, subframe_name, roi_data, True)
        x_sub, y_sub, w_sub, h_sub = sub_coordinate
    else:
        x_sub, y_sub = 0, 0

    # Define color range for the selection (bright yellow)
    lower_yellow = np.array([24, 253, 253])   
    upper_yellow = np.array([28, 255, 255])

    # Create a binary mask
    mask = cv2.inRange(frame, lower_yellow, upper_yellow)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour by area (most yellow pixels)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the smallest bounding rectangle around it
        x, y, w, h = cv2.boundingRect(largest_contour)

        return (x+x_sub, y+y_sub, w, h)  # Return both the mask and bounding rectangle
    
    return None  # No selection found


MAIN_MENU.add_connection(BOXES)
MAIN_MENU.add_connection(PICNIC)
MAIN_MENU.add_connection(POKEMON_SUMMARY)
MAIN_MENU.add_connection(OVERWORLD)

OVERWORLD.add_connection(MAP_POKEDEX_PROFILE)
OVERWORLD.add_connection(CAMERA)
BOXES.add_connection(POKEMON_SUMMARY)
PICNIC.add_connection(CAMERA)


def outbound_main_menu(destination, fs):
    """
    Determines the current selection position in the game's main menu based on OCR data,
    compares it to the desired destination position, and generates the appropriate macro
    to navigate to the target location using simulated controller input.

    This function analyzes the current cursor position (highlight box) in the main menu
    using the latest frame from FrameStore, compares it to the given destination,
    and generates movement instructions accordingly.

    Parameters
    ----------
    destination : list[int]
        Target menu position as [row, column]. If [-1, -1], it indicates a return to the overworld.
    fs : FrameStore
        Thread-safe container that provides the current video frame for OCR/ROI analysis.

    Returns
    -------
    position : list[int]
        The detected current position of the selection box (row, column) in the main menu.
    macro : str
        A string-formatted macro that represents controller inputs to navigate from
        current to destination position. If destination is [-1, -1], returns "B 0.1s\n 0.2s".

    Notes
    -----
    - If current selection cannot be detected, a fallback macro is returned to move the cursor right.
    - Position calculation is based on bounding box heuristics.
    - Handles menu UI inconsistencies (e.g., row index 7 showing up incorrectly).
    """
    
    current_xywh = get_current_selection_box(fs, vp.roi_data)
    if current_xywh is None:
        instruction = {'up': 0, 'down': 0, 'left': 0, 'right': 1}
        macro = tc.movements_to_nxbt_macro(instruction)
        return([0,0], macro)
    
    x, y, w, h = current_xywh
    
    x, y = x + w/2, y + h/2

    menu_options_subframe, xywh_options = vp.get_roi_by_name(fs.get_frame()[0], "Main_Menu_Options", vp.roi_data, True)
    xo, yo, wo, ho = xywh_options
    
    position = [0,0]
    
    if destination == [-1, -1]:
        if OVERWORLD.identifier(fs):
            position = [-1, -1]
        return(position, "B 0.1s\n 0.2s")

    if x > xo and y > yo and y < yo + ho:
        position[1] = 1
        h_one_option = ho/6
        position[0] = int((y- yo)//h_one_option)
    if x > xo and y > yo + ho:
        position[1] = 1
        position[0] = 6
    if x < xo and y > 80: # numbers are from the bounding box of the party in main menu
        position[0] = int((y - 80)//50)
        
    if position[0] == 7: # dealing with some oddities in UI 
        position[0] = 6
        
    if destination[1] == position[1]:
        instruction = tc.move_to(position, destination)
    if destination[1] > position[1]:
        instruction = {'up': 0, 'down': 0, 'left': 0, 'right': 1}
    if destination[1] < position[1]:
        instruction = {'up': 0, 'down': 0, 'left': 1, 'right': 0}
        
    macro = tc.movements_to_nxbt_macro(instruction)
    return(position, macro)



menu_items = ["BAG", "BOXES", "PICNIC", "POKEPORTAL", "OPTIONS", "SAVE", "DOWNLOADABLE_CONTENT"]
menu_dict = {name: [row, 1] for row, name in enumerate(menu_items)}
menu_dict.update({f"PARTY_{x}": [x, 0] for x in range(6)})
menu_dict.update({f"RIDE": [6, 0]})
menu_dict.update({f"OVERWORLD": [-1, -1]})

def from_main_menu_to(destination_name, fs, ctrler, nx):
    
    """
    Navigates from the main menu to a specified destination.

    Parameters
    ----------
    destination_name : str
        Name of the target menu option.
    fs : FrameStore
        Frame container used to detect current menu position.
    ctrler : str
        Controller ID.
    nx : Nxbt instance
        Nxbt controller interface.
    """
    
    destination = menu_dict[destination_name]
    position, macro = outbound_main_menu(destination, fs)

    while position != destination:

        nx.macro(ctrler, macro)

        time.sleep(0.8)
        position, macro = outbound_main_menu(destination, fs)

        
def to_main_menu(fs, ctrler, nx):
    
    """
    Exits the current interface and returns to the main menu.

    Parameters
    ----------
    fs : FrameStore
        Frame container used to detect current interface state.
    ctrler : str
        Controller ID.
    nx : Nxbt instance
        Nxbt controller interface.
    """
    
    current_interface = find_current_interface(fs)

    while current_interface.name != "MAIN MENU":
        macro_list = []
        back_actions = current_interface.back_action
        for action in back_actions:
            macro_list.append(action + " 0.1s")
            macro_list.append("0.1s")
        macro = "\n".join(macro_list)
        if macro.strip() != "":
            nx.macro(ctrler, macro)
        time.sleep(1)
        current_interface = find_current_interface(fs)
        if current_interface.name == "OTHER":

            time.sleep(1)
            current_interface = find_current_interface(fs)
            

        
def go_to(destination_name, fs, ctrler, nx, refresh = False):
    
    """
    Navigates to a specified in-game menu interface using OCR and controller macros.

    If the current interface is not the desired destination, or if refresh is True,
    this function returns to the main menu and navigates to the target destination.

    Parameters
    ----------
    destination_name : str
        Name of the desired interface to navigate to.
    fs : FrameStore
        Frame container used to detect current interface.
    ctrler : str
        Controller ID.
    nx : Nxbt instance
        Nxbt controller interface.
    refresh : bool, optional
        If True, forces navigation even if already at the destination (default is False).
    """
    
    current_interface = find_current_interface(fs)
    while refresh or current_interface.name != destination_name:

        to_main_menu(fs, ctrler, nx)
        time.sleep(0.5)
        from_main_menu_to(destination_name, fs, ctrler, nx)
        nx.press_buttons(ctrler, Buttons.A, 0.3, 0.1)
        
        time_out = time.time() + 10
        while find_current_interface(fs).name != destination_name and time.time() < time_out:
            time.sleep(0.2)
        
        current_interface = find_current_interface(fs)
        refresh = False
        


