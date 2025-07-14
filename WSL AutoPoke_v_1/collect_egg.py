from __future__ import annotations
import time
import threading
import argparse
import sys
from nxbt import Buttons
from nxbt import Sticks

from src import video_processing as vp
from src import macro as mcr
from src import init_controller
from src import navigation as nvg
from src.log_entry import add_to_log

Conversation_List = [
    """ You peeked inside the basket! """,
    """ Doesn't look like there's anything in the basket
    so far. """,
    """ There's a Pokémon Egg inside! Do you want to take it? """,
    """ You took the Egg! """,
    """ ...Oh? There's something else in the basket! """,
    """ It's another Pokémon Egg! Do you want to take it? """]

egg_count = 0
box_full = 0
log_path = r"summary/egg_log.txt"

def kill_conversation(fs,ctrler,nx):
    """
    Automatically skips through in-game conversation dialogs using button presses.

    This function continuously presses the 'A' button to skip ongoing conversations,
    checking the conversation state via OCR. It monitors for key phrases such as
    "You took the Egg!" to increment the egg count and detect when the storage box is full.

    Parameters
    ----------
    fs : FrameStore
        Frame container used to access the current OCR-detected conversation text.
    ctrler : str
        NXBT controller ID used to send button inputs.
    nx : Nxbt
        NXBT instance to execute controller commands.

    Notes
    -----
    - This function runs a maximum of 100 loops to avoid infinite hangs.
    - Increments `egg_count` if an egg is collected.
    - Sets the `box_full` flag if the message indicates the storage is full.
    - Uses global variables `egg_count` and `box_full` for tracking state.
    - Logs each egg collection and prints progress in-place to the console.
    """
    global egg_count, box_full
    i = 1
    while fs.get_convo() != "" and i <= 100:
        # keep pressing A until the conversation goes away
        nx.press_buttons(ctrler, [Buttons.A], 0.1, 0.2)
        i += 1
        time.sleep(1.8)
        
        if "already full" in fs.get_convo():
            box_full = 1
        
        # if we happen to be in the egg conversation, get an egg count
        if fs.get_convo() == "You took the Egg!":
            timestamp = time.strftime("%H%M")
            egg_count += 1
            msg = f"[{timestamp}] Eggs Collected: {egg_count}   "
            add_to_log(msg, filename= log_path)
            sys.stdout.write("\r"+msg)
            sys.stdout.flush()

def collect_eggs(fs, ctrler, nx, t = 1800):
    """
    Automates the egg collection loop from the picnic basket in Pokémon SV.

    This function repeatedly interacts with the basket to collect eggs and skips 
    through conversations using `kill_conversation`. It loops for a given time limit 
    or until the storage box becomes full.

    Parameters
    ----------
    fs : FrameStore
        Frame container used to monitor OCR-detected conversation state.
    ctrler : str
        NXBT controller ID used for sending inputs.
    nx : Nxbt
        NXBT instance used to execute controller macros.
    t : int, optional
        Maximum time (in seconds) to run the collection loop (default: 1800 seconds = 30 minutes).

    Notes
    -----
    - Eggs are collected by repeatedly pressing 'A' near the basket and monitoring for egg-related messages.
    - After each egg interaction, the script sleeps for 90 seconds to simulate idle waiting.
    - Global `box_full` flag is used to stop collection early if the box is full.
    - `kill_conversation()` handles updating the egg count and skipping dialogues.
    - At the end of the loop, a final interaction ensures no remaining eggs are left uncollected.
    """
    start_time = time.time()
    time_elapsed = 0
    
    while time_elapsed < t:
        
        if box_full == 1:
            break
        
        nx.press_buttons(ctrler, [Buttons.A], 0.1, 0.2)
        time.sleep(2)
        kill_conversation(fs, ctrler, nx)  
        
        sleep_time = 90  # Total sleep duration
        while sleep_time > 0 and time_elapsed < t:
            time.sleep(1)  # Sleep in 1-second chunks
            sleep_time -= 1
            time_elapsed = time.time() - start_time  # Recalculate elapsed time
        
    nx.press_buttons(ctrler, [Buttons.A], 0.1, 0.2)
    time.sleep(2)
    kill_conversation(fs, ctrler, nx)


def start_over(fs, ctrler, nx):
    """
    Resets the character position to the picnic table and initiates the egg collection conversation.

    This function ensures the player is in the picnic interface and standing in front of the basket
    with the conversation window open. It performs navigation, macro walking, and interaction to 
    reach the desired in-game state.

    Parameters
    ----------
    fs : FrameStore
        Frame container used to monitor the game state and conversation text.
    ctrler : str
        NXBT controller ID used for input commands.
    nx : Nxbt
        NXBT instance for executing macros and button inputs.

    Notes
    -----
    - Waits until a valid frame is available before proceeding.
    - Navigates to the picnic using `nvg.go_to()`, performs walking macro, and initiates interaction.
    - If a "peeked inside the basket" message appears, `kill_conversation()` is called to clear the dialogue.
    - Additional stick movement ensures the character is properly repositioned at the table.
    - Loop continues until the conversation indicates the player is in the picnic scene.
    """
    # This function will put us at the sandwich table, with the conversation open
    while fs.get_frame() is None:
        time.sleep(0.5)
    go = True
    while go:
        
        nvg.go_to("PICNIC", fs, ctrler, nx, True)
        time.sleep(6)
        nx.macro(ctrler, mcr.box_walk)
        nx.press_buttons(ctrler, [Buttons.A], 0.1, 2)
        
        if "peeked" in fs.get_convo():
            kill_conversation(fs, ctrler, nx)
            nx.tilt_stick(ctrler, Sticks.LEFT_STICK, 100, 0, 1, 1)
            nx.press_buttons(ctrler, Buttons.A, 0.1, 2)
        
        go = "Picnic" not in fs.get_convo()
        
        
def reposition_to_sandwich(fs, ctrler, nx):
    """
    Repositions the character back to the sandwich table during a picnic session.

    This function is used when the character has potentially moved away from the table.
    It attempts to reposition by walking in a loop and pressing interaction buttons
    until the picnic conversation is detected again. If all attempts fail, it falls back
    to a full reset using `start_over()`.

    Parameters
    ----------
    fs : FrameStore
        Frame container used to monitor the game state and conversation text.
    ctrler : str
        NXBT controller ID used for input commands.
    nx : Nxbt
        NXBT instance used to execute macros and button presses.

    Notes
    -----
    - If a basket conversation is detected mid-way, it is cleared via `kill_conversation()`.
    - Movement is done using alternating stick tilts to simulate repositioning near the table.
    - Interaction is attempted repeatedly until the "Picnic" text is detected.
    - If the target state is not achieved after 20 iterations, `start_over()` is invoked
      to ensure recovery and repositioning.
    """
    nx.press_buttons(ctrler, Buttons.A, 0.1, 2)
    
    i = 1
    while("Picnic" not in fs.get_convo() and i < 20):
        
        if "peeked" in fs.get_convo():
            kill_conversation(fs, ctrler, nx)

        time.sleep(2)
        nx.tilt_stick(ctrler, Sticks.LEFT_STICK, 0, 100, 0.1, 0.1)
        time.sleep(1.9)
        nx.tilt_stick(ctrler, Sticks.LEFT_STICK, 100, 0, 0.1, 0.1)
        time.sleep(1.9)
        nx.press_buttons(ctrler, Buttons.A, 0.1, 5)
        i += 1
    
    if i == 20 and "Picnic" not in fs.get_convo():
        start_over(fs, ctrler, nx)    


def make_sandwich(ctrler, nx):
    """
    Executes the sequence to initiate and complete sandwich making in a picnic session.

    Assumes the character is already positioned at the sandwich table with the interaction
    dialog open. This function simulates pressing the necessary buttons and executes a 
    predefined sandwich-making macro.

    Parameters
    ----------
    ctrler : str
        NXBT controller ID used to send input commands.
    nx : Nxbt
        NXBT instance used to execute macros and stick movements.

    Notes
    -----
    - Button A is held for an extended press to start the sandwich interaction.
    - A macro defined in `mcr.sandwich` performs the sandwich-making process.
    - A slight stick tilt is added after the macro to ensure interaction continues smoothly.
    """
    # here we should be facing the sandwich table, and have the conversation open
    nx.press_buttons(ctrler, Buttons.A, 0.1, 6)
    nx.macro(ctrler, mcr.sandwich)
    time.sleep(2)
    
    nx.tilt_stick(ctrler, Sticks.LEFT_STICK, 0, 100, 0.05, 0.1)    
    

def find_egg_basket(fs, ctrler, nx):
    """
    Moves the character vertically to locate the egg basket.

    Steps:
    1. Press 'A' to check if the basket is nearby.
    2. If not found (fs.get_convo() == ""), move up and down in small steps.
    3. Stop once the conversation box appears (fs.get_convo() != "").
    """
    """
    Locates the egg basket during a picnic session by scanning vertically.

    This function attempts to find the egg basket by simulating in-game movement and checking
    for the associated conversation dialog that appears when the basket is found.

    Parameters
    ----------
    fs : FrameStore
        Frame container used to read the latest OCR-detected conversation text.
    ctrler : str
        NXBT controller ID used to send input commands.
    nx : Nxbt
        NXBT instance used to execute controller inputs.

    Returns
    -------
    bool
        True if the egg basket was found (i.e., the conversation text includes "peeked"),
        False if not found after scanning the area.

    Notes
    -----
    - The function first attempts a stationary check by pressing 'A' to see if the basket is
      already within reach.
    - If no dialog appears, the function begins vertical movement (downward first, then upward),
      simulating up to 20 steps in each direction, checking after each move.
    - Movement stops immediately upon detecting the basket.
    - If no basket is found after the full search, it logs the failure and returns False.
    """
    time.sleep(3)
    # Step 1: Check if we are already at the basket
    nx.press_buttons(ctrler, Buttons.A, 0.1, 0.2)
    time.sleep(3)  # Small delay for text to appear

    if "peeked" in fs.get_convo():
        add_to_log("Basket found!", filename=log_path)
        return True  # Already at the basket

    # Step 2: Search for the basket by moving up and down
    add_to_log("Searching for the basket...", filename= log_path)
    for sign in [-1, 1]:  # First move down (1), then up (-1)
        for _ in range(20):  # Move up to 10 steps in each direction
            nx.tilt_stick(ctrler, Sticks.LEFT_STICK, 0, sign * 100, 0.1, 0.1)
            time.sleep(2)  # Allow movement to take effect

            # Step 3: Check again after moving
            nx.press_buttons(ctrler, [Buttons.A], 0.1, 0.2)
            time.sleep(1)  # Wait for potential text update

            if "peeked" in fs.get_convo():  # Basket found
                add_to_log("Basket found!", filename= log_path)
                return True

    # If we finished moving and didn't find the basket
    add_to_log("Basket not found!", filename= log_path)
    return False
    
    

def n_rounds_of_sandwich(n, fs, ctrler, nx):
    """
    Automates multiple rounds of sandwich-based egg farming in Pokémon Scarlet/Violet.

    Each round consists of making a sandwich, locating the egg basket, collecting eggs
    for a fixed duration, and repositioning back to the sandwich table. The loop continues
    for `n` rounds or until the box becomes full.

    Parameters
    ----------
    n : int
        Number of sandwich-egg collection rounds to perform.
    fs : FrameStore
        Frame container used for conversation monitoring and game state detection.
    ctrler : str
        NXBT controller ID used for sending inputs.
    nx : Nxbt
        NXBT instance used to execute controller macros.

    Notes
    -----
    - Each round starts by making a sandwich and searching for the egg basket.
    - If the basket is not found, up to 10 retries are performed by resetting the picnic setup.
    - Eggs are collected for a fixed duration (1800 seconds or 30 minutes) per round.
    - After egg collection, the player is repositioned back to the sandwich table.
    - The loop exits early if the storage box becomes full during egg collection.
    - Logs are written for the start and end of each round.
    """
    start_over(fs, ctrler, nx)

    for counter in range(n):
        
        add_to_log(f"Round {counter+1} Initialized", filename= log_path)
        make_sandwich(ctrler, nx)
        add_to_log("sandwich made", filename= log_path)
        found_the_basket = find_egg_basket(fs, ctrler, nx)
        
        i = 1
        while not found_the_basket and i <= 10:
            start_over(fs, ctrler, nx)
            make_sandwich(ctrler, nx)
            found_the_basket = find_egg_basket(fs, ctrler, nx)
            i += 1         

        collect_eggs(fs, ctrler, nx, 1800)
        
        nx.tilt_stick(ctrler, Sticks.LEFT_STICK, 100, 0, 0.05, 0.1)
        
        # check if we are at the sandwich table        
        reposition_to_sandwich(fs, ctrler, nx)
        add_to_log(f"Round {counter+1} Finished", filename= log_path)
        
        if box_full == 1:
            break
    
    
        


# Next we need to implement a sandwich check.

if __name__ == "__main__":
# ------------------------------------------------------------------------------
# Entry point for executing the egg collection automation script.
#
# This block initializes the NXBT controller, starts the video capture thread,
# and executes a specified number of sandwich-egg collection rounds.
#
# Steps:
# 1. Initialize NXBT controller and prepare for macro execution.
# 2. Start real-time video feed processing in a separate thread using FrameStore.
# 3. Parse optional command-line argument `-n` or `--num` to specify number of rounds.
# 4. Run `n_rounds_of_sandwich()` to perform egg farming automation.
# 5. Wait for the video processing thread to terminate and cleanly exit.
#
# Notes:
# - Run this script using `sudo python3 script_name.py -n <number_of_rounds>`.
# - Default number of rounds is 1 if not specified.
# ------------------------------------------------------------------------------
    nx, ctrler = init_controller.nx_init()
    time.sleep(1)
    print("Controller Ready")

    cap = vp.make_video_capture()
    fs = vp.FrameStore()
    stop_event = threading.Event()
    video_thread = threading.Thread(target = vp.update_video_feed, 
                                    args = (cap, fs, vp.roi_data), 
                                    kwargs={'ocr_threshold': 70, 'display': True, 'stop_event': stop_event},
                                    daemon=True)
    video_thread.start()
    time.sleep(1)
    # Parse command-line argument using a flag (-n or --num)
    parser = argparse.ArgumentParser(description="Run egg collection for a specified number of rounds.")
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of egg collection rounds (default: 1)")
    args = parser.parse_args()
    n_rounds_of_sandwich(args.num, fs, ctrler, nx)
    
    print("Done")
    stop_event.set()
    video_thread.join()
    
    
    
    

    
    
    

    
        
    
    
