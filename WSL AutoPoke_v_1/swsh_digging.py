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
import collect_egg as CE



def dig(k, fs, ctrler, nx):
    for i in range(k):
        nx.press_buttons(ctrler, [Buttons.A], 0.1, 1)
        CE.kill_conversation(fs, ctrler, nx) 
        
        



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
    dig(1000, fs, ctrler, nx)
    
    print("Done")
    stop_event.set()
    video_thread.join()