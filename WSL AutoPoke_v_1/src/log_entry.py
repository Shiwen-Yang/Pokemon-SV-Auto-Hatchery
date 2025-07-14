
import time
import re
import csv
import os
from datetime import datetime

path = "summary/log.txt"
summary_path = "summary/summary_hatch.csv"


def add_to_log(message, timestamp=None, filename=path):
    """
    Appends a log message to log.txt with a timestamp.

    Args:
        message (str): The log message to be recorded.
        timestamp (float, optional): A UNIX timestamp (time.time()). If None, uses the current time.
        filename (str): The log file name (default: "log.txt").
    """
    # Use current time if no timestamp is provided
    if timestamp is None:
        timestamp = time.time()
    
    # Convert timestamp to a readable format
    readable_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    
    # Format the log entry
    log_entry = f"[{readable_time}] {message}\n"
    
    # Append the log entry to the file
    with open(filename, "a") as file:  # "a" mode ensures appending
        file.write(log_entry)
        

def summarize_and_clear_log(log_path=path, summary_csv=summary_path):
    
    with open(log_path, "r") as f:
        lines = f.readlines()
    
    if not lines:
        print("Log is empty. Nothing to summarize.")
        return

    # Extract timestamps and egg count
    start_line = lines[0]
    end_line = lines[-1] if "Pipeline complete." in lines[-1] else None

    if not end_line:
        print("No completion line found. Skipping summary.")
        return

    # Parse start time
    start_match = re.match(r"\[(.*?)\]", start_line)
    start_time = datetime.strptime(start_match.group(1), "%Y-%m-%d %H:%M:%S") if start_match else None

    # Parse end time and total eggs
    end_match = re.match(r"\[(.*?)\] Pipeline complete\. Total eggs hatched: (\d+)", end_line)
    if end_match:
        end_time = datetime.strptime(end_match.group(1), "%Y-%m-%d %H:%M:%S")
        total_eggs = int(end_match.group(2))
    else:
        print("End line malformed. Skipping summary.")
        return

    # Compute hatching speed
    duration_seconds = (end_time - start_time).total_seconds()
    speed_per_hr = round(3600 * total_eggs / duration_seconds, 2)

    # Append to summary CSV
    header = ["start_time", "end_time", "duration_sec", "total_eggs", "speed_eggs_per_hr"]
    new_row = [start_time, end_time, int(duration_seconds), total_eggs, speed_per_hr]
    file_exists = os.path.exists(summary_csv)

    with open(summary_csv, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(new_row)

    # Clear the log
    with open(log_path, "w") as f:
        f.write("")
