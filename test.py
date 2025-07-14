import time
import threading
import pandas as pd

from src import video_processing as vp
from src import init_controller
import src.traverse_coordinate as tc
from src import navigation as nvg
import hatch

roi_data = pd.read_csv(r"src/roi_data.csv")


# import easyocr
# reader = easyocr.Reader(['en'], gpu=True) 


# def get_stat(fs, nx, ctrler):
#     xywh = nvg.get_current_selection_box(fs, roi_data, "Entire_Box")
#     position = tc.analyze_selection_box(xywh, fs)
    
#     box_number_roi = vp.get_roi_by_name(fs.get_frame()[0], "Stat_Box_Number", roi_data)

#     box_number = reader.readtext(box_number_roi)
#     print(box_number)
#     # stat_rows = roi_data[roi_data["Name"].str.contains("Stat_")]
    
#     # stat_dic = {"Box_Number": box_number, "row": position[0], 'col': position[1]}
    
#     # for _, row in stat_rows.iterrows():
#     #     name = row['Name']
#     #     roi = vp.get_roi_by_name(fs.get_frame()[0], name, roi_data)
#     #     value = vp.get_text(roi)[0]
#     #     stat_dic[name.replace("Stat_", "")] = value
    
#     return(None)


# nx, ctrler = init_controller.nx_init()
cap = vp.make_video_capture()
fs = vp.FrameStore()
stop_event = threading.Event()
video_thread = threading.Thread(target = vp.update_video_feed, 
                                args = (cap, fs, vp.roi_data), 
                                kwargs={'ocr_threshold': 70, 'display': True, 'stop_event': stop_event},
                                daemon=True)
video_thread.start()
time.sleep(1)


# while True:
#     print(get_stat(fs, nx = None, ctrler = None))
#     time.sleep(2)


# print("Done")
# stop_event.set()
video_thread.join()




