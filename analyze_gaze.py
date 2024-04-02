from full_setup import Tracker
from full_setup import Window
import pandas as pd
from datetime import datetime

if __name__ == "__main__":
    # define desired area of interest for plotting purposes
    rectangle_size = (600, 600)
    ellipsoid_size = (1200, 600)
    elmo_head = (18, 14) # centimeters of elmo head (width * height)
    screen = (38, 25) # centimeters of screen
    screen_pixels = (1920, 1080)
    ratio = (18/38, 14/25)
    elmo_pixels = (int(screen_pixels[0]*ratio[0]), int(screen_pixels[1]*ratio[1]))
    ellipse_size = (elmo_pixels[0], elmo_pixels[1])
    window = Window(elmo_pixels, ellipse_size)
    tobii = False
    tracker = Tracker(window, tobii)
    file = 'gaze_data/gaze_data_19-03-15.46'
    input_file = file + '.csv'
    data = pd.read_csv(input_file)
    # gaze data columns
    selected_columns = data[['LeftGazePoint2DX', 'LeftGazePoint2DY', 'RightGazePoint2DX', 'RightGazePoint2DY', 'Timestamp']]
    left_gaze_x = selected_columns['LeftGazePoint2DX']
    left_gaze_y = selected_columns['LeftGazePoint2DY']
    right_gaze_x = selected_columns['RightGazePoint2DX']
    right_gaze_y = selected_columns['RightGazePoint2DY']
    time_stamp = selected_columns['Timestamp']
    tracker.plot_gaze_data_from_file(left_gaze_x, left_gaze_y, right_gaze_x, right_gaze_y, time_stamp)
    current_time = datetime.now()
    time_format = current_time.strftime("%d-%m-%H-%M")
    name = file + '_video_' + time_format + '.mp4'
    tracker.plot_video_gaze_data(left_gaze_x, left_gaze_y, right_gaze_x, right_gaze_y, time_stamp, name)



    



    

