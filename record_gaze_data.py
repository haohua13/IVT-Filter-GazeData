import tobii_research as tr
import time
import keyboard
import csv
from datetime import datetime
from pygame.locals import QUIT
import sys
# find Tobii eye trackers
found_eyetrackers = tr.find_all_eyetrackers()
my_eyetracker = found_eyetrackers[0]
print("Address: " + my_eyetracker.address)
print("Model: " + my_eyetracker.model)
# print("Name (It's OK if this is empty): " + my_eyetracker.device_name)
print("Serial number: " + my_eyetracker.serial_number)
# calibrate eye tracker in Manager Software or run calibrate.py or calibration_during_gaze_recording.py

# lists for gaze data
left_gaze_x = []
left_gaze_y = []
right_gaze_x = []
right_gaze_y = []
timestamps = []
system_timestamps = []
left_gaze_x_tbcs = []
left_gaze_y_tbcs = []
left_gaze_z_tbcs = []
right_gaze_x_tbcs = []
right_gaze_y_tbcs = []
right_gaze_z_tbcs = []


left_gaze_point_ucs_validity = []
right_gaze_point_ucs_validity = []
left_gaze_origin_ucs_validity = []
right_gaze_origin_ucs_validity = []

# gaze point in user coordinate system (UCS) data in millimeters
left_gaze_point_ucs_x = []
left_gaze_point_ucs_y = []
left_gaze_point_ucs_z = []

right_gaze_point_ucs_x = []
right_gaze_point_ucs_y = []
right_gaze_point_ucs_z = []

# gaze origin in user coordinate system (UCS) data in millimeters
left_gaze_origin_ucs_x = []
left_gaze_origin_ucs_y = []
left_gaze_origin_ucs_z = []

right_gaze_origin_ucs_x = []
right_gaze_origin_ucs_y = []
right_gaze_origin_ucs_z = []
    
# pupil data in millimeters
right_pupil_diameter = []
left_pupil_diameter = []
right_pupil_validity = []
left_pupil_validity = []

computer_timestamps = []

def write_all_data_to_file(filename):
        current_time = datetime.now()
        time_format = current_time.strftime("%d-%m-%H-%M")
        csv_file_name = "gaze_data/" + filename + time_format + ".csv"
        with open(csv_file_name, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            if csv_file.tell() == 0:
                header = ['Timestamp', 'LeftValidity', 'LeftGazePoint2DX', 'LeftGazePoint2DY', 
                          'LeftGazePoint3DX', 'LeftGazePoint3DY', 'LeftGazePoint3DZ', 
                          'LeftEyePosition3DX', 'LeftEyePosition3DY', 'LeftEyePosition3DZ',
                          'LeftPupilDiameter', 'LeftPupilValidity', 'LeftOriginValidity', 

                          'RightValidity', 'RightGazePoint2DX', 'RightGazePoint2DY', 
                          'RightGazePoint3DX', 'RightGazePoint3DY', 'RightGazePoint3DZ', 
                          'RightEyePosition3DX', 'RightEyePosition3DY', 'RightEyePosition3DZ',
                          'RightPupilDiameter', 'RightPupilValidity', 'RightOriginValidity', 
                          
                          'LeftGazeTBCSX', 'LeftGazeTBCSY', 'LeftGazeTBCSZ', 
                          'RightGazeTBCSX','RightGazeTBCSX', 'RightGazeTBCSX', 'SystemTimestamp', 'ComputerTimestamp'
                          ]
                
                csv_writer.writerow(header)
            for i in range(len(left_gaze_x)):
                row_data = [
                    timestamps[i],
                    # Left Eye 
                    left_gaze_point_ucs_validity[i],
                    left_gaze_x[i],
                    left_gaze_y[i],
                    left_gaze_point_ucs_x[i],
                    left_gaze_point_ucs_y[i],
                    left_gaze_point_ucs_z[i],
                    left_gaze_origin_ucs_x[i],
                    left_gaze_origin_ucs_y[i],
                    left_gaze_origin_ucs_z[i],
                    left_pupil_diameter[i],
                    left_pupil_validity[i],
                    left_gaze_origin_ucs_validity[i],

                    # Right Eye
                    right_gaze_point_ucs_validity[i],
                    right_gaze_x[i],
                    right_gaze_y[i],
                    right_gaze_point_ucs_x[i],
                    right_gaze_point_ucs_y[i],
                    right_gaze_point_ucs_z[i],
                    right_gaze_origin_ucs_x[i],
                    right_gaze_origin_ucs_y[i],
                    right_gaze_origin_ucs_z[i],
                    right_pupil_diameter[i],
                    right_pupil_validity[i],
                    right_gaze_origin_ucs_validity[i],

                    # not needed
                    left_gaze_x_tbcs[i],
                    left_gaze_y_tbcs[i],
                    left_gaze_z_tbcs[i],
                    right_gaze_x_tbcs[i],
                    right_gaze_y_tbcs[i],
                    right_gaze_z_tbcs[i],
                    system_timestamps[i],
                    computer_timestamps[i]

                ]
                csv_writer.writerow(row_data)

def save_gaze_data(gaze_data):
        # append gaze points to lists
        timestamps.append(gaze_data['device_time_stamp'])
        left_gaze_x.append(gaze_data['left_gaze_point_on_display_area'][0])
        left_gaze_y.append(gaze_data['left_gaze_point_on_display_area'][1])
        right_gaze_x.append(gaze_data['right_gaze_point_on_display_area'][0])
        right_gaze_y.append(gaze_data['right_gaze_point_on_display_area'][1])
        left_gaze_x_tbcs.append(gaze_data['left_gaze_origin_in_trackbox_coordinate_system'][0])
        left_gaze_y_tbcs.append(gaze_data['left_gaze_origin_in_trackbox_coordinate_system'][1])
        left_gaze_z_tbcs.append(gaze_data['left_gaze_origin_in_trackbox_coordinate_system'][2])
        right_gaze_x_tbcs.append(gaze_data['right_gaze_origin_in_trackbox_coordinate_system'][0])
        right_gaze_y_tbcs.append(gaze_data['right_gaze_origin_in_trackbox_coordinate_system'][1])
        right_gaze_z_tbcs.append(gaze_data['right_gaze_origin_in_trackbox_coordinate_system'][2])
        left_gaze_point_ucs_validity.append(gaze_data['left_gaze_point_validity'])
        right_gaze_point_ucs_validity.append(gaze_data['right_gaze_point_validity'])
        left_gaze_origin_ucs_validity.append(gaze_data['left_gaze_origin_validity'])
        right_gaze_origin_ucs_validity.append(gaze_data['right_gaze_origin_validity'])
        left_gaze_point_ucs_x.append(gaze_data['left_gaze_point_in_user_coordinate_system'][0])
        left_gaze_point_ucs_y.append(gaze_data['left_gaze_point_in_user_coordinate_system'][1])
        left_gaze_point_ucs_z.append(gaze_data['left_gaze_point_in_user_coordinate_system'][2])
        right_gaze_point_ucs_x.append(gaze_data['right_gaze_point_in_user_coordinate_system'][0])
        right_gaze_point_ucs_y.append(gaze_data['right_gaze_point_in_user_coordinate_system'][1])
        right_gaze_point_ucs_z.append(gaze_data['right_gaze_point_in_user_coordinate_system'][2])
        left_gaze_origin_ucs_x.append(gaze_data['left_gaze_origin_in_user_coordinate_system'][0])
        left_gaze_origin_ucs_y.append(gaze_data['left_gaze_origin_in_user_coordinate_system'][1])
        left_gaze_origin_ucs_z.append(gaze_data['left_gaze_origin_in_user_coordinate_system'][2])
        right_gaze_origin_ucs_x.append(gaze_data['right_gaze_origin_in_user_coordinate_system'][0])
        right_gaze_origin_ucs_y.append(gaze_data['right_gaze_origin_in_user_coordinate_system'][1])
        right_gaze_origin_ucs_z.append(gaze_data['right_gaze_origin_in_user_coordinate_system'][2])
        left_pupil_diameter.append(gaze_data['left_pupil_diameter'])
        right_pupil_diameter.append(gaze_data['right_pupil_diameter'])
        left_pupil_validity.append(gaze_data['left_pupil_validity'])
        right_pupil_validity.append(gaze_data['right_pupil_validity'])
        system_timestamps.append(gaze_data['system_time_stamp'])
        computer_timestamps.append(time.time())

def gaze_data_callback(gaze_data):
    print(gaze_data)
    save_gaze_data(gaze_data)

if __name__ == "__main__":
    # start data collection
    args = sys.argv
    # first argument is identifier of the user, second argument is optional for elmo
    if len(args) > 1:
        filename = "gaze_data_" + args[1] 
        if args[2] == "elmo": 
             filename = filename + "_" + args[2]      
        
    print("Subscribing to data for eye tracker with serial number {0}.".format(my_eyetracker.serial_number))
    time.sleep(3)
    # my_eyetracker.subscribe_to(tr.EYETRACKER_USER_POSITION_GUIDE, user_position_guide_callback, as_dictionary=True)
    my_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)

    while True:
        if keyboard.is_pressed('w'): # press w keyboard to stop the recording
            write_all_data_to_file(filename)
            break
    my_eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
