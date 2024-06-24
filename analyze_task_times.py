import csv
import pandas as pd
from datetime import datetime
df = pd.read_csv('gaze_data/data_clicks_elmo.csv')
# get all the relevant times, convert to timestamps, also ID

date_time = df['Date_time'][0]
date_format = "%m/%d/%Y %I:%M:%S %p"
time_object = datetime.strptime(date_time, date_format)
timestamps = time_object.timestamp()
processed_data = []
# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    event_name = row['Name'].strip()  # Extract the event name
    current_time = row['Date_time'].strip()  # Extract the timestamp string
    time_obj = datetime.strptime(current_time, date_format)
    timestamp_str = time_obj.timestamp()
    obs_text = str(row['Obs']).strip()  # Extract the 'Obs' column as string
    
    # Check if the event is 'click_start', 'click_to_task', or if 'Obs' contains 'human decision in mushroom' or 'final decision in mushroom'
    if event_name == 'click_start':
        processed_data.append([row['ID'], timestamp_str, 'click_start'])
    elif event_name == 'click_to_tutorial':
        processed_data.append([row['ID'], timestamp_str, 'click_to_tutorial'])
    elif event_name == 'click_to_task':
        processed_data.append([row['ID'], timestamp_str, 'tutorial_task'])
    elif 'human decision in mushroom' in obs_text:
        processed_data.append([row['ID'], timestamp_str, 'human_mushroom_task'])
    elif 'final decision in mushroom' in obs_text:
        processed_data.append([row['ID'], timestamp_str, 'final_mushroom_task'])

# Save the processed data to a CSV file
with open('task_click_times.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'timestamp', 'event'])
    writer.writerows(processed_data)



# We can divide the task in
# 1. click_start/start_time to click_to_tutorial (before tutorial)
# 2. click_to_tutorial to click_to_task (tutorial)
# 3. click_to_task to final_decision_in_mushroom (1st task)
# 4. final_decision_in_mushroom to final_deicision_in_mushroom (2nd to nth task)

# can be interesting to see the time spent on each task
# can be interesting to see the fixation between human_decision and final_decision_in_mushroom (thinking time)











# Parse the date string into a datetime object
# start_object = datetime.strptime(start_time, date_format)
# start_timestamp = start_object.timestamp()
