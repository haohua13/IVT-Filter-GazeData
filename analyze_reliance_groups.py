import pandas as pd
import os
def classify_over_reliance(row):
    if row['perimeter_o_reliance'] == 1:
        if row['overreliance'] == 1: # if overreliance, then it is also reliance
            return 1
        else:
            return 0 # if no overreliance, then it is no reliance
    return 'undefined' # can't be classified
def classify_under_reliance(row):

    if row['perimeter_u_reliance'] == 1:
        if row['underreliance'] == 1:
            return 1 # if underreliance, then it is also no reliance
        else:
            return 0 # if no underreliance, then it is reliance
        
    return 'undefined'

def classify_reliance(row):
    if row['perimeter_reliance'] == 1:
        if row['reliance'] == 1:
            return 1
        else:
            return 0
    return 'undefined'

def plot_time_series(data, group):
    pass

def obtain_dataset(data):
        data['reliance'] = data.apply(classify_reliance, axis=1)
        data['overreliance'] = data.apply(classify_over_reliance, axis = 1)
        data['underreliance'] = data.apply(classify_under_reliance, axis = 1)
        data['task'] = data['Round'] 
        # save only current_id, group and Round
        data = data[['ID', 'reliance', 'overreliance', 'underreliance', 'task']]
        # create directory
        if not os.path.exists('gaze_data/reliance_groups'):
            os.makedirs('gaze_data/reliance_groups')
        data.to_csv('gaze_data/reliance_groups/groups.csv', index=False)

            
if __name__ == '__main__':
    file = pd.read_csv('gaze_data/data_melted_elmo_spss.csv')
    conditions = ['O20', 'O21', 'O22', 'O24', 'O25', 'O26','O27', 'O28','O29','O30',\
                  'O31','O32','O33','O34','OX20', 'OX21', 'OX22','OX23', 'OX24', 'OX25', 'OX26','OX27',\
                  'OX28','OX29','OX31','OX32','OX33','OX34']
    obtain_dataset(file)
    reliance_data = pd.read_csv('gaze_data/reliance_groups/groups.csv')
    interval_data = pd.read_csv('mushroom_data/interval_data_by_tasks_with_average_durations.csv')

    # get all reliance groups (overreliance and no underreliance)
    over_reliance = reliance_data[(reliance_data['overreliance'] == '1')]

    no_overreliance = reliance_data[(reliance_data['overreliance'] == '0')]
    no_reliance = reliance_data[(reliance_data['reliance'] == '0')]
    reliance = reliance_data[reliance_data['reliance'] == '1']
    under_reliance = reliance_data[(reliance_data['underreliance'] == '1')]
    no_under_reliance = reliance_data[(reliance_data['underreliance'] == '0')]

    # get all intervals and gaze data for each group, ID and task
    over_reliance_intervals = interval_data[interval_data['ID']].isin(over_reliance['ID'])
    no_over_reliance_intervals = interval_data[interval_data['ID']].isin(no_overreliance['ID'])
    no_reliance_intervals = interval_data[interval_data['ID']].isin(no_reliance['ID'])
    reliance_intervals = interval_data[interval_data['ID']].isin(reliance['ID'])
    under_reliance_intervals = interval_data[interval_data['ID']].isin(under_reliance['ID'])
    no_under_reliance_intervals = interval_data[interval_data['ID']].isin(no_under_reliance['ID'])    
    print(over_reliance_intervals)

    # time plot






    # obtain fixation durations and counts for each group, independent of task







    

    

        






