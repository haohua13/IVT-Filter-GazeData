import pandas as pd
import os
import matplotlib.pyplot as plt

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

def plot_over_reliance(data1, group1, data2, group2):
    # obtain fixation durations and counts for task 5, 10, 13 for the given group (over_reliance)
    data1_task5 = data1[data1['task'] == 5] 
    data1_task10 = data1[data1['task'] == 10]
    data1_task13 = data1[data1['task'] == 13]
    data2_task5 = data2[data2['task'] == 5]
    data2_task10 = data2[data2['task'] == 10]
    data2_task13 = data2[data2['task'] == 13]
    fixation_durations_task_5 = data1_task5['fixation_durations'].mean()
    fixation_counts_task_5 = data1_task5['fixation_counts'].mean()
    fixation_durations_task_10 = data1_task10['fixation_durations'].mean()
    fixation_counts_task_10 = data1_task10['fixation_counts'].mean()
    fixation_durations_task_13 = data1_task13['fixation_durations'].mean()
    fixation_counts_task_13 = data1_task13['fixation_counts'].mean()
    fixation_durations_task_5_no = data2_task5['fixation_durations'].mean()
    fixation_counts_task_5_no = data2_task5['fixation_counts'].mean()
    fixation_durations_task_10_no = data2_task10['fixation_durations'].mean()
    fixation_counts_task_10_no = data2_task10['fixation_counts'].mean()
    fixation_durations_task_13_no = data2_task13['fixation_durations'].mean()
    fixation_counts_task_13_no = data2_task13['fixation_counts'].mean()
    x_position = [5, 10, 13]
    plt.figure(figsize=(10, 6))
    # plot fixation durations
    plt.scatter(x_position, [fixation_durations_task_5, fixation_durations_task_10, fixation_durations_task_13], marker = 'o')
    plt.scatter(x_position, [fixation_durations_task_5_no, fixation_durations_task_10_no, fixation_durations_task_13_no], marker = 'o')
    plt.plot(x_position, [fixation_durations_task_5, fixation_durations_task_10, fixation_durations_task_13], linestyle = '-')
    plt.plot(x_position, [fixation_durations_task_5_no, fixation_durations_task_10_no, fixation_durations_task_13_no], linestyle = '--')
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color = 'tab:blue', markersize=10, label=group1),
               plt.Line2D([0], [0], marker='o', color = 'tab:orange',markersize=10, label=group2)
    ]
    plt.legend(handles=handles)
    plt.xlabel('Task')
    # only show x_positions
    plt.xticks(x_position)
    plt.ylabel('Computer Avg. Fixation Duration [s]')
    plt.title('Overreliance Group')
    plt.grid()
    plt.savefig('reliance_images/overreliance_fixation_duration.png', dpi = 1600)

    # plot fixation counts
    plt.figure(figsize=(10, 6))
    plt.scatter(x_position, [fixation_counts_task_5, fixation_counts_task_10, fixation_counts_task_13], marker = 'o')
    plt.scatter(x_position, [fixation_counts_task_5_no, fixation_counts_task_10_no, fixation_counts_task_13_no], marker = 'o')
    plt.plot(x_position, [fixation_counts_task_5, fixation_counts_task_10, fixation_counts_task_13], linestyle = '-')
    plt.plot(x_position, [fixation_counts_task_5_no, fixation_counts_task_10_no, fixation_counts_task_13_no], linestyle = '--')
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color = 'tab:blue', markersize=10, label=group1),
               plt.Line2D([0], [0], marker='o', color = 'tab:orange',markersize=10, label=group2)
    ]
    plt.legend(handles=handles)
    plt.xlabel('Task')
    # only show x_positions
    plt.xticks(x_position)
    plt.ylabel('Computer Avg. Fixation Count')
    plt.title('Overreliance Group')
    plt.grid()
    plt.savefig('reliance_images/overreliance_fixation_count.png', dpi = 1600)

    # elmo data
    elmo_fixation_durations_task_5 = data1_task5['fixation_durations_elmo'].mean()
    elmo_fixation_counts_task_5 = data1_task5['fixation_counts_elmo'].mean()
    elmo_fixation_durations_task_10 = data1_task10['fixation_durations_elmo'].mean()
    elmo_fixation_counts_task_10 = data1_task10['fixation_counts_elmo'].mean()
    elmo_fixation_durations_task_13 = data1_task13['fixation_durations_elmo'].mean()
    elmo_fixation_counts_task_13 = data1_task13['fixation_counts_elmo'].mean()
    elmo_fixation_durations_task_5_no = data2_task5['fixation_durations_elmo'].mean()

    elmo_fixation_counts_task_5_no = data2_task5['fixation_counts_elmo'].mean()
    elmo_fixation_durations_task_10_no = data2_task10['fixation_durations_elmo'].mean()
    elmo_fixation_counts_task_10_no = data2_task10['fixation_counts_elmo'].mean()
    elmo_fixation_durations_task_13_no = data2_task13['fixation_durations_elmo'].mean()
    elmo_fixation_counts_task_13_no = data2_task13['fixation_counts_elmo'].mean()

    plt.figure(figsize=(10, 6))
    # plot fixation durations
    plt.scatter(x_position, [elmo_fixation_durations_task_5, elmo_fixation_durations_task_10, elmo_fixation_durations_task_13], marker = 'o')
    plt.scatter(x_position, [elmo_fixation_durations_task_5_no, elmo_fixation_durations_task_10_no, elmo_fixation_durations_task_13_no], marker = 'o')
    plt.plot(x_position, [elmo_fixation_durations_task_5, elmo_fixation_durations_task_10, elmo_fixation_durations_task_13], linestyle = '-')
    plt.plot(x_position, [elmo_fixation_durations_task_5_no, elmo_fixation_durations_task_10_no, elmo_fixation_durations_task_13_no], linestyle = '--')
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color = 'tab:blue', markersize=10, label=group1),
                plt.Line2D([0], [0], marker='o', color = 'tab:orange',markersize=10, label=group2)
     ]
    plt.legend(handles=handles)
    plt.xlabel('Task')
    # only show x_positions
    plt.xticks(x_position)
    plt.ylabel('Elmo Avg. Fixation Duration [s]')
    plt.title('Overreliance Group')
    plt.grid()
    plt.savefig('reliance_images/overreliance_elmo_fixation_duration.png', dpi = 1600)

    # plot fixation counts
    plt.figure(figsize=(10, 6))
    plt.scatter(x_position, [elmo_fixation_counts_task_5, elmo_fixation_counts_task_10, elmo_fixation_counts_task_13], marker = 'o')
    plt.scatter(x_position, [elmo_fixation_counts_task_5_no, elmo_fixation_counts_task_10_no, elmo_fixation_counts_task_13_no], marker = 'o')

    plt.plot(x_position, [elmo_fixation_counts_task_5, elmo_fixation_counts_task_10, elmo_fixation_counts_task_13], linestyle = '-')
    plt.plot(x_position, [elmo_fixation_counts_task_5_no, elmo_fixation_counts_task_10_no, elmo_fixation_counts_task_13_no], linestyle = '--')
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color = 'tab:blue', markersize=10, label=group1),
                plt.Line2D([0], [0], marker='o', color = 'tab:orange',markersize=10, label=group2)
     ]
    plt.legend(handles=handles)
    plt.xlabel('Task')
    # only show x_positions
    plt.xticks(x_position)
    plt.ylabel('Elmo Avg. Fixation Count')
    plt.title('Overreliance Group')
    plt.grid()
    plt.savefig('reliance_images/overreliance_elmo_fixation_count.png', dpi = 1600)
    plt.show()

def plot_reliance(data1, group1, data2, group2):
    # For all tasks, plot fixation durations and counts for the given group
    fixation_durations = []
    fixation_counts = []
    fixation_durations_no = []
    fixation_counts_no = []
    for i in range(1, 16):
        data1_task = data1[data1['task'] == i]
        data2_task = data2[data2['task'] == i]
        fixation_durations.append(data1_task['fixation_durations'].mean())
        fixation_counts.append(data1_task['fixation_counts'].mean())
        fixation_durations_no.append(data2_task['fixation_durations'].mean())
        fixation_counts_no.append(data2_task['fixation_counts'].mean())

    # tasks 1-15
    x_position = [i for i in range(1, 16)]
    plt.figure(figsize=(10, 6))
    # plot fixation durations
    plt.scatter(x_position, fixation_durations, marker = 'o')
    plt.scatter(x_position, fixation_durations_no, marker = 'o')
    plt.plot(x_position, fixation_durations, linestyle = '-')
    plt.plot(x_position, fixation_durations_no, linestyle = '--')
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color = 'tab:blue', markersize=10, label=group1),
                plt.Line2D([0], [0], marker='o', color = 'tab:orange',markersize=10, label=group2)
     ]
    plt.legend(handles=handles)
    plt.xlabel('Task')
    # only show x_positions
    plt.xticks(x_position)
    plt.ylabel('Computer Avg. Fixation Duration [s]')
    plt.title('Reliance Group')
    plt.grid()
    plt.savefig('reliance_images/reliance_fixation_duration.png', dpi = 1600)

    # plot fixation counts
    plt.figure(figsize=(10, 6))
    plt.scatter(x_position, fixation_counts, marker = 'o')
    plt.scatter(x_position, fixation_counts_no, marker = 'o')
    plt.plot(x_position, fixation_counts, linestyle = '-')
    plt.plot(x_position, fixation_counts_no, linestyle = '--')
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color = 'tab:blue', markersize=10, label=group1),
                plt.Line2D([0], [0], marker='o', color = 'tab:orange',markersize=10, label=group2)
     ]
    plt.legend(handles=handles)
    plt.xlabel('Task')
    # only show x_positions
    plt.xticks(x_position)
    plt.ylabel('Computer Avg. Fixation Count')
    plt.title('Reliance Group')
    plt.grid()
    plt.savefig('reliance_images/reliance_fixation_count.png', dpi = 1600)

    # elmo data

    elmo_fixation_durations = []
    elmo_fixation_counts = []
    elmo_fixation_durations_no = []
    elmo_fixation_counts_no = []
    for i in range(1, 16):
        data1_task = data1[data1['task'] == i]
        data2_task = data2[data2['task'] == i]
        elmo_fixation_durations.append(data1_task['fixation_durations_elmo'].mean())
        elmo_fixation_counts.append(data1_task['fixation_counts_elmo'].mean())
        elmo_fixation_durations_no.append(data2_task['fixation_durations_elmo'].mean())
        elmo_fixation_counts_no.append(data2_task['fixation_counts_elmo'].mean())

    plt.figure(figsize=(10, 6))
    # plot fixation durations
    plt.scatter(x_position, elmo_fixation_durations, marker = 'o')
    plt.scatter(x_position, elmo_fixation_durations_no, marker = 'o')
    plt.plot(x_position, elmo_fixation_durations, linestyle = '-')
    plt.plot(x_position, elmo_fixation_durations_no, linestyle = '--')
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color = 'tab:blue', markersize=10, label=group1),
                plt.Line2D([0], [0], marker='o', color = 'tab:orange',markersize=10, label=group2)
     ]
    plt.legend(handles=handles)
    plt.xlabel('Task')
    # only show x_positions
    plt.xticks(x_position)
    plt.ylabel('Elmo Avg. Fixation Duration [s]')
    plt.title('Reliance Group')
    plt.grid()
    plt.savefig('reliance_images/reliance_elmo_fixation_duration.png', dpi = 1600)

    # plot fixation counts
    plt.figure(figsize=(10, 6))
    plt.scatter(x_position, elmo_fixation_counts, marker = 'o')
    plt.scatter(x_position, elmo_fixation_counts_no, marker = 'o')
    plt.plot(x_position, elmo_fixation_counts, linestyle = '-')
    plt.plot(x_position, elmo_fixation_counts_no, linestyle = '--')
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color = 'tab:blue', markersize=10, label=group1),
                plt.Line2D([0], [0], marker='o', color = 'tab:orange',markersize=10, label=group2)
     ]
    plt.legend(handles=handles)
    plt.xlabel('Task')
    # only show x_positions
    plt.xticks(x_position)
    plt.ylabel('Elmo Avg. Fixation Count')
    plt.title('Reliance Group')
    plt.grid()
    plt.savefig('reliance_images/reliance_elmo_fixation_count.png', dpi = 1600)
    plt.show()


def plot_underreliance(data1, group1, data2, group2):

    # For all tasks, plot fixation durations and counts for the given group
    fixation_durations = []
    fixation_counts = []
    fixation_durations_no = []
    fixation_counts_no = []
    for i in range(1, 16) :
        if i in [5, 10, 13]:
            continue
        data1_task = data1[data1['task'] == i]
        data2_task = data2[data2['task'] == i]
        fixation_durations.append(data1_task['fixation_durations'].mean())
        fixation_counts.append(data1_task['fixation_counts'].mean())
        fixation_durations_no.append(data2_task['fixation_durations'].mean())
        fixation_counts_no.append(data2_task['fixation_counts'].mean())

    # tasks 1-15, except 5, 10, 13
    x_position = [i for i in range(1, 16) if i not in [5, 10, 13]]
    plt.figure(figsize=(10, 6))
    # plot fixation durations
    plt.scatter(x_position, fixation_durations, marker = 'o')
    plt.scatter(x_position, fixation_durations_no, marker = 'o')
    plt.plot(x_position, fixation_durations, linestyle = '-')
    plt.plot(x_position, fixation_durations_no, linestyle = '--')
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color = 'tab:blue', markersize=10, label=group1),
                plt.Line2D([0], [0], marker='o', color = 'tab:orange',markersize=10, label=group2)
     ]
    plt.legend(handles=handles)
    plt.xlabel('Task')
    # only show x_positions
    plt.xticks(x_position)
    plt.ylabel('Computer Avg. Fixation Duration [s]')
    plt.title('Underreliance Group')
    plt.grid()
    plt.savefig('reliance_images/underreliance_fixation_duration.png', dpi = 1600)

    # plot fixation counts
    plt.figure(figsize=(10, 6))
    plt.scatter(x_position, fixation_counts, marker = 'o')
    plt.scatter(x_position, fixation_counts_no, marker = 'o')
    plt.plot(x_position, fixation_counts, linestyle = '-')
    plt.plot(x_position, fixation_counts_no, linestyle = '--')
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color = 'tab:blue', markersize=10, label=group1),
                plt.Line2D([0], [0], marker='o', color = 'tab:orange',markersize=10, label=group2)
     ]
    plt.legend(handles=handles)
    plt.xlabel('Task')
    # only show x_positions
    plt.xticks(x_position)
    plt.ylabel('Computer Avg. Fixation Count')
    plt.title('Underreliance Group')
    plt.grid()
    plt.savefig('reliance_images/underreliance_fixation_count.png', dpi = 1600)

    # elmo data

    elmo_fixation_durations = []
    elmo_fixation_counts = []
    elmo_fixation_durations_no = []
    elmo_fixation_counts_no = []
    for i in range(1, 16):
        if i in [5, 10, 13]:
            continue
        data1_task = data1[data1['task'] == i]
        data2_task = data2[data2['task'] == i]
        elmo_fixation_durations.append(data1_task['fixation_durations_elmo'].mean())
        elmo_fixation_counts.append(data1_task['fixation_counts_elmo'].mean())
        elmo_fixation_durations_no.append(data2_task['fixation_durations_elmo'].mean())
        elmo_fixation_counts_no.append(data2_task['fixation_counts_elmo'].mean())

    plt.figure(figsize=(10, 6))
    # plot fixation durations
    plt.scatter(x_position, elmo_fixation_durations, marker = 'o')
    plt.scatter(x_position, elmo_fixation_durations_no, marker = 'o')
    plt.plot(x_position, elmo_fixation_durations, linestyle = '-')
    plt.plot(x_position, elmo_fixation_durations_no, linestyle = '--')
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color = 'tab:blue', markersize=10, label=group1),
                plt.Line2D([0], [0], marker='o', color = 'tab:orange',markersize=10, label=group2)
     ]
    plt.legend(handles=handles)
    plt.xlabel('Task')
    # only show x_positions
    plt.xticks(x_position)
    plt.ylabel('Elmo Avg. Fixation Duration [s]')
    plt.title('Underreliance Group')
    plt.grid()
    plt.savefig('reliance_images/underreliance_elmo_fixation_duration.png', dpi = 1600)

    # plot fixation counts
    plt.figure(figsize=(10, 6))
    plt.scatter(x_position, elmo_fixation_counts, marker = 'o')
    plt.scatter(x_position, elmo_fixation_counts_no, marker = 'o')
    plt.plot(x_position, elmo_fixation_counts, linestyle = '-')
    plt.plot(x_position, elmo_fixation_counts_no, linestyle = '--')
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color = 'tab:blue', markersize=10, label=group1),
                plt.Line2D([0], [0], marker='o', color = 'tab:orange',markersize=10, label=group2)
     ]
    plt.legend(handles=handles)
    plt.xlabel('Task')
    # only show x_positions
    plt.xticks(x_position)
    
    plt.ylabel('Elmo Avg. Fixation Count')
    plt.title('Underreliance Group')
    plt.grid()
    plt.savefig('reliance_images/underreliance_elmo_fixation_count.png', dpi = 1600)
    plt.show()

    
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

def get_XAI(idd):
    if 'X' in idd:
        return 1
    else:
        return 0
    
def get_group(group, xai):
    if group == 1 and xai == 1:
        return 1
    elif group == 1 and xai == 0:
        return 0
    elif group == 0 and xai == 1:
        return 10
    else:
        return 11

            
if __name__ == '__main__':
    file = pd.read_csv('gaze_data/data_melted_elmo_spss.csv')
    conditions = ['O20', 'O21', 'O22', 'O24', 'O25', 'O26','O27', 'O28','O29','O30',\
                  'O31','O32','O33','O34','OX20', 'OX21', 'OX22','OX23', 'OX24', 'OX25', 'OX26','OX27',\
                  'OX28','OX29','OX31','OX32','OX33','OX34']
    

    obtain_dataset(file)
    reliance_data = pd.read_csv('gaze_data/reliance_groups/groups.csv')

    # remove 'O23' or user == 'OX30' from the dataset
    reliance_data = reliance_data[~reliance_data['ID'].isin(['O23', 'OX30'])]

    interval_data = pd.read_csv('mushroom_data/interval_data_by_tasks_with_average_durations.csv')

    # get all reliance groups (overreliance and no underreliance)
    over_reliance = reliance_data[(reliance_data['overreliance'] == '1')]
    no_overreliance = reliance_data[(reliance_data['overreliance'] == '0')]

    no_reliance = reliance_data[(reliance_data['reliance'] == '0')]
    reliance = reliance_data[reliance_data['reliance'] == '1']

    under_reliance = reliance_data[(reliance_data['underreliance'] == '1')]
    no_under_reliance = reliance_data[(reliance_data['underreliance'] == '0')]

    # Filter interval_data to get rows where both ID and task are in group
    over_reliance_ids = over_reliance['ID'].tolist()
    over_reliance_tasks = over_reliance['task'].tolist()

    # for each (id, task) pair, find interval_data and save it
    over_reliance_pairs = set(zip(over_reliance_ids, over_reliance_tasks))
    over_reliance_intervals = interval_data[
        interval_data.apply(lambda row: (row['ID'], row['task']) in over_reliance_pairs, axis=1)
    ]
    no_over_reliance_ids = no_overreliance['ID'].tolist()
    no_over_reliance_tasks = no_overreliance['task'].tolist()
    no_over_reliance_pairs = set(zip(no_over_reliance_ids, no_over_reliance_tasks))
    no_over_reliance_intervals = interval_data[
        interval_data.apply(lambda row: (row['ID'], row['task']) in no_over_reliance_pairs, axis=1)
    ]
    # save no_over_reliance_intervals and over_reliance_intervals
    over_reliance_intervals['XAI_value'] = over_reliance_intervals['ID'].apply(get_XAI)
    over_reliance_intervals['overreliance'] = 1
    no_over_reliance_intervals['XAI_value'] = no_over_reliance_intervals['ID'].apply(get_XAI)
    no_over_reliance_intervals['overreliance'] = 0

    over_reliance_intervals['condition'] = over_reliance_intervals.apply(lambda row: get_group(row['overreliance'], row['XAI_value']), axis=1)
    no_over_reliance_intervals['condition'] = no_over_reliance_intervals.apply(lambda row: get_group(row['overreliance'], row['XAI_value']), axis=1)

    over_reliance_intervals.to_csv('gaze_data/reliance_groups/over_reliance_intervals.csv', index=False)
    no_over_reliance_intervals.to_csv('gaze_data/reliance_groups/no_over_reliance_intervals.csv', index=False)

    # append over_reliance_intervals and no_over_reliance_intervals
    over_reliance_global = pd.concat([over_reliance_intervals, no_over_reliance_intervals])
    over_reliance_global.to_csv('gaze_data/reliance_groups/over_reliance_global.csv', index=False)

    # plot_over_reliance(over_reliance_intervals, 'Overreliance', no_over_reliance_intervals, 'No Overreliance')
    print('Reliance: ', reliance.shape)
    reliance_ids = reliance['ID'].tolist()
    reliance_tasks = reliance['task'].tolist()
    reliance_pairs = set(zip(reliance_ids, reliance_tasks))
    reliance_intervals = interval_data[
        interval_data.apply(lambda row: (row['ID'], row['task']) in reliance_pairs, axis=1)
    ]
    no_reliance_ids = no_reliance['ID'].tolist()
    no_reliance_tasks = no_reliance['task'].tolist()
    no_reliance_pairs = set(zip(no_reliance_ids, no_reliance_tasks))
    no_reliance_intervals = interval_data[
        interval_data.apply(lambda row: (row['ID'], row['task']) in no_reliance_pairs, axis=1)
    ]

    print('No Reliance: ', no_reliance.shape)
    # print size of reliance_intervals and no_reliance_intervals
    print('Reliance Group: ', reliance_intervals.shape)
    print('No Reliance Group: ', no_reliance_intervals.shape)
    reliance_intervals['XAI_value'] = reliance_intervals['ID'].apply(get_XAI)
    no_reliance_intervals['XAI_value'] = no_reliance_intervals['ID'].apply(get_XAI)
    reliance_intervals['reliance'] = 1
    no_reliance_intervals['reliance'] = 0

    reliance_intervals['condition'] = reliance_intervals.apply(lambda row: get_group(row['reliance'], row['XAI_value']), axis=1)
    no_reliance_intervals['condition'] = no_reliance_intervals.apply(lambda row: get_group(row['reliance'], row['XAI_value']), axis=1)


    reliance_intervals.to_csv('gaze_data/reliance_groups/reliance_intervals.csv', index=False)
    no_reliance_intervals.to_csv('gaze_data/reliance_groups/no_reliance_intervals.csv', index=False)

    reliance_global = pd.concat([reliance_intervals, no_reliance_intervals])
    reliance_global.to_csv('gaze_data/reliance_groups/reliance_global.csv', index=False)



    # plot_reliance(reliance_intervals, 'Reliance', no_reliance_intervals, 'No Reliance')

    under_reliance_ids = under_reliance['ID'].tolist()
    under_reliance_tasks = under_reliance['task'].tolist()
    under_reliance_pairs = set(zip(under_reliance_ids, under_reliance_tasks))
    under_reliance_intervals = interval_data[
        interval_data.apply(lambda row: (row['ID'], row['task']) in under_reliance_pairs, axis=1)
    ]
    no_under_reliance_ids = no_under_reliance['ID'].tolist()
    no_under_reliance_tasks = no_under_reliance['task'].tolist()
    no_under_reliance_pairs = set(zip(no_under_reliance_ids, no_under_reliance_tasks))
    no_under_reliance_intervals = interval_data[
        interval_data.apply(lambda row: (row['ID'], row['task']) in no_under_reliance_pairs, axis=1)
    ]
    print('Underreliance: ', under_reliance.shape)
    print('No Underreliance: ', no_under_reliance.shape)
    # print size of under_reliance_intervals and no_under_reliance
    print('Underreliance Group: ', under_reliance_intervals.shape)
    print('No Underreliance Group: ', no_under_reliance_intervals.shape)
    under_reliance_intervals['XAI_value'] = under_reliance_intervals['ID'].apply(get_XAI)
    no_under_reliance_intervals['XAI_value'] = no_under_reliance_intervals['ID'].apply(get_XAI)
    under_reliance_intervals['underreliance'] = 1
    no_under_reliance_intervals['underreliance'] = 0

    under_reliance_intervals['condition'] = under_reliance_intervals.apply(lambda row: get_group(row['underreliance'], row['XAI_value']), axis=1)
    no_under_reliance_intervals['condition'] = no_under_reliance_intervals.apply(lambda row: get_group(row['underreliance'], row['XAI_value']), axis=1)


    under_reliance_intervals.to_csv('gaze_data/reliance_groups/under_reliance_intervals.csv', index=False)
    no_under_reliance_intervals.to_csv('gaze_data/reliance_groups/no_under_reliance_intervals.csv', index=False)

    under_reliance_global = pd.concat([under_reliance_intervals, no_under_reliance_intervals])
    under_reliance_global.to_csv('gaze_data/reliance_groups/under_reliance_global.csv', index=False)


    # plot_underreliance(under_reliance_intervals, 'Underreliance', no_under_reliance_intervals, 'No Underreliance')








    

    

        






