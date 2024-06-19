import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

condition_o = {'O20', 'O21', 'O22', 'O24', 'O25', 'O26','O27', 'O28','O29','O30','O31','O32','O33','O34'}
condition_ox = {'OX20', 'OX21', 'OX22','OX23', 'OX24', 'OX25', 'OX26','OX27', 'OX28','OX29','OX31','OX32','OX33','OX34'}

def box_plot_mean_fixation(o_data, ox_data):
    plt.figure(figsize=(10, 6))
    # boxplot for computer_average_fixation_duration
    box1 = plt.boxplot([o_data['computer_average_fixation_duration'], ox_data['computer_average_fixation_duration']], labels=['O', 'OX'])
    plt.title('Computer Screen', fontsize = 18)
    plt.ylabel('Mean Fixation Duration [s]', fontsize = 14)
    plt.xlabel('Group', fontsize = 16)
    plt.grid(True)  # Show grid
    # extracting mean and standard deviation for computer_average_fixation_duration
    mean_computer = [box1['medians'][0].get_ydata()[0], box1['medians'][1].get_ydata()[0]]
    std_computer = [box1['whiskers'][0].get_ydata()[1], box1['whiskers'][1].get_ydata()[1]]
    print('Computer Mean Fixation Duration:')
    print('Mean:')
    print('O:', mean_computer[0])
    print('OX:', mean_computer[1])
    print('Standard Deviation:')
    print('O:', std_computer[0])
    print('OX:', std_computer[1])
    
    plt.savefig('results/computer_average_fixation_duration.png', dpi = 1200, bbox_inches = 'tight')


    # boxplot for elmo_average_fixation_duration
    plt.figure(figsize=(10, 6))
    box2 = plt.boxplot([o_data['elmo_average_fixation_duration'], ox_data['elmo_average_fixation_duration']], labels=['O', 'OX'])
    plt.title('Elmo Robot', fontsize = 18)
    plt.ylabel('Mean Fixation Duration [s]', fontsize = 14)
    plt.xlabel('Group', fontsize = 16)
    plt.grid()
    # extracting mean and standard deviation for elmo_average_fixation_duration
    mean_elmo = [box2['medians'][0].get_ydata()[0], box2['medians'][1].get_ydata()[0]]
    std_elmo = [box2['whiskers'][0].get_ydata()[1], box2['whiskers'][1].get_ydata()[1]]
    
    print('\nElmo Mean Fixation Duration:')
    print('Mean:')
    print('O:', mean_elmo[0])
    print('OX:', mean_elmo[1])
    print('Standard Deviation:')
    print('O:', std_elmo[0])
    print('OX:', std_elmo[1])
    plt.tight_layout()
    plt.savefig('results/elmo_average_fixation_duration.png', dpi = 1200, bbox_inches = 'tight')

def box_plot_total_fixation(o_data, ox_data):
    o_normalized_total_fixation = o_data['computer_total_fixation_duration']/o_data['game_time']*100
    o_normalized_total_fixation_elmo = o_data['elmo_total_fixation_duration']/o_data['game_time']*100
    ox_normalized_total_fixation = ox_data['computer_total_fixation_duration']/ox_data['game_time']*100
    ox_normalized_total_fixation_elmo = ox_data['elmo_total_fixation_duration']/ox_data['game_time']*100
    plt.figure(figsize=(10, 6))
    # boxplot for computer_total_fixation_duration
    box1 = plt.boxplot([o_normalized_total_fixation, ox_normalized_total_fixation], labels=['O', 'OX'])
    plt.title('Computer Screen', fontsize = 18)
    plt.ylabel('Fixation Duration [%]', fontsize = 14)
    plt.xlabel('Group', fontsize = 16)
    plt.grid(True)
    plt.savefig('results/computer_total_fixation_duration.png', dpi = 1200, bbox_inches = 'tight')
    # extracting mean and standard deviation for computer_total_fixation_duration
    mean_computer = [box1['medians'][0].get_ydata()[0], box1['medians'][1].get_ydata()[0]]
    std_computer = [box1['whiskers'][0].get_ydata()[1], box1['whiskers'][1].get_ydata()[1]]
    print('Computer Total Fixation Duration:')
    print('Mean:')
    print('O:', mean_computer[0])
    print('OX:', mean_computer[1])
    print('Standard Deviation:')
    print('O:', std_computer[0])
    print('OX:', std_computer[1])
    # boxplot for elmo_total_fixation_duration
    plt.figure(figsize=(10, 6))
    box2 = plt.boxplot([o_normalized_total_fixation_elmo, ox_normalized_total_fixation_elmo], labels=['O', 'OX'])
    plt.title('Elmo Robot', fontsize = 18)
    plt.ylabel('Fixation Duration [%]', fontsize = 14)
    plt.xlabel('Group', fontsize = 16)
    plt.grid()

    # extracting mean and standard deviation for elmo_total_fixation_duration
    mean_elmo = [box2['medians'][0].get_ydata()[0], box2['medians'][1].get_ydata()[0]]
    std_elmo = [box2['whiskers'][0].get_ydata()[1], box2['whiskers'][1].get_ydata()[1]]
    print('\nElmo Total Fixation Duration:')
    print('Mean:')
    print('O:', mean_elmo[0])
    print('OX:', mean_elmo[1])
    print('Standard Deviation:')
    print('O:', std_elmo[0])
    print('OX:', std_elmo[1])
    plt.tight_layout()
    plt.savefig('results/elmo_total_fixation_duration.png', dpi = 1200, bbox_inches = 'tight')
    
def box_plot_fixation_count(o_data, ox_data):
    o_normalized_total_count = o_data['computer_fixation_count']/o_data['game_time']
    o_normalized_total_count_elmo = o_data['elmo_fixation_count']/o_data['game_time']
    ox_normalized_total_count = ox_data['computer_fixation_count']/ox_data['game_time']
    ox_normalized_total_count_elmo = ox_data['elmo_fixation_count']/ox_data['game_time']
    plt.figure(figsize=(10, 6))
    # boxplot for computer_total_fixation_count
    box1 = plt.boxplot([o_normalized_total_count, ox_normalized_total_count], labels=['O', 'OX'])
    plt.title('Computer Screen', fontsize = 18)
    plt.ylabel('Fixation Count Rate', fontsize = 14)
    plt.xlabel('Group', fontsize = 16)
    plt.grid(True)
    # extracting mean and standard deviation for computer_total_fixation_count
    mean_computer = [box1['medians'][0].get_ydata()[0], box1['medians'][1].get_ydata()[0]]
    std_computer = [box1['whiskers'][0].get_ydata()[1], box1['whiskers'][1].get_ydata()[1]]
    print('Computer Total Fixation Count:')
    print('Mean:')
    print('O:', mean_computer[0])
    print('OX:', mean_computer[1])
    print('Standard Deviation:')
    print('O:', std_computer[0])
    print('OX:', std_computer[1])
    plt.savefig('results/computer_total_fixation_count.png', dpi = 1200, bbox_inches = 'tight')
    # boxplot for elmo_total_fixation_count
    plt.figure(figsize=(10, 6))
    box2 = plt.boxplot([o_normalized_total_count_elmo, ox_normalized_total_count_elmo], labels=['O', 'OX'])
    plt.title('Elmo Robot', fontsize = 18)
    plt.ylabel('Fixation Count Rate', fontsize = 14)
    plt.xlabel('Group', fontsize = 16)
    plt.grid()
    # extracting mean and standard deviation for elmo_total_fixation_count
    mean_elmo = [box2['medians'][0].get_ydata()[0], box2['medians'][1].get_ydata()[0]]
    std_elmo = [box2['whiskers'][0].get_ydata()[1], box2['whiskers'][1].get_ydata()[1]]
    print('\nElmo Total Fixation Count:')
    print('Mean:')
    print('O:', mean_elmo[0])
    print('OX:', mean_elmo[1])
    print('Standard Deviation:')
    print('O:', std_elmo[0])
    print('OX:', std_elmo[1])
    plt.tight_layout()
    plt.savefig('results/elmo_total_fixation_count.png', dpi = 1200, bbox_inches = 'tight')


def box_plot_count_tension(o_data, ox_data):
    o_count_tension = o_data['elmo_fixation_count']/o_data['computer_fixation_count']
    ox_count_tension = ox_data['elmo_fixation_count']/ox_data['computer_fixation_count']
    plt.figure(figsize=(10, 6))
    # boxplot for count_tension
    box1 = plt.boxplot([o_count_tension, ox_count_tension], labels=['O', 'OX'])
    plt.title('Relative Fixation Count', fontsize = 18)
    plt.ylabel('Fixation Count Tension', fontsize = 14)
    plt.xlabel('Group', fontsize = 16)
    plt.grid(True)
    # extracting mean and standard deviation for count_tension
    mean_computer = [box1['medians'][0].get_ydata()[0], box1['medians'][1].get_ydata()[0]]
    std_computer = [box1['whiskers'][0].get_ydata()[1], box1['whiskers'][1].get_ydata()[1]]
    print('Count Tension:')
    print('Mean:')
    print('O:', mean_computer[0])
    print('OX:', mean_computer[1])
    print('Standard Deviation:')
    print('O:', std_computer[0])
    print('OX:', std_computer[1])
    plt.savefig('results/count_tension.png', dpi = 1200, bbox_inches = 'tight')

def box_plot_time_tension(o_data, ox_data):
    o_time_tension = o_data['elmo_total_fixation_duration']/o_data['computer_total_fixation_duration']
    ox_time_tension = ox_data['elmo_total_fixation_duration']/ox_data['computer_total_fixation_duration']
    plt.figure(figsize=(10, 6))
    # boxplot for time_tension
    box1 = plt.boxplot([o_time_tension, ox_time_tension], labels=['O', 'OX'])
    plt.title('Relative Fixation Duration', fontsize = 18)
    plt.ylabel('Fixation Duration Tension', fontsize = 14)
    plt.xlabel('Group', fontsize = 16)
    plt.grid(True)
    # extracting mean and standard deviation for time_tension
    mean_computer = [box1['medians'][0].get_ydata()[0], box1['medians'][1].get_ydata()[0]]
    std_computer = [box1['whiskers'][0].get_ydata()[1], box1['whiskers'][1].get_ydata()[1]]
    print('Time Tension:')
    print('Mean:')
    print('O:', mean_computer[0])
    print('OX:', mean_computer[1])
    print('Standard Deviation:')
    print('O:', std_computer[0])
    print('OX:', std_computer[1])
    plt.savefig('results/time_tension.png', dpi = 1200, bbox_inches = 'tight')

def plot_count_over_time(data):
    value = data[data['id'] == 'O20']
    game_percentage = value['game_percentage']
    fixation_counts = value['fixation_counts']
    fixation_counts_elmo = value['fixations_counts_elmo']
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(game_percentage, fixation_counts, marker = 'o', linestyle = '--', color='b', label='Computer')
    plt.plot(game_percentage, fixation_counts_elmo, marker = 'o', linestyle = '--', color='r', label='Elmo')

    # Add labels and title
    plt.xlabel('Task Progress [%]')
    plt.ylabel('Fixation Count')
    plt.title('Fixation Counts Over Time')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig('results/example_fixation_counts_over_time.png', dpi = 1200, bbox_inches = 'tight')

def plot_average_counts_over_time(o_data, ox_data):
    N = 14

    game_percentage = o_data['game_percentage']
    o_array_fixation_counts = []
    o_array_fixation_counts_elmo = []
    ox_array_fixation_counts = []
    ox_array_fixation_counts_elmo = []

    for j in range(0, 20):
        o_fixation_counts = 0
        o_fixation_counts_elmo = 0
        ox_fixation_counts = 0
        ox_fixation_counts_elmo = 0
        for i in range(0, len(game_percentage), 20):
            # print(o_data['fixation_counts'][i+j])
            o_fixation_counts += o_data['fixation_counts'][i+j]
            o_fixation_counts_elmo += o_data['fixations_counts_elmo'][i+j]
            ox_fixation_counts += ox_data['fixation_counts'][i+j+len(game_percentage)]
            ox_fixation_counts_elmo += ox_data['fixations_counts_elmo'][i+j+len(game_percentage)]     
        average_o_fixation_counts = o_fixation_counts / N
        average_o_fixation_counts_elmo = o_fixation_counts_elmo / N
        average_ox_fixation_counts = ox_fixation_counts / N
        average_ox_fixation_counts_elmo = ox_fixation_counts_elmo / N
        o_array_fixation_counts.append(average_o_fixation_counts)
        o_array_fixation_counts_elmo.append(average_o_fixation_counts_elmo)
        ox_array_fixation_counts.append(average_ox_fixation_counts)
        ox_array_fixation_counts_elmo.append(average_ox_fixation_counts_elmo)


    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(game_percentage[0:19], o_array_fixation_counts[0:19], marker = 'o', linestyle = '--', color='b', label='Computer O')
    plt.plot(game_percentage[0:19], o_array_fixation_counts_elmo[0:19], marker = 'o', linestyle = '--', color='r', label='Elmo O')
    plt.plot(game_percentage[0:19], ox_array_fixation_counts[0:19], marker = 'o', linestyle = '--', color='g', label='Computer OX')
    plt.plot(game_percentage[0:19], ox_array_fixation_counts_elmo[0:19], marker = 'o', linestyle = '--', color='y', label='Elmo OX')

    # Add labels and title
    plt.xlabel('Task Progress [%]')
    plt.ylabel('Fixation Count')
    plt.title('Fixation Counts Over Time')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 100)

    # Show the plot
    plt.tight_layout()
    plt.savefig('results/average_fixation_counts_over_time.png', dpi = 1200, bbox_inches = 'tight')

def plot_average_durations_over_time(o_data, ox_data):
    N = 14
    game_percentage = o_data['game_percentage']
    o_array_fixation_durations = []
    o_array_fixation_durations_elmo = []
    ox_array_fixation_durations = []
    ox_array_fixation_durations_elmo = []
    for j in range(0, 20):
        o_fixation_duration = 0
        o_fixation_duration_elmo = 0
        ox_fixation_duration = 0
        ox_fixation_duration_elmo = 0
        for i in range(0, len(game_percentage), 20):
            o_fixation_duration += o_data['fixation_durations'][i+j]
            o_fixation_duration_elmo += o_data['fixation_durations_elmo'][i+j]
            ox_fixation_duration += ox_data['fixation_durations'][i+j+len(game_percentage)]
            ox_fixation_duration_elmo += ox_data['fixation_durations_elmo'][i+j+len(game_percentage)]
        average_o_fixation_duration = o_fixation_duration / N
        average_o_fixation_duration_elmo = o_fixation_duration_elmo / N
        average_ox_fixation_duration = ox_fixation_duration / N
        average_ox_fixation_duration_elmo = ox_fixation_duration_elmo / N
        o_array_fixation_durations.append(average_o_fixation_duration)
        o_array_fixation_durations_elmo.append(average_o_fixation_duration_elmo)
        ox_array_fixation_durations.append(average_ox_fixation_duration)
        ox_array_fixation_durations_elmo.append(average_ox_fixation_duration_elmo)
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(game_percentage[0:19], o_array_fixation_durations[0:19], marker = 'o', linestyle = '--', color='b', label='Computer O')
    plt.plot(game_percentage[0:19], o_array_fixation_durations_elmo[0:19], marker = 'o', linestyle = '--', color='r', label='Elmo O')
    plt.plot(game_percentage[0:19], ox_array_fixation_durations[0:19], marker = 'o', linestyle = '--', color='g', label='Computer OX')
    plt.plot(game_percentage[0:19], ox_array_fixation_durations_elmo[0:19], marker = 'o', linestyle = '--', color='y', label='Elmo OX')
    plt.xlim(0, 100)
    # Add labels and title
    plt.xlabel('Task Progress [%]')
    plt.ylabel('Fixation Duration [s]')
    plt.title('Fixation Durations Over Time')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig('results/average_fixation_durations_over_time.png', dpi=1200, bbox_inches='tight')


def plot_duration_over_time(data):
    value = data[data['id'] == 'O20']
    game_percentage = value['game_percentage']
    fixation_duration = value['fixation_durations']
    fixation_duration_elmo = value['fixation_durations_elmo']
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(game_percentage, fixation_duration, marker = 'o', linestyle = '--', color='b', label='Computer')
    plt.plot(game_percentage, fixation_duration_elmo, marker = 'o', linestyle = '--', color='r', label='Elmo')

    # Add labels and title
    plt.xlabel('Task Progress [%]')
    plt.ylabel('Fixation Duration [s]')
    plt.title('Fixation Duration Over Time')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig('results/example_fixation_duration_over_time.png', dpi = 1200, bbox_inches = 'tight')
    



if __name__ == '__main__':
    file = 'tracker_results_final.csv'
    df = pd.read_csv(file)

    o_data = df[df['ID'].str.startswith('O')]
    ox_data = df[df['ID'].str.startswith('OX')]
    plt.rcParams['font.family'] = 'sans-serif'  # Set global font to sans-serif
    plt.rcParams['font.sans-serif'] = 'Arial'  # Specify the font name explicitly
    # box_plot_mean_fixation(o_data, ox_data)
    # box_plot_total_fixation(o_data, ox_data)
    # box_plot_fixation_count(o_data, ox_data)
    # box_plot_count_tension(o_data, ox_data)
    # box_plot_time_tension(o_data, ox_data)

    file2 = 'interval_data.csv'
    df = pd.read_csv(file2)
    # plot_count_over_time(df)
    # plot_duration_over_time(df)

    # get all o condition data
    o_data = df[df['id'].isin(condition_o)]
    # get all ox condition data
    ox_data = df[df['id'].isin(condition_ox)]

    plot_average_counts_over_time(o_data, ox_data)
    plot_average_durations_over_time(o_data, ox_data)

    plt.show()

