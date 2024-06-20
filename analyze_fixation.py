import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
condition_o = {'O20', 'O21', 'O22', 'O24', 'O25', 'O26','O27', 'O28','O29','O30','O31','O32','O33','O34'}
condition_ox = {'OX20', 'OX21', 'OX22','OX23', 'OX24', 'OX25', 'OX26','OX27', 'OX28','OX29','OX31','OX32','OX33','OX34'}
statistical_results = 'statistical_results.csv'

def box_plot_mean_fixation(o_data, ox_data):
    plt.figure(figsize=(10, 6))
    # boxplot for computer_average_fixation_duration
    box1 = plt.boxplot([o_data['computer_average_fixation_duration'], ox_data['computer_average_fixation_duration']], labels=['O', 'OX'])
    plt.title('Computer Screen', fontsize = 18)
    plt.ylabel('Mean Fixation Duration [s]', fontsize = 14)
    plt.xlabel('Group', fontsize = 16)
    plt.grid(True)  # Show grid
    # extracting mean and standard deviation for computer_average_fixation_duration
    mean_computer = o_data['computer_average_fixation_duration'].mean()
    mean_computer_ox = ox_data['computer_average_fixation_duration'].mean()
    std_computer = o_data['computer_average_fixation_duration'].std()
    std_computer_ox = ox_data['computer_average_fixation_duration'].std()
    # Standard Error of the Mean (SEM)
    sem_computer = std_computer / np.sqrt(len(o_data['computer_average_fixation_duration']))
    sem_computer_ox = std_computer_ox / np.sqrt(len(ox_data['computer_average_fixation_duration']))
    # Median of boxplot
    median_computer = box1['medians'][0].get_ydata()[0]
    median_computer_ox = box1['medians'][1].get_ydata()[0]


    print('Computer Mean Fixation Duration:')
    print('Mean:')
    print('O:', mean_computer)
    print('OX:', mean_computer_ox)
    print('Standard Deviation:')
    print('O:', std_computer)
    print('OX:', std_computer_ox)
    print('Standard Error of the Mean:')
    print('O:', sem_computer)
    print('OX:', sem_computer_ox)

    plt.savefig('results/computer_average_fixation_duration.png', dpi = 1200, bbox_inches = 'tight')

    # boxplot for elmo_average_fixation_duration
    plt.figure(figsize=(10, 6))
    box2 = plt.boxplot([o_data['elmo_average_fixation_duration'], ox_data['elmo_average_fixation_duration']], labels=['O', 'OX'])
    plt.title('Elmo Robot', fontsize = 18)
    plt.ylabel('Mean Fixation Duration [s]', fontsize = 14)
    plt.xlabel('Group', fontsize = 16)
    plt.grid()

    # extracting mean and standard deviation for elmo_average_fixation_duration
    mean_elmo = o_data['elmo_average_fixation_duration'].mean()
    mean_elmo_ox = ox_data['elmo_average_fixation_duration'].mean()
    std_elmo = o_data['elmo_average_fixation_duration'].std()
    std_elmo_ox = ox_data['elmo_average_fixation_duration'].std()
    # Standard Error of the Mean (SEM)
    sem_elmo = std_elmo / np.sqrt(len(o_data['elmo_average_fixation_duration']))
    sem_elmo_ox = std_elmo_ox / np.sqrt(len(ox_data['elmo_average_fixation_duration']))
    # Median of boxplot
    median_elmo = box2['medians'][0].get_ydata()[0]
    median_elmo_ox = box2['medians'][1].get_ydata()[0]

    print('\nElmo Mean Fixation Duration:')
    print('Mean:')
    print('O:', mean_elmo)
    print('OX:', mean_elmo_ox)
    print('Standard Deviation:')
    print('O:', std_elmo)
    print('OX:', std_elmo_ox)
    print('Standard Error of the Mean:')
    print('O:', sem_elmo)
    print('OX:', sem_elmo_ox)

    plt.tight_layout()
    plt.savefig('results/elmo_average_fixation_duration.png', dpi = 1200, bbox_inches = 'tight')

    # write to csv file, columns are mean, std, sem
    with open(statistical_results, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow(['metric', 'mean', 'std', 'sem', 'median', 'AOI', 'condition'])
        writer.writerow(['average_fixation_duration', mean_computer, std_computer, sem_computer, median_computer, 'Computer', 'O'])
        writer.writerow(['average_fixation_duration', mean_computer_ox, std_computer_ox, sem_computer_ox, median_computer_ox,  'Computer', 'OX'])
        writer.writerow(['average_fixation_duration', mean_elmo, std_elmo, sem_elmo, median_elmo, 'Elmo', 'O'])
        writer.writerow(['average_fixation_duration', mean_elmo_ox, std_elmo_ox, sem_elmo_ox, median_elmo_ox, 'Elmo', 'OX'])



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
    mean_computer = o_data['computer_total_fixation_duration'].mean()
    mean_computer_ox = ox_data['computer_total_fixation_duration'].mean()
    std_computer = o_data['computer_total_fixation_duration'].std()
    std_computer_ox = ox_data['computer_total_fixation_duration'].std()
    # Standard Error of the Mean (SEM)
    sem_computer = std_computer / np.sqrt(len(o_data['computer_total_fixation_duration']))
    sem_computer_ox = std_computer_ox / np.sqrt(len(ox_data['computer_total_fixation_duration']))

    median_computer = box1['medians'][0].get_ydata()[0]
    median_computer_ox = box1['medians'][1].get_ydata()[0]
    
    
    # boxplot for elmo_total_fixation_duration
    plt.figure(figsize=(10, 6))
    box2 = plt.boxplot([o_normalized_total_fixation_elmo, ox_normalized_total_fixation_elmo], labels=['O', 'OX'])
    plt.title('Elmo Robot', fontsize = 18)
    plt.ylabel('Fixation Duration [%]', fontsize = 14)
    plt.xlabel('Group', fontsize = 16)
    plt.grid()

    # extracting mean and standard deviation for elmo_total_fixation_duration
    mean_elmo = o_data['elmo_total_fixation_duration'].mean()
    mean_elmo_ox = ox_data['elmo_total_fixation_duration'].mean()
    std_elmo = o_data['elmo_total_fixation_duration'].std()
    std_elmo_ox = ox_data['elmo_total_fixation_duration'].std()

    # Standard Error of the Mean (SEM)
    sem_elmo = std_elmo / np.sqrt(len(o_data['elmo_total_fixation_duration']))
    sem_elmo_ox = std_elmo_ox / np.sqrt(len(ox_data['elmo_total_fixation_duration']))
    median_elmo = box2['medians'][0].get_ydata()[0]
    median_elmo_ox = box2['medians'][1].get_ydata()[0]

    plt.tight_layout()
    plt.savefig('results/elmo_total_fixation_duration.png', dpi = 1200, bbox_inches = 'tight')

    # write to csv file
    with open(statistical_results, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow(['total_fixation_duration', mean_computer, std_computer, sem_computer, median_computer, 'Computer', 'O'])
        writer.writerow(['total_fixation_duration', mean_computer_ox, std_computer_ox, sem_computer_ox, median_computer_ox,  'Computer', 'OX'])
        writer.writerow(['total_fixation_duration', mean_elmo, std_elmo, sem_elmo, median_elmo, 'Elmo', 'O'])
        writer.writerow(['total_fixation_duration', mean_elmo_ox, std_elmo_ox, sem_elmo_ox, median_elmo_ox, 'Elmo', 'OX'])

    
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
    mean_computer = o_normalized_total_count.mean()
    mean_computer_ox = ox_normalized_total_count.mean()
    std_computer = o_normalized_total_count.std()
    std_computer_ox = ox_normalized_total_count.std()
    sem_computer = std_computer / np.sqrt(len(o_data['computer_fixation_count']))
    sem_computer_ox = std_computer_ox / np.sqrt(len(ox_data['computer_fixation_count']))
    median_computer = box1['medians'][0].get_ydata()[0]
    median_computer_ox = box1['medians'][1].get_ydata()[0]

    print('Computer Total Fixation Count:')
    print('Mean:')
    print('O:', mean_computer)
    print('OX:', mean_computer_ox)
    print('Standard Deviation:')    
    print('O:', std_computer)
    print('OX:', std_computer_ox)
    print('Standard Error of the Mean:')
    print('O:', sem_computer)
    print('OX:', sem_computer_ox)

    plt.savefig('results/computer_total_fixation_count.png', dpi = 1200, bbox_inches = 'tight')
    # boxplot for elmo_total_fixation_count
    plt.figure(figsize=(10, 6))
    box2 = plt.boxplot([o_normalized_total_count_elmo, ox_normalized_total_count_elmo], labels=['O', 'OX'])
    plt.title('Elmo Robot', fontsize = 18)
    plt.ylabel('Fixation Count Rate', fontsize = 14)
    plt.xlabel('Group', fontsize = 16)
    plt.grid()

    # extracting mean and standard deviation for elmo_total_fixation_count
    mean_elmo = o_normalized_total_count_elmo.mean()
    mean_elmo_ox = ox_normalized_total_count_elmo.mean()
    std_elmo = o_normalized_total_count_elmo.std()
    std_elmo_ox = ox_normalized_total_count_elmo.std()
    sem_elmo = std_elmo / np.sqrt(len(o_data['elmo_fixation_count']))
    sem_elmo_ox = std_elmo_ox / np.sqrt(len(ox_data['elmo_fixation_count']))
    median_elmo_ox = box2['medians'][1].get_ydata()[0]
    median_elmo = box2['medians'][0].get_ydata()[0]

    print('\nElmo Total Fixation Count:')
    print('Mean:')
    print('O:', mean_elmo)
    print('OX:', mean_elmo_ox)
    print('Standard Deviation:')
    print('O:', std_elmo)
    print('OX:', std_elmo_ox)
    print('Standard Error of the Mean:')
    print('O:', sem_elmo)
    print('OX:', sem_elmo_ox)

    plt.tight_layout()
    plt.savefig('results/elmo_total_fixation_count.png', dpi = 1200, bbox_inches = 'tight')

    # write to csv file
    with open(statistical_results, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow(['total_fixation_count', mean_computer, std_computer, sem_computer, median_computer, 'Computer', 'O'])
        writer.writerow(['total_fixation_count', mean_computer_ox, std_computer_ox, sem_computer_ox, median_computer_ox, 'Computer', 'OX'])
        writer.writerow(['total_fixation_count', mean_elmo, std_elmo, sem_elmo, median_elmo, 'Elmo', 'O'])
        writer.writerow(['total_fixation_count', mean_elmo_ox, std_elmo_ox, sem_elmo_ox, median_elmo_ox, 'Elmo', 'OX'])

def box_plot_count_tension(o_data, ox_data):
    o_count_tension = o_data['elmo_fixation_count']/o_data['computer_fixation_count']
    ox_count_tension = ox_data['elmo_fixation_count']/ox_data['computer_fixation_count']
    plt.figure(figsize=(10, 6))
    # boxplot for count_tension
    box1 = plt.boxplot([o_count_tension, ox_count_tension], labels=['O', 'OX'])
    plt.title('Relative Fixation Count', fontsize = 18)
    plt.ylabel('Relative Fixation Count', fontsize = 14)
    plt.xlabel('Group', fontsize = 16)
    plt.grid(True)
    # extracting mean and standard deviation for count_tension
    mean_computer = o_count_tension.mean()
    mean_computer_ox = ox_count_tension.mean()
    std_computer = o_count_tension.std()
    std_computer_ox = ox_count_tension.std()
    sem_computer = std_computer / np.sqrt(len(o_count_tension))
    sem_computer_ox = std_computer_ox / np.sqrt(len(ox_count_tension))
    median_computer = box1['medians'][0].get_ydata()[0]
    median_computer_ox = box1['medians'][1].get_ydata()[0]
    print('Relative Count:')
    print('Mean:')
    print('O:', mean_computer)
    print('OX:', mean_computer_ox)
    print('Standard Deviation:')
    print('O:', std_computer)
    print('OX:', std_computer_ox)
    print('Standard Error of the Mean:')
    print('O:', sem_computer)
    print('OX:', sem_computer_ox)

    # write to csv file
    with open(statistical_results, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow(['relative_count', mean_computer, std_computer, sem_computer, median_computer,'Relative', 'O'])
        writer.writerow(['relative_count', mean_computer_ox, std_computer_ox, sem_computer_ox, median_computer_ox, 'Relative', 'OX'])

    plt.savefig('results/count_tension.png', dpi = 1200, bbox_inches = 'tight')

def box_plot_time_tension(o_data, ox_data):
    o_time_tension = o_data['elmo_total_fixation_duration']/o_data['computer_total_fixation_duration']
    ox_time_tension = ox_data['elmo_total_fixation_duration']/ox_data['computer_total_fixation_duration']
    plt.figure(figsize=(10, 6))
    # boxplot for time_tension
    box1 = plt.boxplot([o_time_tension, ox_time_tension], labels=['O', 'OX'])
    plt.title('Relative Fixation Duration', fontsize = 18)
    plt.ylabel('Relative Fixation Duration', fontsize = 14)
    plt.xlabel('Group', fontsize = 16)
    plt.grid(True)
    # extracting mean and standard deviation for time_tension
    mean_computer = o_time_tension.mean()
    mean_computer_ox = ox_time_tension.mean()
    std_computer = o_time_tension.std()
    std_computer_ox = ox_time_tension.std()
    sem_computer = std_computer / np.sqrt(len(o_time_tension))
    sem_computer_ox = std_computer_ox / np.sqrt(len(ox_time_tension))
    median_computer = box1['medians'][0].get_ydata()[0]
    median_computer_ox = box1['medians'][1].get_ydata()[0]

    print('Relative Fixation Duration:')
    print('Mean:')
    print('O:', mean_computer)
    print('OX:', mean_computer_ox)
    print('Standard Deviation:')
    print('O:', std_computer)
    print('OX:', std_computer_ox)
    print('Standard Error of the Mean:')
    print('O:', sem_computer)
    print('OX:', sem_computer_ox)
    plt.savefig('results/time_tension.png', dpi = 1200, bbox_inches = 'tight')

    # write to csv file
    with open(statistical_results, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow(['relative_duration', mean_computer, std_computer, sem_computer, median_computer, 'Computer', 'O'])
        writer.writerow(['relative_duration', mean_computer_ox, std_computer_ox, sem_computer_ox, median_computer_ox, 'Computer', 'OX'])

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
    box_plot_mean_fixation(o_data, ox_data)
    box_plot_total_fixation(o_data, ox_data)
    box_plot_fixation_count(o_data, ox_data)
    box_plot_count_tension(o_data, ox_data)
    box_plot_time_tension(o_data, ox_data)

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

    # plt.show()

