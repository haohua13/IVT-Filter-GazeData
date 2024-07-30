import pandas as pd
import csv

def get_XAI(idd):
    if 'X' in idd:
        return 1
    else:
        return 0

def get_spss_data(tracker_results):
    tracker_results['XAI_value'] = tracker_results['ID'].apply(get_XAI)
    tracker_results['elmo_fixation_rate'] = tracker_results['elmo_fixation_count']/tracker_results['game_time']
    tracker_results['computer_fixation_rate'] = tracker_results['computer_fixation_count']/tracker_results['game_time']

    tracker_results['elmo_fixation_percentage'] = tracker_results['elmo_total_fixation_duration']/(tracker_results['elmo_total_fixation_duration']+tracker_results['computer_total_fixation_duration'])
    tracker_results['computer_fixation_percentage'] = tracker_results['computer_total_fixation_duration']/(tracker_results['elmo_total_fixation_duration']+tracker_results['computer_total_fixation_duration'])

    tracker_results.to_csv('tracker_results_final_XAI_numeric.csv', index=False)

if __name__ == '__main__':
    # tracker_results = pd.read_csv('tracker_results_final_XAI_numeric.csv')
    group = pd.read_csv('gaze_data/reliance_groups/no_over_reliance_intervals.csv')
    group['XAI_value'] = group['ID'].apply(get_XAI)
    group.to_csv('gaze_data/reliance_groups/no_over_reliance_intervals.csv', index=False)
    




