import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class FixationFilter():
    def __init__(self, data):
        self.data = data
        self.max_time_between_fixations = 0.075 # Komogortsev et al. 2010
        self.max_angle_between_fixations = 0.5 # Komogortsev et al. 2010
        self.fixation_duration_threshold = 0.060 # to remove fixations that are too short Komogortsev et al. 2010

        self.velocity_threshold = 10 # Olsen et al 2012 I-VT fixation filter

        self.fixation_count = 0
        self.final_fixation_count = 0
        self.saccades_count = 0
        self.gaps_count = 0
        self.fixations = []
        self.saccades = []
        self.gaps = []

        self.left_gaze_point_3dx = data['LeftGazePoint3DX']
        self.left_gaze_point_3dy = data['LeftGazePoint3DY']
        self.left_gaze_point_3dz = data['LeftGazePoint3DZ']
        self.right_gaze_point_3dx = data['RightGazePoint3DX']
        self.right_gaze_point_3dy = data['RightGazePoint3DY']
        self.right_gaze_point_3dz = data['RightGazePoint3DZ']
        self.left_eye_position_3dx = data['LeftEyePosition3DX']
        self.left_eye_position_3dy = data['LeftEyePosition3DY']
        self.left_eye_position_3dz = data['LeftEyePosition3DZ']
        self.right_eye_position_3dx = data['RightEyePosition3DX']
        self.right_eye_position_3dy = data['RightEyePosition3DY']
        self.right_eye_position_3dz = data['RightEyePosition3DZ']
        self.left_normalized_x = data['LeftGazePoint2DX']
        self.left_normalized_y = data['LeftGazePoint2DY']
        self.right_normalized_x = data['RightGazePoint2DX']
        self.right_normalized_y = data['RightGazePoint2DY']

        self.left_pixels_x = data['LeftGazePoint2DX']*1920
        self.left_pixels_y = data['LeftGazePoint2DY']*1080
        self.right_pixels_x = data['RightGazePoint2DX']*1920
        self.right_pixels_y = data['RightGazePoint2DY']*1080


        self.time_stamp = data['Timestamp']
        
        # Use the system which is the computer time stamp
        # self.system_time_stamp = data['SystemTimestamp']
        # self.time = (self.time_stamp - self.time_stamp[0])/1e6

        # convert to seconds
        self.time = (self.time_stamp - self.time_stamp[0])/1e6
        self.left_validity = data['LeftValidity']
        self.right_validity = data['RightValidity']

        self.denoised_gaze_point_x = []
        self.denoised_gaze_point_y = []

        self.denoised_pixel_x = []
        self.denoised_pixel_y = []


    def selection(self, method = 'average'):
        if method == 'average' or method == 'strict_average':
            # Calculate the mean while ignoring NaN values
            self.gaze_point_x = np.nanmean(np.array([self.left_gaze_point_3dx, self.right_gaze_point_3dx]), axis=0)
            self.gaze_point_y = np.nanmean(np.array([self.left_gaze_point_3dy, self.right_gaze_point_3dy]), axis=0)
            self.gaze_point_z = np.nanmean(np.array([self.left_gaze_point_3dz, self.right_gaze_point_3dz]), axis=0)
            self.eye_position_x = np.nanmean(np.array([self.left_eye_position_3dx, self.right_eye_position_3dx]), axis=0)
            self.eye_position_y = np.nanmean(np.array([self.left_eye_position_3dy, self.right_eye_position_3dy]), axis=0)
            self.eye_position_z = np.nanmean(np.array([self.left_eye_position_3dz, self.right_eye_position_3dz]), axis=0)

            self.normalized_x = np.nanmean(np.array([self.left_normalized_x, self.right_normalized_x]), axis=0)
            self.normalized_y = np.nanmean(np.array([self.left_normalized_y, self.right_normalized_y]), axis=0)
            self.pixels_x = np.nanmean(np.array([self.left_pixels_x, self.right_pixels_x]), axis=0)
            self.pixels_y = np.nanmean(np.array([self.left_pixels_y, self.right_pixels_y]), axis=0)

            if method =='strict_average':
                # check validity of both eyes
                valid_data_mask = (self.left_validity == 1) & (self.right_validity == 1)
                # apply mask to filter out invalid data points
                self.gaze_point_x[~valid_data_mask] = np.nan
                self.gaze_point_y[~valid_data_mask] = np.nan
                self.gaze_point_z[~valid_data_mask] = np.nan
                self.eye_position_x[~valid_data_mask] = np.nan
                self.eye_position_y[~valid_data_mask] = np.nan
                self.eye_position_z[~valid_data_mask] = np.nan

                self.normalized_x[~valid_data_mask] = np.nan
                self.normalized_y[~valid_data_mask] = np.nan
                self.pixels_x[~valid_data_mask] = np.nan
                self.pixels_y[~valid_data_mask] = np.nan
        elif method == 'left':
            self.gaze_point_x = self.left_gaze_point_3dx
            self.gaze_point_y = self.left_gaze_point_3dy
            self.gaze_point_z = self.left_gaze_point_3dz
            self.eye_position_x = self.left_eye_position_3dx
            self.eye_position_y = self.left_eye_position_3dy
            self.eye_position_z = self.left_eye_position_3dz
            self.normalized_x = self.left_normalized_x
            self.normalized_y = self.left_normalized_y
            self.pixels_x = self.left_pixels_x
            self.pixels_y = self.left_pixels_y

        elif method == 'right':
            self.gaze_point_x = self.right_gaze_point_3dx
            self.gaze_point_y = self.right_gaze_point_3dy
            self.gaze_point_z = self.right_gaze_point_3dz
            self.eye_position_x = self.right_eye_position_3dx
            self.eye_position_y = self.right_eye_position_3dy
            self.eye_position_z = self.right_eye_position_3dz
            self.normalized_x = self.right_normalized_x
            self.normalized_y = self.right_normalized_y
            self.pixels_x = self.right_pixels_x
            self.pixels_y = self.right_pixels_y

    def noise_filter(self, method ='moving_average', alpha=0.5, window_size=4):
        '''Remove noise from the data by applying a low-pass filter
        Parameters:
        method: str
            The method used to filter the data. Options are 'median', 'exponential_moving_average', 'moving_average'
        alpha: float
            The smoothing factor for the exponential moving average. A value of 0.5 gives equal weight to the previous value and the current value.
        window_size: int
            The size of the window used for the moving average.
            Smaller window size allows samples close to gaps to be involved in the fixation classification, but allows more noise to be included.
            Larger window size, the more smoothed out the data will be. Velocity threshold should be lower.
        '''
        if method == 'median':
            return self.median_filter(window_size)
        elif method =='exponential_moving_average':
            return self.exponential_moving_average(alpha)
        elif method == 'moving_average':
            return self.moving_average(window_size)

    def median_filter(self, window_size):
        half_window = window_size // 2 
        # has to be not odd number
        for i in range(len(self.gaze_point_x)):
            if i < half_window or i > len(self.gaze_point_x) - half_window:
                self.denoised_gaze_point_x.append(self.gaze_point_x[i])
                self.denoised_gaze_point_y.append(self.gaze_point_y[i])
                self.denoised_pixel_x.append(self.pixels_x[i])
                self.denoised_pixel_y.append(self.pixels_y[i])
            else:
                self.denoised_gaze_point_x.append(np.median(self.gaze_point_x[i-half_window:i+half_window]))
                self.denoised_gaze_point_y.append(np.median(self.gaze_point_y[i-half_window:i+half_window]))
                self.denoised_pixel_x.append(np.median(self.pixels_x[i-half_window:i+half_window]))
                self.denoised_pixel_y.append(np.median(self.pixels_y[i-half_window:i+half_window]))

    def exponential_moving_average(self, alpha):
        # sf = 2 / (window_size + 1)
        ema_x = np.nan * np.ones(len(self.gaze_point_x))
        ema_y = np.nan * np.ones(len(self.gaze_point_y))
        ema_pixel_x = np.nan * np.ones(len(self.pixels_x))
        ema_pixel_y = np.nan * np.ones(len(self.pixels_y))
        
        for i in range(1, len(self.gaze_point_x)):
            if not np.isnan(self.gaze_point_x[i]):
            # for x-coordinate
                if np.isnan(ema_x[i-1]):
                    ema_x[i] = self.gaze_point_x[i]
                    ema_pixel_x[i] = self.pixels_x[i]
                else:
                    ema_x[i] = (alpha * self.gaze_point_x[i]) + ((1 - alpha) * ema_x[i-1])      
                    ema_pixel_x[i] = (alpha * self.pixels_x[i]) + ((1 - alpha) * ema_pixel_x[i-1]) 
            # for y-coordiante
            if not np.isnan(self.gaze_point_y[i]):
                if np.isnan(ema_y[i-1]):
                    ema_y[i] = self.gaze_point_y[i]
                    ema_pixel_y[i] = self.pixels_y[i]
                else:
                    ema_y[i] = (alpha * self.gaze_point_y[i]) + ((1 - alpha) * ema_y[i-1])
                    ema_pixel_y[i] = (alpha * self.pixels_y[i]) + ((1 - alpha) * ema_pixel_y[i-1])

        self.denoised_gaze_point_x = ema_x
        self.denoised_gaze_point_y = ema_y
        self.denoised_pixel_x = ema_pixel_x
        self.denoised_pixel_y = ema_pixel_y

    def moving_average(self, window_size):
        half_window = window_size // 2
        for i in range(len(self.gaze_point_x)):
            if i < half_window:
                self.denoised_gaze_point_x.append(self.gaze_point_x[i])
                self.denoised_gaze_point_y.append(self.gaze_point_y[i])
                self.denoised_pixel_x.append(self.pixels_x[i])
                self.denoised_pixel_y.append(self.pixels_y[i])
            if i > len(self.gaze_point_x) - half_window:
                self.denoised_gaze_point_x.append(self.gaze_point_x[i])
                self.denoised_gaze_point_y.append(self.gaze_point_y[i])
                self.denoised_pixel_x.append(self.pixels_x[i])
                self.denoised_pixel_y.append(self.pixels_y[i])
            else:
                self.denoised_gaze_point_x.append(np.mean(self.gaze_point_x[i-half_window:i+half_window]))
                self.denoised_gaze_point_y.append(np.mean(self.gaze_point_y[i-half_window:i+half_window]))
                self.denoised_pixel_x.append(np.mean(self.pixels_x[i-half_window:i+half_window]))
                self.denoised_pixel_y.append(np.mean(self.pixels_y[i-half_window:i+half_window]))

    def fill_in_gaps(self, max_gap_length=0.075):
        '''Fill in gaps in the data by interpolating between the two closest points. This is not really necessary'''
        self.interpolated_pixel_x = []
        self.interpolated_pixel_y = []
        gap_start_index = None
        gap_length = 0
        # iterate through each sample
        for i in range(len(self.pixels_x)):
            if self.is_valid_sample(i):
                if gap_start_index is not None:
                    # interpolate the gap
                    self.interpolate_gap(gap_start_index, i, gap_length)
                    gap_start_index = None
                    gap_length = 0
                self.interpolated_pixel_x.append(self.pixels_x[i])
                self.interpolated_pixel_y.append(self.pixels_y[i])
            else:
                if gap_start_index is None:
                    gap_start_index = i
                gap_length += 1
                if gap_length > max_gap_length:
                    # consider it as a 'valid' gap
                    gap_start_index = None
                    gap_length = 0
        # check if there is a gap at the end of the data
        if gap_start_index is not None:
            self.interpolate_gap(gap_start_index, len(self.pixel), gap_length)

        print('Interpolated pixel x', self.interpolated_pixel_x)
        print('Interpolated pixel y', self.interpolated_pixel_y)

    def is_valid_sample(self, index):
        # dheck if the sample contains valid gaze data
        # qccess self.left_validity for validity codes
        if self.left_validity[index] == 0 or self.left_validity[index] == 1:
            return True
        else:
            return False
        
    def interpolate_gap(self, start_index, end_index):
        # interpolate the missing data in the gap
        last_valid_index = start_index - 1
        next_valid_index = end_index
        last_valid_timestamp = self.time[last_valid_index]
        next_valid_timestamp = self.time[next_valid_index]
        total_gap_duration = next_valid_timestamp - last_valid_timestamp
        
        for i in range(start_index, end_index):
            current_timestamp = self.time[i]
            scaling_factor = (current_timestamp - last_valid_timestamp) / total_gap_duration
            interpolated_x = (self.pixel[next_valid_index][0] - self.pixel[last_valid_index][0]) * scaling_factor + self.pixel[last_valid_index][0]
            interpolated_y = (self.pixel[next_valid_index][1] - self.pixel[last_valid_index][1]) * scaling_factor + self.pixel[last_valid_index][1]
            self.interpolated_pixel_x.append(interpolated_x)
            self.interpolated_pixel_y.append(interpolated_y)

    def velocity_calculator(self):
        '''Calculates the velocity as the rate of change of the angle between the gaze vector and the eye position vector. In degrees per second.'''
        # calculate velocity
        eye_positions = np.column_stack((self.eye_position_x, self.eye_position_y, self.eye_position_z))
        gaze_positions = np.column_stack((self.gaze_point_x, self.gaze_point_y, self.gaze_point_z))
        # calculate the magnitude of the eye position
        eye_position_magnitude = np.linalg.norm(eye_positions, axis=1)
        # calculate the gaze vector
        gaze_vectors = gaze_positions - eye_positions
        # calculate the magnitude of the gaze vector
        gaze_vector_magnitude = np.linalg.norm(gaze_vectors, axis=1)
        # calculate dot product of gaze vector and eye position
        dot_product = np.sum(gaze_vectors * eye_positions, axis=1)
        # calculate cosine of the angle between the gaze vector and eye position
        cosine_theta = dot_product / (eye_position_magnitude * gaze_vector_magnitude)
        self.theta = np.degrees(np.arccos(cosine_theta))
        self.velocity = np.gradient(self.theta, self.time)

    def average_velocity_calculator(self, window_length=0.020):
        self.average_velocity = []
        self.average_theta = []
        # number of samples in window length
        window_size = int(window_length / np.mean(np.diff(self.time)))
        # calculate the average velocity
        for i in range(len(self.gaze_point_x)):
            # Determine start and end indices of the window
            start_index = max(0, i - window_size // 2)
            end_index = min(len(self.gaze_point_x) - 1, i + window_size // 2)
            if end_index - start_index + 1 >= 2: # minimum of 2 samples
                # extract relevant gaze point and eye position data
                gaze_positions_window = np.column_stack((self.gaze_point_x[start_index:end_index+1],
                                                          self.gaze_point_y[start_index:end_index+1],
                                                          self.gaze_point_z[start_index:end_index+1]))
                eye_position = np.array([self.eye_position_x[i], self.eye_position_y[i], self.eye_position_z[i]])

                # calculate visual angle between first and last samples
                angle = self.calculate_visual_angle(gaze_positions_window[0], gaze_positions_window[-1], eye_position)

                # calculate time difference between first and last samples
                time_difference = self.time[end_index] - self.time[start_index]

                # calculate velocity
                velocity = angle / time_difference
                self.average_theta.append(angle)
                self.average_velocity.append(velocity)
            else:
                self.average_velocity.append(np.nan)
                self.average_theta.append(np.nan)
        
        # print('average velocity', self.average_velocity)
        # print('average theta', self.average_theta)

    def calculate_visual_angle(gaze_point1, gaze_point2, eye_position):
        '''Calculate the visual angle between two gaze points and the eye position'''
        # calculate the gaze vectors
        gaze_vector1 = gaze_point1 - eye_position
        gaze_vector2 = gaze_point2 - eye_position
        # calculate the magnitude of the gaze vectors
        gaze_vector1_magnitude = np.linalg.norm(gaze_vector1)
        gaze_vector2_magnitude = np.linalg.norm(gaze_vector2)
        # calculate the dot product of the gaze vectors
        dot_product = np.sum(gaze_vector1 * gaze_vector2)
        # calculate the cosine of the angle between the gaze vectors
        cosine_theta = dot_product / (gaze_vector1_magnitude * gaze_vector2_magnitude)
        # calculate the angle between the gaze vectors
        theta = np.arccos(cosine_theta)

        return np.degrees(theta)

    def velocity_classifier(self):
        '''Classify fixations based on velocity threshold'''
        self.classifications = np.full(len(self.velocity), np.nan)
        start_fixation = None
        start_saccade = None
        start_gap = None
        for i in range(len(self.velocity)):
            if not(np.isnan(self.velocity[i])):
                if abs(self.velocity[i]) < self.velocity_threshold:
                    self.classifications[i] = 1 # fixation = 1
                    if start_fixation is None:
                        start_fixation = i # start of a fixation
                    # if start_saccade or start_gap is not None, record them, as they are ended by a fixation
                    if start_saccade is not None:
                        saccade_duration = self.time[i] - self.time[start_saccade]
                        self.record_saccade(start_saccade, i, saccade_duration)
                        start_saccade = None
                    if start_gap is not None:
                        gap_duration = self.time[i] - self.time[start_gap]
                        self.record_gap(start_gap, i, gap_duration)
                        start_gap = None
                    
                else:
                    self.classifications[i] = 0 # saccade = 0
                    if start_saccade is None:
                        start_saccade = i
                    # if start_fixation or start_gap is not None, record them, as they are ended by a saccade
                    if start_fixation is not None: # end of a fixation
                        fixation_duration = self.time[i] - self.time[start_fixation]
                        self.record_fixation(start_fixation, i, fixation_duration)
                        start_fixation = None
                    if start_gap is not None:
                        gap_duration = self.time[i] - self.time[start_gap]
                        self.record_gap(start_gap, i, gap_duration)
                        start_gap = None
            # if classification is nan, samples are within a gap or beginning or end of the data
            else:
                if start_gap is None:
                    start_gap = i
                if start_fixation is not None:
                    fixation_duration = self.time[i] - self.time[start_fixation]
                    self.record_fixation(start_fixation, i, fixation_duration)
                    start_fixation = None
                if start_saccade is not None:
                    saccade_duration = self.time[i] - self.time[start_saccade]
                    self.record_saccade(start_saccade, i, saccade_duration)
                    start_saccade = None

        # check if any ongoing fixations, saccades or gaps extend to the end of the data
        if start_fixation is not None:
            fixation_duration = self.time[i] - self.time[start_fixation]
            self.record_fixation(start_fixation, len(self.velocity)-1, fixation_duration)
        if start_saccade is not None:
            saccade_duration = self.time[i] - self.time[start_saccade]
            self.record_saccade(start_saccade, len(self.velocity)-1, saccade_duration)
        if start_gap is not None:
            gap_duration = self.time[i] - self.time[start_gap]
            self.record_gap(start_gap, len(self.velocity)-1, gap_duration)
        # obtain data of fixations
        self.process_fixations()

    def record_fixation(self, start, end, duration):
        '''Record the start and end of a fixation'''
        fixation = {'start': start, 'end': end, 'duration': duration}
        self.fixations.append(fixation)
        self.fixation_count += 1 # total number of fixations
    def record_saccade(self, start, end, duration):
        '''Record the start and end of a saccade'''
        saccade = {'start': start, 'end': end, 'duration': duration}
        self.saccades.append(saccade)

    def record_gap(self, start, end, duration):
        '''Record the start and end of a gap'''
        gap = {'start': start, 'end': end, 'duration': duration}
        self.gaps.append(gap)
    
    def process_fixations(self):
        # obtain the average position of the gaze point during the fixation
        for fixation in self.fixations:
            start = fixation['start']
            end = fixation['end']
            # calculate the mean of the gaze point during the fixation [in pixels]
            fixation['average_position_x'] = np.mean(self.pixels_x[start:end+1])
            fixation['average_position_y'] = np.mean(self.pixels_y[start:end+1])
            fixation['average_time'] = np.mean(self.time[start:end+1])
            # we can save more information about the fixation samples´
        print('Initial detected fixations', self.fixations)
        self.merge_fixations()
        self.discard_fixations()

    def merge_fixations(self):
        '''Merge fixations that are close to each other, minimum time and angle between fixations'''
        self.merged_fixations = []
        current_fixation_start = None # beginning of the current fixation
        previous_fixation_end = None # end of the previous fixation
        for fixation in self.fixations:
            start = fixation['start']
            end = fixation['end']
            if current_fixation_start is None: # first fixation
                current_fixation_start = start
                previous_fixation_end = end
            else:
                # time difference between fixations
                time_between_fixations = self.time[start] - self.time[previous_fixation_end]
                # angle difference between fixations (theta)
                angle_between_fixations = self.theta[start] - self.theta[previous_fixation_end]
                print(time_between_fixations, angle_between_fixations)
                if time_between_fixations < self.max_time_between_fixations and abs(angle_between_fixations) < self.max_angle_between_fixations:
                    # merge fixations
                    previous_fixation_end = end
                else:
                    self.merged_fixations.append({'start': current_fixation_start, 
                                                  'end': previous_fixation_end, 
                                                  'duration': self.time[previous_fixation_end] - self.time[current_fixation_start]})
                    # start of the new merged fixation
                    current_fixation_start = start
                    previous_fixation_end = end
        # add the last fixation
        if current_fixation_start is not None:
            self.merged_fixations.append({'start': current_fixation_start,
                                            'end': previous_fixation_end,
                                            'duration': self.time[previous_fixation_end] - self.time[current_fixation_start]})
        print('Merged fixations', self.merged_fixations)

    def discard_fixations(self):
        '''Discard fixations that are too short, minimum duration of a fixation to keep'''
        self.final_fixations = []
        for fixation in self.merged_fixations:
            start = fixation['start']
            end = fixation['end']
            if fixation['duration'] >= self.fixation_duration_threshold:
                self.final_fixations.append(fixation)
                self.final_fixation_count += 1
            else:
                self.reclassify_fixation(start, end)
        # obtain average position of the gaze point during the final fixations
        for fixation in self.final_fixations:
            start = fixation['start']
            end = fixation['end']
            # calculate the mean of the gaze point during the fixation [in pixels]
            fixation['average_position_x'] = np.mean(self.pixels_x[start:end+1])
            fixation['average_position_y'] = np.mean(self.pixels_y[start:end+1])
            fixation['average_time'] = np.mean(self.time[start:end+1])
            print('Average values from fixations', fixation['average_position_x'], fixation['average_position_y'], fixation['average_time'])
            
        print('Final fixations after discarding short', self.final_fixations)
        # print(self.final_fixation_count)
        
    def reclassify_fixation(self, start, end):
        '''Reclassify the samples within a fixation that are too short as unknown eye movement'''
        for i in range(start, end+1):
            self.classifications[i] = -1 # unknown eye movement = -1
    
    def plot_gaze_point(self):
        plt.figure()
        
        plt.plot(self.time, self.gaze_point_x, label='Gaze position along the x-axis', color='b')
        plt.plot(self.time, self.gaze_point_y, label='Gaze position along the y-axis', color = 'r')
        plt.legend(fontsize = 6)
        plt.xlabel('Time [s]')
        plt.ylabel('Gaze position in [mm]')
        plt.grid()
    
    def plot_normalized_x_y_gaze_point(self):
        plt.figure()
        plt.scatter(self.left_normalized_x, self.left_normalized_y, marker = 'x', color = 'b')
        plt.xlabel('Normalized gaze position along the x-axis')
        plt.ylabel('Normalized gaze position along the y-axis')
        plt.title('Normalized gaze position (x, y)')
        plt.grid()
    
    def plot_x_y_gaze_point(self):
        plt.figure()
        plt.scatter(self.gaze_point_x, self.gaze_point_y, marker = 'x', color = 'b', linewidths = 1)
        plt.scatter(self.denoised_gaze_point_x, self.denoised_gaze_point_y, marker = '.', color = 'r', linewidths= 1)
        plt.legend(['Raw Gaze Data', 'Filtered'], fontsize = 6)
        plt.xlabel('Gaze position along the x-axis')
        plt.ylabel('Gaze position along the y-axis')
        plt.title('Gaze position in [mm]')
        plt.grid()
    
    def plot_denoised_gaze_point(self):
        plt.figure()
        plt.plot(self.time, self.gaze_point_x, label='Gaze position along the x-axis in mm', color='b')
        plt.plot(self.time, self.gaze_point_y, label='Gaze position along the y-axis in mm', color = 'r')
        plt.scatter(self.time, self.denoised_gaze_point_x, label='Denoised gaze position along the x-axis in mm', marker = 'x', color='c')
        plt.scatter(self.time, self.denoised_gaze_point_y, label='Denoised gaze position along the y-axis in mm', marker = '+', color = 'tab:pink')
        plt.legend(fontsize = 6)
        plt.xlabel('Time [s]')
        plt.ylabel('Gaze position')
        plt.title('Gaze position in [mm] in function of time')
        plt.grid()
    
    def plot_denoised_pixel_point(self):
        plt.figure()
        plt.plot(self.time, self.pixels_x, label='Gaze position along the x-axis in pixels', color='b')
        plt.plot(self.time, self.pixels_y, label='Gaze position along the y-axis in pixels', color = 'r')
        plt.scatter(self.time, self.denoised_pixel_x, label='Denoised gaze position along the x-axis in pixels', marker = 'x', color='c')
        plt.scatter(self.time, self.denoised_pixel_y, label='Denoised gaze position along the y-axis in pixels', marker = '+', color = 'tab:pink')
        plt.legend(fontsize = 6)
        plt.xlabel('Time [s]')
        plt.ylabel('Gaze position')
        plt.title('Gaze position in pixels')
        plt.grid()
    
    def plot_interpolated_pixel_point(self):
        plt.figure()
        difference_pixels_x = np.array(self.pixels_x) - np.array(self.interpolated_pixel_x)
        difference_pixels_y = np.array(self.pixels_y) - np.array(self.interpolated_pixel_y)
        plt.scatter(self.time, difference_pixels_x, label='Interpolated Gaze position along the x-axis in pixels', marker = 'x', color='b')
        plt.scatter(self.time, difference_pixels_y, label='Interpolated Gaze position along the y-axis in pixels', marker = '+', color = 'r')

        plt.legend(fontsize = 6)
        plt.xlabel('Time [s]')
        plt.ylabel('Gaze position')
        plt.grid()
    
    
    def plot_gaze_and_velocity(self):
        plt.figure()
        plt.plot(self.time, self.pixels_x, label='Gaze position along the x-axis', color='b')
        plt.plot(self.time, -np.ones(len(self.time)) * self.velocity_threshold, color = 'k', linestyle = '--')
        plt.plot(self.time, self.pixels_y, label='Gaze position along the y-axis', color = 'r')
        # plt.plot(self.time, self.normalized_y, label='Gaze position along the y-axis', color = 'r')
        plt.plot(self.time, self.velocity, label='Velocity', color = 'g')
        plt.plot(self.time, np.ones(len(self.time)) * self.velocity_threshold, label='Velocity Threshold', color = 'k', linestyle = '--')

        # draw the final fixations
        label_shown = False
        for fixation in self.final_fixations:
            # show fixation zones and average positions
            plt.axvspan(self.time[fixation['start']], self.time[fixation['end']], color = 'tab:gray', alpha = 0.5)
            # how to put scatter on top of plot
            plt.scatter(fixation['average_time'], fixation['average_position_x'], color = 'lime', marker = 'x')
            plt.scatter(fixation['average_time'], fixation['average_position_y'], color = 'indigo', marker = 'x')
            if not label_shown:
                # show legend outside of plots
                plt.legend(['Position x-axis', 'Position y-axis', 'Denoised Position y-axis', 'Velocity', 'Velocity Threshold', 'Fixations', 'Average x', 'Average y'], fontsize = 5, loc = 'upper right')
                label_shown = True
        plt.scatter(self.time, self.denoised_pixel_x, label='Denoised gaze position along the x-axis', marker = 'x', color='c')
        plt.scatter(self.time, self.denoised_pixel_y, label='Denoised gaze position along the y-axis', marker = '+', color = 'tab:pink') 
        plt.xlabel('Time [s]')
        # print(self.time)
        plt.ylabel('Gaze position [pixels] and velocity [deg/s]')
        plt.title('IV-T Filter for fixations')
        plt.grid()
        # save high quality
        plt.savefig('gaze_and_velocity_with_denoise.png', dpi = 300)
    

if __name__ == '__main__':
    file = 'gaze_data_elmo_11-04-15-17'
    input_file = file + '.csv'
    data = pd.read_csv(input_file)
    fixation_filter = FixationFilter(data)
    fixation_filter.selection(method = 'average')
    fixation_filter.noise_filter(method = 'median', alpha = 0.5, window_size = 3)
    fixation_filter.velocity_calculator()
    # fixation_filter.fill_in_gaps()
    fixation_filter.average_velocity_calculator()
    fixation_filter.plot_gaze_point()
    fixation_filter.plot_x_y_gaze_point()
    fixation_filter.plot_normalized_x_y_gaze_point()
    fixation_filter.plot_denoised_gaze_point()
    fixation_filter.plot_denoised_pixel_point()
    # fixation_filter.plot_interpolated_pixel_point()
    fixation_filter.velocity_classifier()
    fixation_filter.plot_gaze_and_velocity()
    plt.show()

