import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from utils import gaussian, draw_display, draw_heatmap

class FixationFilter():
    def __init__(self, data, file, data2, file2):

        self.elmo_width = 20  # 18 + 2 tolerance
        self.elmo_height = 16 # 14 + 2 tolerance
        self.computer_width = 38
        self.computer_height = 25
        self.window_size = (int(1920), int(1080))

        self.max_time_between_fixations = 0.075 # Komogortsev et al. 2010
        self.max_angle_between_fixations = 0.5 # Komogortsev et al. 2010
        self.fixation_duration_threshold = 0.060 # to remove fixations that are too short Komogortsev et al. 2010
        self.velocity_threshold = 30 # Olsen et al 2012 I-VT fixation filter
        self.file = file
        self.file2 = file2

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


        self.left_pixels_x = data['LeftGazePoint2DX']*self.window_size[0]
        self.left_pixels_y = data['LeftGazePoint2DY']*self.window_size[1]
        self.right_pixels_x = data['RightGazePoint2DX']*self.window_size[0]
        self.right_pixels_y = data['RightGazePoint2DY']*self.window_size[1]

        
        self.time_stamp = data['SystemTimestamp']
        self.left_validity = data['LeftValidity']
        self.right_validity = data['RightValidity']

        self.left_gaze_point_3dx_elmo = data2['LeftGazePoint3DX']
        self.left_gaze_point_3dy_elmo = data2['LeftGazePoint3DY']
        self.left_gaze_point_3dz_elmo = data2['LeftGazePoint3DZ']
        self.right_gaze_point_3dx_elmo = data2['RightGazePoint3DX']
        self.right_gaze_point_3dy_elmo = data2['RightGazePoint3DY']
        self.right_gaze_point_3dz_elmo = data2['RightGazePoint3DZ']
        self.left_eye_position_3dx_elmo = data2['LeftEyePosition3DX']
        self.left_eye_position_3dy_elmo = data2['LeftEyePosition3DY']
        self.left_eye_position_3dz_elmo = data2['LeftEyePosition3DZ']
        self.right_eye_position_3dx_elmo = data2['RightEyePosition3DX']
        self.right_eye_position_3dy_elmo = data2['RightEyePosition3DY']
        self.right_eye_position_3dz_elmo = data2['RightEyePosition3DZ']
        self.left_normalized_x_elmo = data2['LeftGazePoint2DX']
        self.left_normalized_y_elmo = data2['LeftGazePoint2DY']
        self.right_normalized_x_elmo = data2['RightGazePoint2DX']
        self.right_normalized_y_elmo = data2['RightGazePoint2DY']

        self.left_pixels_x_elmo =  data2['LeftGazePoint2DX']*self.window_size[0]
        # self.left_pixels_x_elmo = self.window_size[0] - self.left_pixels_x_elmo // 2

        self.left_pixels_y_elmo = data2['LeftGazePoint2DY']*self.window_size[1]
        # self.left_pixels_y_elmo = self.window_size[1] - self.left_pixels_y_elmo // 2

        self.right_pixels_x_elmo = data2['RightGazePoint2DX']*self.window_size[0]
        # self.right_pixels_x_elmo = self.window_size[0] - self.right_pixels_x_elmo // 2

        self.right_pixels_y_elmo = data2['RightGazePoint2DY']*self.window_size[1]
        # self.right_pixels_y_elmo = self.window_size[1] - self.right_pixels_x_elmo // 2



        self.time_stamp_elmo = data2['SystemTimestamp']
        
        self.time = (self.time_stamp - self.time_stamp[0])/1e6
        self.time_elmo = (self.time_stamp_elmo - self.time_stamp_elmo[0])/1e6
        # get last time
        max_time = max(self.time)
        max_time_elmo = max(self.time_elmo)
        difference = abs(max_time - max_time_elmo)

        if max_time_elmo > max_time:
            self.time = self.time + difference
        else:
            self.time_elmo = self.time_elmo + difference
            
        print('Max time:', max_time, 'Max time elmo:', max_time_elmo)
        
        self.left_validity_elmo = data2['LeftValidity']
        self.right_validity_elmo = data2['RightValidity']
             
    def process_data(self, method = 'average', filter_method = 'moving_average', alpha = 0.5, window_size = 5):
        '''Process the data by selecting the eye to use, filtering the data and calculating the velocity'''
        self.selection(method)
        self.selection_elmo(method)
        self.denoised_gaze_point_x, self.denoised_gaze_point_y, self.denoised_pixel_x, self.denoised_pixel_y = self.noise_filter(self.gaze_point_x, self.gaze_point_y, self.pixels_x, self.pixels_y, filter_method, alpha, window_size)
        print(len(self.denoised_gaze_point_x), len(self.denoised_gaze_point_y), len(self.denoised_pixel_x), len(self.denoised_pixel_y))
        self.denoised_gaze_point_x_elmo, self.denoised_gaze_point_y_elmo, self.denoised_pixel_x_elmo, self.denoised_pixel_y_elmo = self.noise_filter(self.gaze_point_x_elmo, self.gaze_point_y_elmo, self.pixels_x_elmo, self.pixels_y_elmo, filter_method, alpha, window_size)
        # self.fill_in_gaps()
        self.velocity_calculator()
        self.average_velocity_calculator()
        self.average_velocity_calculator_elmo()
        self.classifications, self.fixations, self.saccades, self.gaps, self.fixation_count, self.saccades_count, self.gaps_count = self.velocity_classifier(self.velocity, self.time)
        self.classifications_elmo, self.fixations_elmo, self.saccades_elmo, self.gaps_elmo, self.fixation_count_elmo, self.saccades_count_elmo, self.gaps_count_elmo = self.velocity_classifier(self.velocity_elmo, self.time_elmo)
        self.process_fixations()  

    def selection_elmo(self, method = 'average'):
        if method == 'average' or method == 'strict_average':
            # calculate the mean while ignoring NaN values
            self.gaze_point_x_elmo = np.nanmean(np.array([self.left_gaze_point_3dx_elmo, self.right_gaze_point_3dx_elmo]), axis=0)
            self.gaze_point_y_elmo = np.nanmean(np.array([self.left_gaze_point_3dy_elmo, self.right_gaze_point_3dy_elmo]), axis=0)
            self.gaze_point_z_elmo = np.nanmean(np.array([self.left_gaze_point_3dz_elmo, self.right_gaze_point_3dz_elmo]), axis=0)
            self.eye_position_x_elmo = np.nanmean(np.array([self.left_eye_position_3dx_elmo, self.right_eye_position_3dx_elmo]), axis=0)
            self.eye_position_y_elmo = np.nanmean(np.array([self.left_eye_position_3dy_elmo, self.right_eye_position_3dy_elmo]), axis=0)
            self.eye_position_z_elmo = np.nanmean(np.array([self.left_eye_position_3dz_elmo, self.right_eye_position_3dz_elmo]), axis=0)
            self.normalized_x_elmo = np.nanmean(np.array([self.left_normalized_x_elmo, self.right_normalized_x_elmo]), axis=0)
            self.normalized_y_elmo = np.nanmean(np.array([self.left_normalized_y_elmo, self.right_normalized_y_elmo]), axis=0)
            self.pixels_x_elmo = np.nanmean(np.array([self.left_pixels_x_elmo, self.right_pixels_x_elmo]), axis=0)
            self.pixels_y_elmo = np.nanmean(np.array([self.left_pixels_y_elmo, self.right_pixels_y_elmo]), axis=0)

            if method =='strict_average':
                # check validity of both eyes
                valid_data_mask_elmo = (self.left_validity_elmo == 1) & (self.right_validity_elmo == 1)
                # apply mask to filter out invalid data points
                self.gaze_point_x_elmo[~valid_data_mask_elmo] = np.nan
                self.gaze_point_y_elmo[~valid_data_mask_elmo] = np.nan
                self.gaze_point_z_elmo[~valid_data_mask_elmo] = np.nan
                self.eye_position_x_elmo[~valid_data_mask_elmo] = np.nan
                self.eye_position_y_elmo[~valid_data_mask_elmo] = np.nan
                self.eye_position_z_elmo[~valid_data_mask_elmo] = np.nan
                self.normalized_x_elmo[~valid_data_mask_elmo] = np.nan
                self.normalized_y_elmo[~valid_data_mask_elmo] = np.nan
                self.pixels_x_elmo[~valid_data_mask_elmo] = np.nan
                self.pixels_y_elmo[~valid_data_mask_elmo] = np.nan

            elif method == 'left':
                self.gaze_point_x_elmo = self.left_gaze_point_3dx_elmo
                self.gaze_point_y_elmo = self.left_gaze_point_3dy_elmo
                self.gaze_point_z_elmo = self.left_gaze_point_3dz_elmo
                self.eye_position_x_elmo = self.left_eye_position_3dx_elmo
                self.eye_position_y_elmo = self.left_eye_position_3dy_elmo
                self.eye_position_z_elmo = self.left_eye_position_3dz_elmo
                self.normalized_x_elmo = self.left_normalized_x_elmo
                self.normalized_y_elmo = self.left_normalized_y_elmo
                self.pixels_x_elmo = self.left_pixels_x_elmo
                self.pixels_y_elmo = self.left_pixels_y_elmo

            elif method == 'right':
                self.gaze_point_x_elmo = self.right_gaze_point_3dx_elmo
                self.gaze_point_y_elmo = self.right_gaze_point_3dy_elmo
                self.gaze_point_z_elmo = self.right_gaze_point_3dz_elmo
                self.eye_position_x_elmo = self.right_eye_position_3dx_elmo
                self.eye_position_y_elmo = self.right_eye_position_3dy_elmo
                self.eye_position_z_elmo = self.right_eye_position_3dz_elmo
                self.normalized_x_elmo = self.right_normalized_x_elmo
                self.normalized_y_elmo = self.right_normalized_y_elmo
                self.pixels_x_elmo = self.right_pixels_x_elmo
                self.pixels_y_elmo = self.right_pixels_y_elmo
            
    def selection(self, method = 'average'):
        if method == 'average' or method == 'strict_average':
            # calculate the mean while ignoring NaN values
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
                valid_data_mask_elmo = (self.left_validity_elmo == 1) & (self.right_validity_elmo == 1)
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

    def noise_filter(self, x, y, pixel_x, pixel_y, method ='moving_average', alpha=0.5, window_size=4):
        '''Remove noise from the data by applying a low-pass filter
        Parameters:
        gaze_point_x: list
            The x-coordinate of the gaze point
        gaze_point_y: list
            The y-coordinate of the gaze point
        pixels_x: list
            The x-coordinate of the gaze point in pixels
        pixels_y: list
            The y-coordinate of the gaze point in pixels
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
            return self.median_filter(x, y, pixel_x, pixel_y, window_size)
        elif method =='exponential_moving_average':
            return self.exponential_moving_average(x, y, pixel_x, pixel_y, alpha)
        elif method == 'moving_average':
            return self.moving_average(x, y, pixel_x, pixel_y, window_size)
        
    def median_filter(self, x, y, pixel_x, pixel_y, window_size):
        half_window = window_size // 2 
        denoised_gaze_point_x = []
        denoised_gaze_point_y = []
        denoised_pixel_x = []
        denoised_pixel_y = []
        # has to be not odd number
        for i in range(len(x)):
            if i < half_window or i > len(x) - half_window:
                denoised_gaze_point_x.append(x[i])
                denoised_gaze_point_y.append(y[i])
                denoised_pixel_x.append(pixel_x[i])
                denoised_pixel_y.append(pixel_y[i])
            else:
                denoised_gaze_point_x.append(np.median(x[i-half_window:i+half_window]))
                denoised_gaze_point_y.append(np.median(y[i-half_window:i+half_window]))
                denoised_pixel_x.append(np.median(pixel_x[i-half_window:i+half_window]))
                denoised_pixel_y.append(np.median(pixel_y[i-half_window:i+half_window]))
        print(len(denoised_gaze_point_x), len(denoised_gaze_point_y), len(denoised_pixel_x), len(denoised_pixel_y))

        return denoised_gaze_point_x, denoised_gaze_point_y, denoised_pixel_x, denoised_pixel_y

    def exponential_moving_average(self, x, y, pixel_x, pixel_y, alpha):
        '''Apply an exponential moving average to the data'''
        # sf = 2 / (window_size + 1)
        ema_x = np.nan * np.ones(len(x))
        ema_y = np.nan * np.ones(len(y))
        ema_pixel_x = np.nan * np.ones(len(pixel_x))
        ema_pixel_y = np.nan * np.ones(len(pixel_y))
        for i in range(1, len(x)):
            if not np.isnan(x[i]):
            # for x-coordinate
                if np.isnan(ema_x[i-1]):
                    ema_x[i] = x[i]
                    ema_pixel_x[i] = pixel_x[i]
                else:
                    ema_x[i] = (alpha * x[i]) + ((1 - alpha) * ema_x[i-1])      
                    ema_pixel_x[i] = (alpha * pixel_x[i]) + ((1 - alpha) * ema_pixel_x[i-1]) 
            # for y-coordiante
            if not np.isnan(y[i]):
                if np.isnan(ema_y[i-1]):
                    ema_y[i] = y[i]
                    ema_pixel_y[i] = pixel_y[i]
                else:
                    ema_y[i] = (alpha * y[i]) + ((1 - alpha) * ema_y[i-1])
                    ema_pixel_y[i] = (alpha * pixel_y[i]) + ((1 - alpha) * ema_pixel_y[i-1])

        return ema_x, ema_y, ema_pixel_x, ema_pixel_y

    def moving_average(self, x, y, pixel_x, pixel_y, window_size):
        half_window = window_size // 2
        denoised_gaze_point_x = []
        denoised_gaze_point_y = []
        denoised_pixel_x = []
        denoised_pixel_y = []
        for i in range(len(x)):
            if i < half_window or i > len(x) - half_window:
                denoised_gaze_point_x.append(x[i])
                denoised_gaze_point_y.append(y[i])
                denoised_pixel_x.append(pixel_x[i])
                denoised_pixel_y.append(pixel_y[i])
            else:
                denoised_gaze_point_x.append(np.mean(x[i-half_window:i+half_window]))
                denoised_gaze_point_y.append(np.mean(y[i-half_window:i+half_window]))
                denoised_pixel_x.append(np.mean(pixel_x[i-half_window:i+half_window]))
                denoised_pixel_y.append(np.mean(pixel_y[i-half_window:i+half_window]))

        return denoised_gaze_point_x, denoised_gaze_point_y, denoised_pixel_x, denoised_pixel_y

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

        # print('Interpolated pixel x', self.interpolated_pixel_x)
        # print('Interpolated pixel y', self.interpolated_pixel_y)

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
        # calculate velocity for data 1
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

        eye_positions_elmo = np.column_stack((self.eye_position_x_elmo, self.eye_position_y_elmo, self.eye_position_z_elmo))
        gaze_positions_elmo = np.column_stack((self.gaze_point_x_elmo, self.gaze_point_y_elmo, self.gaze_point_z_elmo))
        eye_position_magnitude_elmo = np.linalg.norm(eye_positions_elmo, axis=1)
        gaze_vectors_elmo = gaze_positions_elmo - eye_positions_elmo
        gaze_vector_magnitude_elmo = np.linalg.norm(gaze_vectors_elmo, axis=1)
        dot_product_elmo = np.sum(gaze_vectors_elmo * eye_positions_elmo, axis=1)
        cosine_theta_elmo = dot_product_elmo / (eye_position_magnitude_elmo * gaze_vector_magnitude_elmo)
        self.theta_elmo = np.degrees(np.arccos(cosine_theta_elmo))
        self.velocity_elmo = np.gradient(self.theta_elmo, self.time_elmo)

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

    def average_velocity_calculator_elmo (self, window_length=0.020):
        self.average_velocity_elmo = []
        self.average_theta_elmo = []
        # number of samples in window length
        window_size = int(window_length / np.mean(np.diff(self.time_elmo)))
        # calculate the average velocity
        for i in range(len(self.gaze_point_x_elmo)):
            # Determine start and end indices of the window
            start_index = max(0, i - window_size // 2)
            end_index = min(len(self.gaze_point_x_elmo) - 1, i + window_size // 2)
            if end_index - start_index + 1 >= 2:
                # extract relevant gaze point and eye position data
                gaze_positions_window = np.column_stack((self.gaze_point_x_elmo[start_index:end_index+1],
                                                          self.gaze_point_y_elmo[start_index:end_index+1],
                                                          self.gaze_point_z_elmo[start_index:end_index+1]))
                eye_position = np.array([self.eye_position_x_elmo[i], self.eye_position_y_elmo[i], self.eye_position_z_elmo[i]])

                # calculate visual angle between first and last samples
                angle = self.calculate_visual_angle(gaze_positions_window[0], gaze_positions_window[-1], eye_position)

                # calculate time difference between first and last samples
                time_difference = self.time_elmo[end_index] - self.time_elmo[start_index]
                # calculate velocity
                velocity = angle / time_difference
                self.average_theta_elmo.append(angle)
                self.average_velocity_elmo.append(velocity)
            else:
                self.average_velocity_elmo.append(np.nan)
                self.average_theta_elmo.append(np.nan)

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

    def velocity_classifier(self, velocity, time):
        '''Classify fixations based on velocity threshold'''
        classifications = np.full(len(velocity), np.nan)
        fixations = []
        fixation_count = 0
        saccades = []
        saccade_count = 0
        gaps = []
        gap_count = 0
        start_fixation = None
        start_saccade = None
        start_gap = None
        for i in range(len(velocity)):
            if not(np.isnan(velocity[i])):
                if abs(velocity[i]) < self.velocity_threshold:
                    classifications[i] = 1 # fixation = 1
                    if start_fixation is None:
                        start_fixation = i # start of a fixation
                    # if start_saccade or start_gap is not None, record them, as they are ended by a fixation
                    if start_saccade is not None:
                        saccade_duration = time[i] - time[start_saccade]
                        saccade = self.record_saccade(start_saccade, i, saccade_duration)
                        saccades.append(saccade)
                        saccade_count += 1

                        start_saccade = None
                    if start_gap is not None:
                        gap_duration = time[i] - time[start_gap]
                        gap = self.record_gap(start_gap, i, gap_duration)
                        gaps.append(gap)
                        saccade_count += 1
                        start_gap = None
                    
                else:
                    classifications[i] = 0 # saccade = 0
                    if start_saccade is None:
                        start_saccade = i
                    # if start_fixation or start_gap is not None, record them, as they are ended by a saccade
                    if start_fixation is not None: # end of a fixation
                        fixation_duration = time[i] - time[start_fixation]
                        fixation = self.record_fixation(start_fixation, i, fixation_duration)
                        fixations.append(fixation)
                        fixation_count += 1
                        start_fixation = None
                    if start_gap is not None:
                        gap_duration = time[i] - time[start_gap]
                        gap = self.record_gap(start_gap, i, gap_duration)
                        gaps.append(gap)
                        gap_count += 1
                        start_gap = None
            # if classification is nan, samples are within a gap or beginning or end of the data
            else:
                if start_gap is None:
                    start_gap = i
                if start_fixation is not None:
                    fixation_duration = time[i] - time[start_fixation]
                    fixation = self.record_fixation(start_fixation, i, fixation_duration)
                    fixations.append(fixation)
                    fixation_count += 1
                    start_fixation = None
                if start_saccade is not None:
                    saccade_duration = time[i] - time[start_saccade]
                    saccade = self.record_saccade(start_saccade, i, saccade_duration)
                    saccades.append(saccade)
                    saccade_count += 1
                    start_saccade = None

        # check if any ongoing fixations, saccades or gaps extend to the end of the data
        if start_fixation is not None:
            fixation_duration = time[i] - time[start_fixation]
            fixation = self.record_fixation(start_fixation, len(velocity)-1, fixation_duration)
            fixations.append(fixation)
            fixation_count +=1

    
        if start_saccade is not None:
            saccade_duration = time[i] - time[start_saccade]
            saccade = self.record_saccade(start_saccade, len(velocity)-1, saccade_duration)
            saccades.append(saccade)
            saccade_count += 1
        if start_gap is not None:
            gap_duration = time[i] - time[start_gap]
            gap = self.record_gap(start_gap, len(velocity)-1, gap_duration)
            gaps.append(gap)
            gap_count += 1
    
        return classifications, fixations, saccades, gaps, fixation_count, saccade_count, gap_count

    def record_fixation(self, start, end, duration):
        '''Record the start and end of a fixation'''
        fixation = {'start': start, 'end': end, 'duration': duration}
        return fixation

    def record_saccade(self, start, end, duration):
        '''Record the start and end of a saccade'''
        saccade = {'start': start, 'end': end, 'duration': duration}
        return saccade

    def record_gap(self, start, end, duration):
        '''Record the start and end of a gap'''
        gap = {'start': start, 'end': end, 'duration': duration}
        return gap
    
    def process_fixations(self):
        # obtain the average position of the gaze point during the fixation
        for fixation, fixation_elmo in zip(self.fixations, self.fixations_elmo):
            start = fixation['start']
            end = fixation['end']
            # calculate the mean of the gaze point during the fixation [in pixels]
            fixation['average_position_x'] = np.nanmean(self.pixels_x[start:end+1], axis = 0)
            fixation['average_position_y'] = np.nanmean(self.pixels_y[start:end+1], axis = 0)
            fixation['average_time'] = np.nanmean(self.time[start:end+1], axis = 0)
            start = fixation_elmo['start']
            end = fixation_elmo['end']
            fixation_elmo['average_position_x'] = np.nanmean(self.pixels_x_elmo[start:end+1], axis = 0)
            fixation_elmo['average_position_y'] = np.nanmean(self.pixels_y_elmo[start:end+1], axis = 0)
            fixation_elmo['average_time'] = np.nanmean(self.time_elmo[start:end+1], axis = 0)

        self.merged_fixations = self.merge_fixations(self.fixations, self.time, self.theta)
        self.merged_fixations_elmo = self.merge_fixations(self.fixations_elmo, self.time_elmo, self.theta_elmo)
        self.final_fixations, self.final_fixation_count = self.discard_fixations(self.merged_fixations, self.classifications, self.pixels_x, self.pixels_y, self.time)
        self.final_fixations_elmo, self.final_fixation_count_elmo = self.discard_fixations(self.merged_fixations_elmo, self.classifications_elmo, self.pixels_x_elmo, self.pixels_y_elmo, self.time_elmo)

    def merge_fixations(self, fixations, time, theta):
        '''Merge fixations that are close to each other, minimum time and angle between fixations'''
        merged_fixations = []
        current_fixation_start = None
        previous_fixation_end = None
        for fixation in fixations:
            start = fixation['start']
            end = fixation['end']
            if current_fixation_start is None:
                current_fixation_start = start
                previous_fixation_end = end
            else:
                time_between_fixations = time[start] - time[previous_fixation_end]
                angle_between_fixations = theta[start] - theta[previous_fixation_end]
                if time_between_fixations < self.max_time_between_fixations and abs(angle_between_fixations) < self.max_angle_between_fixations:
                    previous_fixation_end = end
                else:
                    merged_fixations.append({'start': current_fixation_start, 
                                             'end': previous_fixation_end, 
                                             'duration': time[previous_fixation_end] - time[current_fixation_start]})
                    current_fixation_start = start
                    previous_fixation_end = end

        if current_fixation_start is not None:
            merged_fixations.append({'start': current_fixation_start,
                                     'end': previous_fixation_end,
                                     'duration': time[previous_fixation_end] - time[current_fixation_start]})
        return merged_fixations
    
    def discard_fixations(self, merged_fixations, classification, pixels_x, pixels_y, time):
        '''Discard fixations that are too short, minimum duration of a fixation to keep'''
        final_fixations = []
        final_fixation_count = 0
        for fixation in merged_fixations:
            start = fixation['start']
            end = fixation['end']
            if fixation['duration'] >= self.fixation_duration_threshold:
                final_fixations.append(fixation)
                final_fixation_count += 1
            else:
                self.reclassify_fixation(classification, start, end)
        # obtain average position of the gaze point during the final fixations
        for fixation in final_fixations:
            start = fixation['start']
            end = fixation['end']
            # calculate the mean of the gaze point during the fixation [in pixels]
            fixation['average_position_x'] = np.nanmean(pixels_x[start:end+1], axis = 0)
            fixation['average_position_y'] = np.nanmean(pixels_y[start:end+1], axis = 0)
            fixation['average_time'] = np.nanmean(time[start:end+1], axis = 0)
        # print('Final fixations after discarding short', final_fixations)
        # print(final_fixation_count)
        return final_fixations, final_fixation_count
        
    def reclassify_fixation(self, classification, start, end):
        '''Reclassify the samples within a fixation that are too short as unknown eye movement'''
        for i in range(start, end+1):
            classification[i] = 0
    
    def calculate_elmo_pixels(self):
        self.elmo_agent_size = (self.window_size[0]*self.elmo_width/self.computer_width, self.window_size[1]*self.elmo_height/self.computer_height)
        self.elmo_agent_position = ((self.window_size[0] - self.elmo_agent_size[0]) // 2, (self.window_size[1] - self.elmo_agent_size[1]) // 2)
        self.normalized_elmo_agent_size = (self.elmo_width/self.computer_width, self.elmo_height/self.computer_height)
        center = (0.5, 0.5)
        self.normalized_elmo_agent_position = (center[0] - self.normalized_elmo_agent_size[0]/2, center[0] - self.normalized_elmo_agent_size[1]/2)

    def plot_normalized_elmo(self):
        fig, ax = plt.subplots()
        plt.scatter(self.left_normalized_x_elmo, self.left_normalized_y_elmo, marker = 'x', color = 'b')
        # plot the rectangle
        rectangle = patches.Rectangle((self.normalized_elmo_agent_position[0], self.normalized_elmo_agent_position[1]), self.normalized_elmo_agent_size[0], self.normalized_elmo_agent_size[1], linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rectangle)
        # plot the ellipsoid
        ellipsoid = patches.Ellipse((0.5, 0.5), self.normalized_elmo_agent_size[0], self.normalized_elmo_agent_size[1], linewidth=1, edgecolor='black', facecolor='none', label='Area of Interest')
        ax.add_patch(ellipsoid)
        plt.xlabel('Normalized gaze position along the x-axis')
        plt.ylabel('Normalized gaze position along the y-axis')
        plt.title('Normalized gaze position (x, y)')
        plt.gca().invert_yaxis()
        plt.grid()

    def plot_pixel_elmo(self):
        fig, ax = plt.subplots()
        plt.scatter(self.pixels_x_elmo, self.pixels_y_elmo, marker = 'x', color = 'b')
        # plot the rectangle
        rectangle = patches.Rectangle((self.elmo_agent_position[0], self.elmo_agent_position[1]), self.elmo_agent_size[0], self.elmo_agent_size[1], linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rectangle)
        # plot the ellipsoid
        ellipsoid = patches.Ellipse((self.window_size[0]*0.5, self.window_size[1]*0.5), self.elmo_agent_size[0], self.elmo_agent_size[1], linewidth=1, edgecolor='black', facecolor='none', label='Area of Interest')
        ax.add_patch(ellipsoid)
        plt.legend(fontsize = 6)
        plt.xlabel('X-axis Pixels')
        plt.ylabel('Y-axis Pixels')
        plt.title('Gaze position in pixels for Elmo Tracker')
        plt.gca().invert_yaxis()
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

    def plot_heatmap_fixations(self, fixations, file, alpha = 1, gaussiannwh = 200, gaussiansd = 33):

        '''Plot a heatmap of the fixations, taken from https://github.com/TobiasRoeddiger/GazePointHeatMap/blob/master/gazeheatplot.py'''
        # gaussian kernel parameters
        gwh = gaussiannwh
        gsdwh = gaussiansd
        gaus = gaussian(gwh, gsdwh)

        # matrix of zeroes for heatmap
        strt = int(gwh / 2)
        heatmapsize = int(self.window_size[1] + 2 * strt), int(self.window_size[0] + 2 * strt)
        heatmap = np.ones(heatmapsize, dtype=float)

        # create heatmap
        for fixation in fixations:
            x = int(fixation['average_position_x']) 
            y = int(fixation['average_position_y'])
            # correct Gaussian size if fixation is outside of display boundaries
            if (not 0 < x < self.window_size[0]) or (not 0 < y < self.window_size[1]):
                hadj = [0, gwh]
                vadj = [0, gwh]
                if x < 0:
                    hadj[0] = abs(x)
                    x = 0
                elif self.computer_width < x:
                    hadj[1] = gwh - int(x - self.window_size[0])
                if y < 0:
                    vadj[0] = abs(y)
                    y = 0
                elif self.computer_height < y:
                    vadj[1] = gwh - int(y - self.window_size[1])
                
                try: 
                    heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * fixation['duration']
                except:
                    # fixation was probably outside of display
                    pass
            else:
                # add Gaussian to the current heatmap
                heatmap[y:y + gwh, x:x + gwh] += gaus * fixation['duration']
        # resize heatmap
        heatmap = heatmap[int(strt):int(self.window_size[1] + strt), int(strt):int(self.window_size[0] + strt)]

        # remove zeros
        lowbound = np.mean(heatmap[heatmap > 0])
        heatmap[heatmap < lowbound] = 0

        # plot heatmap

        plt.figure()
        plt.imshow(heatmap, cmap = 'viridis', alpha = alpha) 
        # title 
        plt.title('Heatmap of Fixations')
        # labels
        plt.xlabel('X-axis Pixels')
        plt.ylabel('Y-axis Pixels')
        plt.xlim(0, self.window_size[0])
        plt.ylim(0, self.window_size[1])
        plt.gca().invert_yaxis()  
        plt.colorbar(label = 'Based on duration and position of fixation')
        plt.grid()
        plt.savefig('images/heatmap_fixations_'+ file + '.png', dpi = 300)


    def plot_fixation_map(self, fixations, file):
        '''Plot the fixation map of the fixations, size of scatter depends on duration of fixation'''
        plt.figure()
        i = 0
        for fixation in fixations:
            duration = fixation['duration']
            avg_x = fixation['average_position_x']
            avg_y = fixation['average_position_y']
            plt.scatter(avg_x, avg_y, color='tab:green', s = 60+5**(duration*2), alpha = 0.5)
            plt.text(avg_x, avg_y, str(i), fontsize=10, ha='center', va='center', color = 'black')
            i=i+1
        plt.xlabel('X-axis Pixels')
        plt.ylabel('Y-axis Pixels')
        plt.title('Fixation 2-D map')
        plt.gca().invert_yaxis()  
        plt.grid()

    def plot_scanpath_fixations(self, fixations, file):

        '''Plot the scan path of the fixations'''
        plt.figure()
        avg_x = [fixation['average_position_x'] for fixation in fixations]
        avg_y = [fixation['average_position_y'] for fixation in fixations]
        
        # plot scan path
        plt.plot(avg_x, avg_y, color='black', label='Scan Path', linestyle = '--', alpha = 0.5)
        # annotate scatter points with their indices
        for i, (x, y) in enumerate(zip(avg_x, avg_y)):
            plt.text(x, y, str(i), fontsize=10, ha='center', va='center', color = 'black')
            duration = fixations[i]['duration']
            plt.scatter(x, y, color='tab:green', alpha = 0.5, edgecolors = 'black', s = 60+5**(duration*2))

        plt.xlabel('X-axis Pixels')
        plt.ylabel('Y-axis Pixels')
        plt.title('Scan Path of the Fixations')
        plt.gca().invert_yaxis()  
        plt.grid()
        plt.legend()
        plt.savefig('images/scanpath_fixations_'+ file + '.png', dpi = 300)
    
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
                # show legend outside of plot
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
        plt.savefig('images/gaze_and_velocity_plot_'+ self.file + '.png', dpi = 300)
        self.plot_heatmap_fixations(self.final_fixations, self.file)
        # self.call_heatmap(self.final_fixations, self.file)
        self.plot_scanpath_fixations(self.final_fixations, self.file)
        self.plot_fixation_map(self.final_fixations, self.file)
    
    def plot_gaze_and_velocity_elmo(self):
        plt.figure()
        plt.plot(self.time_elmo, self.pixels_x_elmo, label='Gaze position along the x-axis', color='b')
        plt.plot(self.time_elmo, -np.ones(len(self.time_elmo)) * self.velocity_threshold, color = 'k', linestyle = '--')
        plt.plot(self.time_elmo, self.pixels_y_elmo, label='Gaze position along the y-axis', color = 'r')
        # plt.plot(self.time, self.normalized_y, label='Gaze position along the y-axis', color = 'r')
        plt.plot(self.time_elmo, self.velocity_elmo, label='Velocity', color = 'g')
        plt.plot(self.time_elmo, np.ones(len(self.time_elmo)) * self.velocity_threshold, label='Velocity Threshold', color = 'k', linestyle = '--')

        # draw the final fixations
        label_shown = False
        for fixation in self.final_fixations_elmo:
            # show fixation zones and average positions
            plt.axvspan(self.time_elmo[fixation['start']], self.time_elmo[fixation['end']], color = 'tab:gray', alpha = 0.5)
            # how to put scatter on top of plot
            plt.scatter(fixation['average_time'], fixation['average_position_x'], color = 'lime', marker = 'x')
            plt.scatter(fixation['average_time'], fixation['average_position_y'], color = 'indigo', marker = 'x')
            if not label_shown:
                # show legend outside of plot
                plt.legend(['Position x-axis', 'Position y-axis', 'Denoised Position y-axis', 'Velocity', 'Velocity Threshold', 'Fixations', 'Average x', 'Average y'], fontsize = 5, loc = 'upper right')
                label_shown = True
        plt.scatter(self.time_elmo, self.denoised_pixel_x_elmo, label='Denoised gaze position along the x-axis', marker = 'x', color='c')
        plt.scatter(self.time_elmo, self.denoised_pixel_y_elmo, label='Denoised gaze position along the y-axis', marker = '+', color = 'tab:pink')
        plt.xlabel('Time [s]')
        # print(self.time)
        plt.ylabel('Gaze position [pixels] and velocity [deg/s]')
        plt.title('IV-T Filter for fixations Elmo')
        plt.grid()
        # save high quality
        plt.savefig('images/gaze_and_velocity_plot_elmo_'+ self.file2 + '.png', dpi = 300)
        self.plot_heatmap_fixations(self.final_fixations_elmo, self.file2)
        # self.call_heatmap(self.final_fixations_elmo, self.file2)
        self.plot_scanpath_fixations(self.final_fixations_elmo, self.file2)
        self.plot_fixation_map(self.final_fixations_elmo, self.file2)


if __name__ == '__main__':
    filename_elmo = 'gaze_data_elmo_11-04-15-03'
    filename_screen = 'gaze_data_11-04-15-03'
    file_elmo = 'gaze_data/' + filename_elmo + '.csv'
    file_screen = 'gaze_data/' + filename_screen + '.csv'
    data_elmo = pd.read_csv(file_elmo)
    data_screen = pd.read_csv(file_screen)
    
    fixation_filter = FixationFilter(data_screen, filename_screen, data_elmo, filename_elmo)
    fixation_filter.process_data()
    fixation_filter.plot_gaze_and_velocity()
    fixation_filter.plot_gaze_and_velocity_elmo()
    fixation_filter.calculate_elmo_pixels()
    fixation_filter.plot_pixel_elmo()
    fixation_filter.plot_normalized_elmo()

    plt.show()

