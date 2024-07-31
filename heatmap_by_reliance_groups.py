import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import pyplot, image
import csv
import pandas as pd

def gaussian(x, sx, y= None, sy = None):
    """Returns an array of numpy arrays (a matrix) containing values between 1 and 0 in a 2D Gaussian distribution"""

    if y == None:
        y = x
    if sy == None:
        sy = sx
    
    xo = x  /2
    yo = y / 2

    M = np.zeros([y, x], dtype = float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = np.exp(
                -1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))
    return M

def draw_display(dispsize, imagefile=None):
    """Returns a matplotlib.pyplot Figure and its axes, with a size of
    dispsize, a black background colour, and optionally with an image drawn
    onto it

    arguments

    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)

    returns
    fig, ax		-	matplotlib.pyplot Figure and its axes: field of zeros
                    with a size of dispsize, and an image drawn onto it
                    if an imagefile was passed
    """

    # construct screen (black background)
    screen = np.zeros((dispsize[1], dispsize[0], 3), dtype='float32')
    # if an image location has been passed, draw the image
    if imagefile != None:
        # check if the path to the image exists
        if not os.path.isfile(imagefile):
            raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
        # load image
        img = image.imread(imagefile)

        # width and height of the image
        w, h = len(img[0]), len(img)
        # x and y position of the image on the display
        x = dispsize[0] / 2 - w / 2
        y = dispsize[1] / 2 - h / 2
        # draw the image on the screen
        screen[y:y + h, x:x + w, :] += img
    # dots per inch
    # dpi = 100
    # determine the figure size in inches
    # figsize = (dispsize[0] / dpi, dispsize[1] / dpi)

    
    # create a figure
    fig = pyplot.figure(figsize=(3, 1), dpi=300, frameon=False)
    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot display
    ax.axis([0, dispsize[0], 0, dispsize[1]])
    ax.imshow(screen)  # , origin='upper')
    # white screen

    return fig, ax

def draw_heatmap(gazepoints, dispsize, imagefile=None, alpha=0.5, savefilename=None, gaussianwh=200, gaussiansd=None):
    """Draws a heatmap of the provided fixations, optionally drawn over an
    image, and optionally allocating more weight to fixations with a higher
    duration.

    arguments

    gazepoints		-	a list of gazepoint tuples (x, y)
    
    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    alpha		-	float between 0 and 1, indicating the transparancy of
                    the heatmap, where 0 is completely transparant and 1
                    is completely untransparant (default = 0.5)
    savefilename	-	full path to the file in which the heatmap should be
                    saved, or None to not save the file (default = None)

    returns

    fig			-	a matplotlib.pyplot Figure instance, containing the
                    heatmap
    """

    # IMAGE
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # HEATMAP
    # Gaussian
    gwh = gaussianwh
    gsdwh = gwh / 6 if (gaussiansd is None) else gaussiansd
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = int(gwh / 2)
    heatmapsize = dispsize[1] + 2 * strt, dispsize[0] + 2 * strt
    heatmap = np.ones(heatmapsize, dtype=float)
    # create heatmap
    for i in range(0, len(gazepoints)):
        # get x and y coordinates
        x = strt + int(gazepoints[i][0]) - int(gwh / 2)
        y = strt + int(gazepoints[i][1]) - int(gwh / 2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh];
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y - dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * gazepoints[i][2]
            except:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[y:y + gwh, x:x + gwh] += gaus * gazepoints[i][2]
    # resize heatmap
    heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]
    # remove zeros
    lowbound = np.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = np.NaN
    # draw heatmap on top of image
    ax.imshow(heatmap, cmap='jet', alpha=alpha)
    # white background
    ax.set_axis_off()
    ax.set_facecolor('white')
    # put legends
    ax.set_title('Heatmap of the fixations')
    ax.set_xlabel('X-axis Pixels')
    ax.set_ylabel('Y-axis Pixels')
    # FINISH PLOT
    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)

    return fig

def plot_heatmap_fixations(fixations, file, XAI, samples, window_size = (int(1920), int(1080)), computer_width =38, computer_height = 25, alpha = 0.6, gaussiannwh = 300, gaussiansd = 33):
        '''Plot a heatmap of the fixations, taken from https://github.com/TobiasRoeddiger/GazePointHeatMap/blob/master/gazeheatplot.py'''
        # gaussian kernel parameters
        gwh = gaussiannwh
        gsdwh = gaussiansd
        gaus = gaussian(gwh, gsdwh)

        # matrix of zeroes for heatmap
        strt = int(gwh / 2)
        heatmapsize = int(window_size[1] + 2 * strt), int(window_size[0] + 2 * strt)
        heatmap = np.ones(heatmapsize, dtype=float)

        # create heatmap
        for fixation in fixations:
            x = int(fixation['average_position_x']) 
            y = int(fixation['average_position_y'])
            # correct Gaussian size if fixation is outside of display boundaries
            if (not 0 < x < window_size[0]) or (not 0 < y < window_size[1]):
                hadj = [0, gwh]
                vadj = [0, gwh]
                if x < 0:
                    hadj[0] = abs(x)
                    x = 0
                elif computer_width < x:
                    hadj[1] = gwh - int(x - window_size[0])
                if y < 0:
                    vadj[0] = abs(y)
                    y = 0
                elif computer_height < y:
                    vadj[1] = gwh - int(y - window_size[1])
                
                try: 
                    heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * fixation['duration']
                except:
                    # fixation was probably outside of display
                    pass
            else:
                # add Gaussian to the current heatmap
                heatmap[y:y + gwh, x:x + gwh] += gaus * fixation['duration']
        # resize heatmap
        heatmap = heatmap[int(strt):int(window_size[1] + strt), int(strt):int(window_size[0] + strt)]

        # remove zeros
        lowbound = np.mean(heatmap[heatmap > 0])
        heatmap[heatmap < lowbound] = 0

        # plot heatmap
        plt.figure()
        # draw image file
        if file == 'O':
            image = plt.imread('images/screen_print_control.jpeg')
        else:
            image = plt.imread('images/screen_print_explanation.jpeg')

        plt.imshow(image, extent=[0, window_size[0], 0, window_size[1]], alpha=0.7)

        # draw heatmap on top of image
        average_heatmap = heatmap/samples # average heatmap
        # Arial font
        plt.rcParams['font.family'] = 'Arial'

        plt.imshow(average_heatmap, cmap = 'viridis', alpha = alpha) 
        # title 
        plt.title('Fixation Heatmap for XAI = {XAI} condition'.format(XAI = XAI))

        # labels
        plt.xlabel('X-axis Pixels')
        plt.ylabel('Y-axis Pixels')
        plt.xlim(0, window_size[0])
        plt.ylim(0, window_size[1])

        # plt.gca().invert_yaxis()  
        plt.colorbar(label = 'Based on duration and position of fixations')
        plt.grid()
        plt.savefig('images/heatmap_fixations'+ file + '.png', dpi = 1200)

if __name__ == '__main__':

    directory= 'gaze_data/fixation_data_heat_map_reliance/'

    # append all individual dataframes into one dataframe

    reliance_data = pd.read_csv('gaze_data/reliance_groups/groups.csv')

    # remove 'O23' or user == 'OX30' from the dataset
    reliance_data = reliance_data[~reliance_data['ID'].isin(['O23', 'OX30'])]

    # Only OX condition (XAI = True), check reliance groups
    # Reliance, Underreliance, Overreliance, No Underreliance 



    data_frames_o_condition = [] # just the screen
    data_frames_ox_condition = [] # just the screen
    fixations_O = []
    fixations_OX = []
    samples = 14
    for file in os.listdir(directory):
        if file.endswith('.csv') and 'O' in file and 'OX' not in file and 'elmo' not in file:
            data = pd.read_csv(directory + file)
            data_frames_o_condition.append(data)
        elif file.endswith('.csv') and 'OX' in file and 'elmo' not in file:
            data = pd.read_csv(directory + file)
            data_frames_ox_condition.append(data)

    if data_frames_o_condition:
        combined_data_o = pd.concat(data_frames_o_condition, ignore_index=True)
        for _, row in combined_data_o.iterrows():
            fixation = {
                'average_position_x': row['average_x'],
                'average_position_y': row['average_y'],
                'duration': row['duration']
            }
            fixations_O.append(fixation)
        print(fixations_O)
        XAI = False
        plot_heatmap_fixations(fixations_O, 'O', XAI, samples)

    if data_frames_ox_condition:
        combined_data_ox = pd.concat(data_frames_ox_condition, ignore_index=True)
        for _, row in combined_data_ox.iterrows():
            fixation = {
                'average_position_x': row['average_x'],
                'average_position_y': row['average_y'],
                'duration': row['duration']
            }
            fixations_OX.append(fixation)
        XAI = True
        plot_heatmap_fixations(fixations_OX, 'OX', XAI, samples)

        plt.show()















    
    