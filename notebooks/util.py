# encoding: utf-8
# pylint: disable=C0103
"""
Util
====

Utility functions
-----------------

.. autosummary::
    :toctree: generated/

    compute_bpms
    smooth_bpms
    trim_beats


"""

import csv
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.axes import Axes
import matplotlib.ticker as mticker

from scipy import signal
from scipy.io import wavfile
import scipy.ndimage.filters
from scipy.signal import savgol_filter
from scipy.stats import pearsonr


def trim_beats(beats, beat_labs, downbeats, ini_bar, num_bars):
    """ Trim beats to select certain section within the recording.
    """

    ini_downbeat = downbeats[ini_bar-1]
    end_downbeat = downbeats[ini_bar-1+num_bars-1]

    inds_beats = np.nonzero(np.logical_and(beats >= ini_downbeat,
                                           beats <= end_downbeat))
    inds_beats = inds_beats[0]
    beats = beats[inds_beats]
    beat_labs = beat_labs[inds_beats[0]:inds_beats[-1]+1]

    return beats, beat_labs


def trim_downbeats(downbeats, downbeat_labs, ini_bar, num_bars):
    """ Trim downbeats to select certain section within the recording.
    """

    downbeats = downbeats[ini_bar-1:ini_bar-1+num_bars]
    downbeat_labs = downbeat_labs[ini_bar-1:ini_bar-1+num_bars]

    return downbeats, downbeat_labs


def compute_bpms(beats):
    """ Compute tempo curve from beat annotations
    """

    durs = beats[1:] - beats[:-1]
    bpms = np.round(60 / durs)

    return bpms

def smooth_bpms(bpms, win_len=15, poly_ord=3):
    """ Smooth tempo curve using savgol filter
    """
    
    bpms_smooth = savgol_filter(bpms, win_len, poly_ord)

    return bpms_smooth


def visualize_scape_plot(SP, Fs=1, ax=None, figsize=(4, 3), title='', 
                         xlabel='Centro (compás)', ylabel='Largo (compases)',
                         thr=2, txt=''):  
    """Visualize scape plot

    Args:
        SP: Scape plot data (encodes as start-duration matrix)
        Fs: Sampling rate
        ax, figsize, title, xlabel, ylabel: Standard parameters for plotting

    Returns:
        fig, ax, im
    """    
    fig = None
    if(ax==None):
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()    
    N = SP.shape[0]
    SP_vis = np.zeros((N,N))
    for length_minus_one in range(N):
        for start in range(N-length_minus_one):
            center = start + length_minus_one//2
            SP_vis[length_minus_one,center] = SP[length_minus_one,start]
            
    extent = np.array([-0.5, (N-1)+0.5, -0.5, (N-1)+0.5])/Fs  
    # im = plt.imshow(SP_vis, cmap='hot_r', aspect='auto', origin='lower', extent=extent)
    # cmap_custom = cm.get_cmap('Spectral_r')
    cmap_custom = copy.copy(cm.get_cmap("Spectral_r"))
    cmap_custom.set_bad('white')
    im = plt.imshow(np.ma.masked_values(SP_vis, 0), cmap=cmap_custom, aspect='auto', origin='lower', extent=extent, vmin=-1, vmax=1) 
    #im = plt.imshow(SP_vis, cmap='Spectral_r', aspect='auto', origin='lower', extent=extent, vmin=-1, vmax=1) 
    
    # Loop over data dimensions and create text annotations.
    for i in range(N):
        for j in range(N):
            if SP_vis[i, j] >= thr:
                #text = ax.text(j, i, SP_vis[i, j],
                 text = ax.text(j, i, txt,
                               ha="center", va="center", color="w")

    x = np.asarray(range(N))
    x_half_lower = x/2
    x_half_upper = x/2 + N/2 - 1/2 
    plt.plot(x_half_lower/Fs, x/Fs+3/4, '-', linewidth=3, color='black')
    #plt.plot(x_half_lower/Fs, x/Fs, '-', linewidth=3, color='black')
    plt.plot(x_half_upper/Fs, np.flip(x, axis=0)/Fs, '-', linewidth=3, color='black')    
    #plt.plot(x_half_upper/Fs, np.flip(x, axis=0)/Fs, '-', linewidth=3, color='black')    
    plt.plot(x/Fs, np.zeros(N)/Fs, '-', linewidth=3, color='black')
    plt.xlim([0,(N-1)/Fs])
    plt.ylim([0,(N-1)/Fs])

    ticks = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    new_ticks = [int(tick)+1 for tick in ticks]
    ax.set_xticklabels(new_ticks)
    ticks = ax.get_yticks().tolist()
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    new_ticks = [int(tick)+1 for tick in ticks]
    ax.set_yticklabels(new_ticks)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.colorbar(im, ax=ax) 
    
    return fig, ax, im


def visualize_tempo_plot(timing_data, ax=None, figsize=(4, 3), smooth=True, colors=None,
                         title='Tempo curve', xlabel='Time (bar)', ylabel='Tempo (BPM)'):  
    """Visualize tempo plot

    Args:
        ax, figsize, title, xlabel, ylabel: standard parameters for plotting

    Returns:
        fig, ax, im
    """    
    fig = None
    if(ax==None):
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()    
    if colors==None:
        colors = ['royalblue', 'seagreen', 'crimson', 'blueviolet', 'orange']

    for ind, recording in enumerate(timing_data):
        if smooth:
            plt.plot(recording["bpms_smooth"], label=recording["name"], color=colors[ind])
        else:
            plt.plot(recording["bpms"], label=recording["name"], color=colors[ind])

    xlabs = [x.replace('.1', '') if '.1' in x else ' ' for x in recording["beat_labs"]]
    ax.xaxis.set_ticks(range(recording["bpms"].shape[0]+1))
    ax.set_xticklabels(xlabs)
    plt.legend(loc='upper right')
    ax.set_title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    return fig, ax


def load_beats(labels_file, delimiter=',', times_col=0, labels_col=1):
    """Load annotated beats from text (csv) file.

    Parameters
    ----------
    labels_file : str
        name (including path) of the input file
    delimiter : str
        string used as delimiter in the input file
    times_col : int
        column index of the time data
    labels_col : int
        column index of the label data

    Returns
    -------
    beat_times : np.ndarray
        time instants of the beats
    beat_labels : list
        labels at the beats (e.g. 1.1, 1.2, etc)

    Examples
    --------

    Load an included example file from the candombe dataset.
    http://www.eumus.edu.uy/candombe/datasets/ISMIR2015/

    >>> annotations_file = carat.util.example_beats_file(num_file=1)
    >>> beats, beat_labs = annotations.load_beats(annotations_file)
    >>> beats[0]
    0.548571428
    >>> beat_labs[0]
    '1.1'

    Load an included example file from the samba dataset.
    http://www.smt.ufrj.br/~starel/datasets/brid.html

    >>> annotations_file = carat.util.example_beats_file(num_file=3)
    >>> beats, beat_labs = annotations.load_beats(annotations_file, delimiter=' ')
    >>> beats
    array([ 2.088,  2.559,  3.012,   3.48,  3.933,   4.41,  4.867,   5.32,
            5.771,  6.229,   6.69,  7.167,  7.633,  8.092,  8.545,   9.01,
             9.48,  9.943, 10.404, 10.865, 11.322, 11.79 , 12.251, 12.714,
           13.167, 13.624, 14.094, 14.559, 15.014, 15.473, 15.931,   16.4,
           16.865, 17.331, 17.788, 18.249, 18.706, 19.167, 19.643, 20.096,
           20.557, 21.018, 21.494, 21.945, 22.408, 22.869, 23.31 , 23.773,
           24.235, 24.692, 25.151, 25.608, 26.063, 26.52 ])

    >>> beat_labs
    ['1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2',
     '1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2',
     '1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2',
     '1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2']


    Notes
    -----
    It is assumed that the beat annotations are provided as a text file (csv).
    Apart from the time data (mandatory) a label can be given for each beat (optional).
    The time data is assumed to be given in seconds.
    The labels may indicate the beat number within the rhythm cycle (e.g. 1.1, 1.2, or 1, 2).
    """

    # read beat time instants
    beat_times = np.genfromtxt(labels_file, delimiter=delimiter, usecols=(times_col))

    # read beat labels
    with open(labels_file, 'r') as fi:
        reader = csv.reader(fi, delimiter=delimiter)
        # number of columns
        ncol = len(next(reader))
        # check if there are no labels
        if ncol == 1:
            beat_labels = []
        else:
            fi.seek(0)
            beat_labels = [row[labels_col] for row in reader]

    return beat_times, beat_labels


def load_downbeats(labels_file, delimiter=',', times_col=0, labels_col=1, downbeat_label='.1'):
    """Load annotated downbeats from text (csv) file.

    Parameters
    ----------
    labels_file : str
        name (including path) of the input file
    delimiter : str
        string used as delimiter in the input file
    times_col : int
        column index of the time data
    labels_col : int
        column index of the label data
    downbeat_label : str
        string to look for in the label data to select downbeats

    Returns
    -------
    downbeat_times : np.ndarray
        time instants of the downbeats
    downbeat_labels : list
        labels at the downbeats

    Examples
    --------

    Load an included example file from the candombe dataset.
    http://www.eumus.edu.uy/candombe/datasets/ISMIR2015/

    >>> annotations_file = carat.util.example_beats_file(num_file=1)
    >>> downbeats, downbeat_labs = carat.annotations.load_downbeats(annotations_file)
    >>> downbeats[:3]
    array([0.54857143, 2.33265306, 4.11530612])
    >>> downbeat_labs[:3]
    ['1.1', '2.1', '3.1']


    Load an included example file from the samba dataset.
    http://www.smt.ufrj.br/~starel/datasets/brid.html

    >>> annotations_file = carat.util.example_beats_file(num_file=3)
    >>> downbeats, downbeat_labs = annotations.load_downbeats(annotations_file,
                                                              delimiter=' ', downbeat_label='1')
    >>> downbeats
    array([ 2.088,  3.012,  3.933,  4.867,  5.771,  6.69 ,  7.633,  8.545,
            9.48 , 10.404, 11.322, 12.251, 13.167, 14.094, 15.014, 15.931,
           16.865, 17.788, 18.706, 19.643, 20.557, 21.494, 22.408,  23.31,
           24.235, 25.151, 26.063])
    >>> downbeat_labs
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',
     '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']


    Notes
    -----
    It is assumed that the annotations are provided as a text file (csv).
    Apart from the time data (mandatory) a label can be given for each downbeat (optional).
    The time data is assumed to be given in seconds.

    If a single file contains both beats and downbeats then the downbeat_label is used to select
    downbeats. The downbeats are those beats whose label has the given downbeat_label string. For
    instance the beat labels can be numbers, e.g. '1', '2'. Then, the downbeat_label is just '1'.
    This is the case for the BRID samba dataset. In the case of the candombe dataset, the beat
    labels indicate bar number and beat number. For instance, '1.1', '1.2', '1.3' and '1.4' are the
    four beats of the first bar. Hence, the string needed to indetify the downbeats is '.1'.
    """

    # read file as beats
    beat_times, beat_labs = load_beats(labels_file, delimiter=delimiter, times_col=times_col,
                                       labels_col=labels_col)

    # if there are no labels in the file or downbeat_label is None, then all entries are downbeats
    if not beat_labs or downbeat_label is None:
        downbeat_times, downbeat_labs = beat_times, beat_labs
    else:
        # get downbeat instants and labels by finding the string downbeat_label in the beat labels
        ind_downbeats = [ind_beat for ind_beat in range(len(beat_labs)) if downbeat_label in
                         beat_labs[ind_beat]]
        downbeat_times = beat_times[ind_downbeats]
        downbeat_labs = [beat_labs[ind_downbeat] for ind_downbeat in ind_downbeats]

    return downbeat_times, downbeat_labs


def compute_correlation_matrix(timing_data, n=4,):
    """Compute correlation matrix for two recordings.

    Parameters
    ----------
    timing_data : list
        list of dictionaries containing the timing data
    n : int
        grouping factor

    Returns
    -------
    CM : np.ndarray
        correlation matrix
    """

    # select the two recordings to compare
    recording1 = timing_data[0]
    recording2 = timing_data[1]
    data1 = recording1["bpms_smooth"]
    data2 = recording2["bpms_smooth"]

    # total data length
    N = data1.shape[0]
    # matrix elements
    M = int(np.floor(N / n))

    # correlation matrix
    CM = np.zeros((M,M))

    for k in range(M):
        # segment length
        sl = (k+1) * n
        # segment hop
        sh = n
        for s in range(M-k):
            # segment indexes
            ind_ini = s * sh 
            ind_end = ind_ini + sl
            # calculate Pearson's correlation
            corr, _ = pearsonr(data1[ind_ini:ind_end], data2[ind_ini:ind_end])
            # save correlation value
            CM[k, s] = corr

    return CM


def wave_plot(y, sr=22050, x_axis='time', beats=None, beat_labs=None,
              ax=None, **kwargs):
    '''Plot an audio waveform and beat labels (optinal).


    Parameters
    ----------
    y : np.ndarray
        audio time series

    sr : number > 0 [scalar]
        sampling rate of `y`

    x_axis : str {'time', 'off', 'none'} or None
        If 'time', the x-axis is given time tick-marks.

    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.

    kwargs
        Additional keyword arguments to `matplotlib.`

    Returns
    -------

    See also
    --------


    Examples
    --------
    '''

    kwargs.setdefault('color', 'royalblue')
    kwargs.setdefault('linestyle', '-')
    kwargs.setdefault('alpha', 0.6)

    if y.ndim > 1:
        raise ValueError("`y` must be a one dimensional array. "
                         "Found y.ndim={}".format(y.ndim))

    # time array in seconds
    time = np.arange(y.size)/sr
    # its maximum value
    max_time = np.max(time)

    # check axes and create it if needed
    axes = __check_axes(ax)

    # plot waveform
    out = axes.plot(time, y, **kwargs)

    if beats is not None:
        __plot_beats(beats, max_time, axes, beat_labs=beat_labs, **kwargs)

    # format x axis
    if x_axis == 'time':
        # axes.xaxis.set_major_formatter(TimeFormatter(lag=False))
        axes.xaxis.set_label_text('Tiempo (segundos)')
    elif x_axis is None or x_axis in ['off', 'none']:
        axes.set_xticks([])

    return out


def __plot_beats(beats, max_time, ax, beat_labs=None, **kwargs):
    '''Plot beat labels.


    Parameters
    ----------
    beats : np.ndarray
        audio time series

    beat_labs : list
        beat labels

    x_axis : str {'time', 'off', 'none'} or None
        If 'time', the x-axis is given time tick-marks.

    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.

    kwargs
        Additional keyword arguments to `matplotlib.`

    Returns
    -------

    See also
    --------


    Examples
    --------
    '''

    kwargs['color'] = 'black'
    kwargs.setdefault('linestyle', '-')
    kwargs['alpha'] = 0.3
    kwargs.setdefault('linewidth', 2)

    # replace nan values to 0
    beats = np.nan_to_num(beats)

    # consider beats (and labels) bellow max_time
    ind_beat = find_nearest(beats, max_time)
    new_beats = beats[:ind_beat]
    if beat_labs is not None:
        new_labs = beat_labs[:ind_beat]

    # plot beat annotations
    for beat in new_beats:
        ax.axvline(x=beat, **kwargs)

    # set ticks and labels
    # ax2 = ax.twiny()
    # ax2.set_xlim(ax.get_xlim())
    # ax2.set_xticks(new_beats)
    # if beat_labs is not None:
    #     ax2.set_xticklabels(new_labs)
    # else:
    #     ax2.set_xticklabels([])

    # return ax2

def __check_axes(axes):
    '''Check if "axes" is an instance of an axis object.'''
    if axes is None:
        import matplotlib.pyplot as plt
        axes = plt.gca()
    elif not isinstance(axes, Axes):
        raise ValueError("`axes` must be an instance of matplotlib.axes.Axes. "
                         "Found type(axes)={}".format(type(axes)))
    return axes

def wavread(filename):
    """ read wave file using scipy.io.wavfile
    """
    # read wave file 
    fs, y = wavfile.read(filename)
    
    # convert scipy data array to float and normalize amplitude
    if y.dtype == np.int16:
        y = y.astype(float)
        y /= 32767
    
    return fs, y

def find_nearest(array, value):
    """Find index of the nearest value of an array to a given value

    Parameters
    ----------
    array (numpy.ndarray)  : array
    value (float)          : value

    Returns
    -------
    idx (int)              : index of nearest value in the array
    """

    idx = (np.abs(array-value)).argmin()

    return idx

