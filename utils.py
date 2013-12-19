#!/usr/bin/env python
# Copyright 2013 Quy Tonthat <qtonthat@gmail.com>

from __future__ import print_function

import os
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import bz2, gzip

from logdata import APlogData
from apmlogplotlib import APMlogPlotter, logdata_open
from apmlogplotlib import log_plotter

def open_logfile(filename):
    if filename.endswith('.log'):
        return open(filename)
    if filename.endswith('.bz2'):
        return bz2.BZ2File(filename)   # for python3 it can be bz2.open()
    if filename.endswith('.gz'):
        return gzip.GzipFile(filename)

def plot_AccZ_BarAlt(filename):
    data = APlogData()
    data.parse(open_logfile(filename))
    plotter = APMlogPlotter(data)
    plotter.add_plot('AccX', hlines=(3,-3), redraw=False)
    plotter.add_plot('BarAlt', subplot=1, redraw=False)
    plotter.draw()
    return plotter


def hist_of_Acc(filename, acc_type='X', lower_limit=-3, upper_limit=3, label=None, with_good_example=False, added=False, **kwargs):
    if label == None:
        label = os.path.basename(filename)
        label = re.sub(r'\.log((\.bz2|\.gzip)*)$','', label)
    acc_type = acc_type.upper()
    if acc_type not in ['X', 'Y', 'Z']:
        raise Exeption("Illegal acc_type %s. Must be one of 'X', 'Y','Z'" % acc_type)
    if acc_type in ['X', 'Y']:
        xmin = -15
        xmax = 15
    else:
        xmin = -30
        xmax = 10

    data = APlogData()
    data.parse(open_logfile(filename))
    plt.hist([float(x) for x in data['IMU']['Acc%s' % acc_type] if float(x) >= xmin and float(x) <= xmax], 200, label=label, **kwargs)

    if with_good_example:
        data_goodvibr = APlogData()
        data_goodvibr.parse(open_logfile('logs/good_vibr.log.bz2'))
        plt.hist([float(x) for x in data_goodvibr['IMU']['Acc%s' % acc_type] if float(x) >= xmin and float(x) <= xmax], 200, label='good', color='yellow', alpha=0.5, zorder=50)

    plt.legend(shadow=True)
    plt.draw()

    if not added:
        axes = plt.gca()
        # axes.axvline(lower_limit, color='r', linestyle='--')
        # axes.axvline(upper_limit, color='r', linestyle='--')
        axes.axvspan(lower_limit, upper_limit, facecolor='0.8', alpha=0.5, edgecolor='red', zorder=100)

        plt.draw()

        # This has to be done after the draw() or we will get an empty list
        # for ticks and labels.
        # TODO: This is my way. There must be a simpler way to add ticks and
        # set color for them
    
        # Make negative numbers look like the rest of the tickers.
        if lower_limit < 0:
            #lower_label = u'\u2212%d' % -lower_limit    #'2212' is Latex hyphen
            lower_label = r'$ %d $' % lower_limit
        else:
            lower_label = str(lower_limit)
        if upper_limit < 0:
            upper_label = r'$ %d $' % -upper_limit
        else:
            upper_label = str(upper_limit)
    
        new_ticks = list(axes.get_xticks()) + [lower_limit, upper_limit]
        new_labels = [ x.get_text() for x in axes.get_xticklabels() ] + [lower_label, upper_label]
        axes.set_xticks(new_ticks)
        axes.set_xticklabels(new_labels)
        # Now paint the added tickers red
        for l in axes.get_xticklabels()[-2:]:
            l.set_color('r')

def hist_of_AccX(filename, added=False, label=None, with_good_example=False, **kwargs):
    """To add more files into an existing hist graph, call these functions
    with add=True. Optionally, set alpha=f where 0.0 < f < 1.0. **kwargs will
    be passed to hist()
    """
    hist_of_Acc(filename, added=added, acc_type='X', lower_limit=-3, upper_limit=3, label=label, with_good_example=with_good_example, **kwargs)

def hist_of_AccY(filename, added=False, label=None, with_good_example=False, **kwargs):
    hist_of_Acc(filename, added=added, acc_type='Y', lower_limit=-3, upper_limit=3, label=label, with_good_example=with_good_example, **kwargs)

def hist_of_AccZ(filename, added=False, label=None, with_good_example=False, **kwargs):
    hist_of_Acc(filename, added=added, acc_type='Z', lower_limit=-15, upper_limit=-5, label=label, with_good_example=with_good_example, **kwargs)

class HSelect:
    def __init__(self, callback, figure=None):
        self.callback = callback
        if not figure:
            self.figure = plt.gcf()
        else:
            self.figure = figure
        self.start = None
        self.cid_leftclick = None
        self.cid_rightclick = None

    def connect(self):
        """connect to all the events we need"""
        self.cid_click = self.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        # print('HSelect: connected')

    def on_press(self, event):
        # print('HSelect: got event')
        if event.button == 1:   # left button
            self.start = event.xdata
            #print('left button: %f' % self.start)
        elif event.button == 3: # right button
            if self.start == None:
                return
            #print('right button: %f' % event.xdata)
            self.callback(self.start, event.xdata)
            self.start = None
        else:
            pass
        
def plot_vibration(logfile):
    """Plot Acc parameters. Select with left and right mouse button to plot
    histogram for the selected  sectioni
    """
    plotter = log_plotter(logfile)
    plotter.add_plot('AccX', hlines=(3,-3), redraw=False)
    plotter.add_plot('AccY', redraw=False)
    plotter.add_plot('AccZ', subplot=1, hlines=(-15,-5), redraw=False)
    plotter.add_plot('BarAlt', subplot=1, right_side=True, redraw=False)

    plotter.draw()
    plt.draw()

    def select_callback(start, end):
        accx = plotter._logdata['IMU']['AccX']
        start_index = int(len(accx) * start/plotter._max_samples)
        end_index = int(len(accx) * end/plotter._max_samples)
        if start_index > end_index:
            start_index, end_index = end_index, start_index
        # print('hist on %d-%d' % (start_index, end_index))
        data = [ float(x) for x in accx[start_index:end_index] ]
        figure = plt.figure()
        plt.hist(data, bins=100, figure=figure)


    hselect = HSelect(select_callback, plt.gcf())
    hselect.connect()
    # just to keep it survived.
    plotter.hselect = hselect

    # need to keep hselect somewhere or it will be destroyed
    return plotter

def failsafe(filename):
    data = APlogData()
    data.parse(open_logfile(filename))

    errs = zip(data['ERR']['Subsys'], data['ERR']['ECode'])
    for (errnum, errcode) in errs:
        print('error %s(%s):' % (errnum, errcode), end=' ')
        if errnum == '1':    # main. Never used
            print('Main -- Never used')
        elif errnum == '2':
            print('RC Radio', end=' ')
            if errcode == '0':
                print('resolved')
            elif errcode == '1':
                print('No signal for at least 2 seconds')
            else:
                print('unknown')
        elif errnum == '3':
            print('Compass', end=' ')
            if errcode == '0':
                print('resolved')
            elif errcode == '1':
                print('Failed to initialise')
            elif errcode == '2':
                print('Failed on read')
            else:
                print('Unknown')
        elif errnum == '4':
            print('Optical Flow', end=' ')
            if errcode == '1':
                print('Failed to initialise')
            else:
                print('Unknown')
        elif errnum == '5':
            print('Throttle', end=' ')
            if errcode == '0':
                print('resolved')
            elif errcode == '1':
                print('Too low')    # below FS_THR_VALUE
            else:
                print('unknown')
        elif errnum == '6':
            print('Battery', end=' ')
            if errcode == '1':
                print('low voltage or capacity') # voltage < LOW_VOLT or current > BATT_CAPACITY
            else:
                print('Unknown')
        elif errnum == '7':
            print('GPS', end=' ')
            if errcode == '0':
                print('recovered')
            elif errcode == '1':
                print('GPS lock lost for at least 5 seconds')
            else:
                print('unknown')
        elif errnum == '8':
            print('GCS', end=' ')
            if errcode == '0':
                print('recover')
            elif errcode == '1':
                print('joystick lost for at least 5 seconds')
            else:
                print('unknown')
        elif errnum == '9':
            print('Fence', end=' ')
            if errcode == '0':
                print('back within fence')
            elif errcode == '1':
                print('Altitude fence breached')
            elif errcode == '2':
                print('Circular fence breached')
            elif errcode == '3':
                print('Both altitude and Circular fence breached')
            else:
                print('unknown')
        elif errnum == '10':
            print('Unable to enter', end=' ')
            if errcode == '0':
                print('stabilise mode')
            elif errcode == '1':
                print('Acro mode')
            elif errcode == '2':
                print('AltHold mode')
            elif errcode == '3':
                print('Auto mode')
            elif errcode == '4':
                print('Guided mode')
            elif errcode == '5':
                print('Loiter mode')
            elif errcode == '6':
                print('RTL mode')
            elif errcode == '7':
                print('Circle mode')
            elif errcode == '8':
                print('Position mode')
            elif errcode == '9':
                print('Land mode')
            elif errcode == '10':
                print('OF_Loiter mode')
            else:
                print('unknown')
        else:
            print('Unknown error %s' % errnum)

def plot_fs_battery(filename, num_subplots=2):
    if num_subplots < 2:
        num_subplots = 2
    data = APlogData()
    data.parse(open_logfile(filename))

    if '6' not in data['ERR']['Subsys']:
        print('Note: No battery error in the log file')
        #return

    plotter = APMlogPlotter(data, num_subplots=num_subplots,
                                title=os.path.basename(filename))
    samples = plotter._max_samples
    low_volt = 100 * float(plotter._logdata.parms['LOW_VOLT'])
    plotter.add_plot('CURR.Volt', hlines=[low_volt], redraw=False)
    batt_capacity = float(plotter._logdata.parms['BATT_CAPACITY'])
    plotter.add_plot('CURR.CurrTot', hlines=[batt_capacity], subplot=1, redraw=False)

    plotter.add_plot('ThrOut', right_side=True, redraw=False)
    plotter.add_plot('BarAlt', subplot=1, right_side=True, own_axes=True, redraw=False)

    plotter.draw()
    plt.draw()
    return plotter

def plot_gps_status(logfile):
    plotter = log_plotter(logfile)
    plotter.add_plot('GPS.NSats', hlines=[9], redraw=False)
    plotter.add_plot('GPS.HDop', subplot=1, hlines=[2], redraw=False)
    plotter.add_plot('GPS.Status', subplot=1, right_side=True, redraw=False)

    plotter.draw()
    plt.draw()
    return plotter

def plot_alt(logfile):
    plotter = log_plotter(logfile)
    plotter.add_plot('CTUN.BarAlt', redraw=False)                 # From baro
    plotter.add_plot('GPS.RelAlt', redraw=False)                  # baro and acc
    plotter.add_plot('GPS.Alt', right_side=True, redraw=False)    # From GPS

    plotter.draw()
    plt.draw()
    return plotter

# Plot ThrOut for tuning THR_MID
def plot_throttle_out(logfile):
    plotter = log_plotter(logfile)
    plotter.add_plot('CTUN.ThrOut', redraw=False)
    #plotter.add_plot('CTUN.ThrIn', right_side=True, redraw=False)
    plotter.add_plot('CTUN.ThrIn', redraw=False)
    plotter.add_plot('GPS.RelAlt', subplot=1, redraw=False)
    plotter.add_plot('MODE.Mode', subplot=1, right_side=True, redraw=False)

    plotter.draw()
    plt.draw()
    return plotter

def plot_roll_pitch(logfile):
    plotter = log_plotter(logfile)
    plotter.add_plot('ATT.RollIn', redraw=False)
    plotter.add_plot('ATT.Roll', redraw=False)
    plotter.add_plot('ATT.PitchIn', 1, redraw=False)
    plotter.add_plot('ATT.Pitch', 1, redraw=False)

    plotter.draw()
    plt.draw()
    return plotter

def show_gps_time(*logfilenames):
    for filename in logfilenames:
        if filename.endswith('.bz2'):
            fp = bz2.BZ2File(filename)   # for python3 it can be bz2.open()
        elif filename.endswith('.gz'):
            fp = gzip.GzipFile(filename)
        else:
            fp = open(filename)
        data = APlogData()
        data.parse(fp)
        if 'GPS' not in data.keys():
            print('%s: No GPS data' % filename)
        else:
            # TODO: wrap around at the begin/end of the week
            print('%s:\t%s\t%s\t%d ms' % (filename, data['GPS']['Time'][0], data['GPS']['Time'][-1], int(data['GPS']['Time'][-1]) - int(data['GPS']['Time'][0])))

def plot_flightpath(logfile):
    data = logdata_open(logfile)
    if 'GPS' not in data.keys():
        print('GPS data not available from the log file')
        exit(1)
    lng = data['GPS']['Lng']
    lat = data['GPS']['Lat']
    ax = plt.gca()
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.plot(lng, lat, 'o-')
    plt.plot(lng[0], lat[0], 'rs', markersize=10)
    plt.plot(lng[-1], lat[-1], 'r*', markersize=20)

def plot_flightpath3D(logfile):
    data = logdata_open(logfile)
    if 'GPS' not in data.keys():
        print('GPS data not available from the log file')
        exit(1)
    lng = np.array([float(x) for x in data['GPS']['Lng']])
    lat = np.array([float(x) for x in data['GPS']['Lat']])
    alt = np.array([float(x) for x in data['GPS']['RelAlt']])
    ax = plt.gca(projection='3d')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude')

    plt.plot(lng, lat, alt)


# vim:set sw=4 softtabstop=4 et syntax=python filetype=python:
