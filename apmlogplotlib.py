# Copyright 2013 Quy Tonthat <qtonthat@gmail.com>
# Useful with ipython

from __future__ import print_function

import numpy
from numpy import arange
# from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
from matplotlib.ticker import MultipleLocator, FuncFormatter
import bz2, gzip

from logdata import APlogData, logdata_open

# Convert flight mode to number as in ArduCopter/defines.h
# new flight modes added in the future will break this!
flight_mode2num = {
	'STABILIZE':	0,
	'ACRO':	        1,
	'ALT_HOLD':	2,
	'AUTO':	        3,
	'GUIDED':	4,
	'LOITER':	5,
	'RTL':	        6,
	'CIRCLE':	7,
	'POSITION':	8,
	'LAND':	        9,
	'OF_LOITER':	10,
	'TOY_A':	11,
	'TOY_M':	12,
	'SPORT':	13,
}

def _mode_formatter(num, pos):
    for k in flight_mode2num:
        if flight_mode2num[k] == num:
            return k
    return '?'



def log_plotter(filename, **kwargs):
    data = APlogData()
    if filename.endswith('.bz2'):
        fp = bz2.BZ2File(filename)   # for python3 it can be bz2.open()
    elif filename.endswith('.gz'):
        fp = gzip.GzipFile(filename)
    else:
        fp = open(filename)
    data.parse(fp)
    plotter = APMlogPlotter(data, **kwargs)
    return plotter

class APMlogPlotLibError(Exception):
    """Base class for exceptions in this module."""
    pass

# Assuming vals and labels have the same lengh
def _add_ticks(axes, vals, labels, vaxis=True, **kwargs):

    nticks = len(vals)
    if nticks <= 0:
        return

    if vaxis:
        ticks = list(axes.get_yticks()) + list(vals)
        axes.set_yticks(ticks)

        # TODO: Change color of the added ticks.
        # get_text() returns empty string here (OK with hist)
        # This happens even after everything is already displayed.
        # We may have to use a tick formatter to do this.

        #labels = [ x.get_text() for x in axes.get_yticklabels() ] + labels
        # print(labels)
        #axes.set_yticklabels(labels)
        #for l in axes.get_yticklabels()[-nticks:]:
        #   l.set_color('r')
    else:
        ticks = list(axes.get_xticks()) + list(vals)
        axes.set_xticks(ticks)

        #labels = [ x.get_text() for x in axes.get_xticklabels() ] + labels
        #axes.set_xticklabels(labels)
        #for l in axes.get_xticklabels()[-nticks:]:
        #    l.set_color('r')

class LogGraph(object):
    def __init__(self, group, parm, xdata, ydata,
            hlines=[], vlines=[], extra_lines=[],
            graph_type = 'LINR', right_side=False, own_axes=False,
            x_locator=None, y_locator=None, x_formatter=None, y_formatter=None):
        self.axes = None
        self.lines = None
        self.group = group
        self.parm = parm
        self._ydata = ydata
        self._xdata = xdata
        self._graphtype = graph_type
        self._hlines = hlines
        self._vlines = vlines
        self._extra_lines = extra_lines
        # self.plotted = False
        self.right_side = right_side
        self.own_axes = own_axes
        self.x_locator = x_locator
        self.y_locator = y_locator
        self.x_formatter = x_formatter
        self.y_formatter = y_formatter

        self.visible = True
        self.all_artists = []

    def toggle_visible(self):
        self.visible = not self.visible
        self.update_visibility()
        return self.visible

    def plot(self, axes, **kwargs):
        self.all_artists = []
        self.lines = axes.plot(self._xdata, self._ydata, label=self.parm, **kwargs)
        self.all_artists = self.all_artists + self.lines
        self._axes = axes
        label = self.group + ' : ' + self.parm
        axis_label = axes.get_ylabel()
        if axis_label == '':
            axes.set_ylabel(label)
            if self.right_side:
                axes.axis["right"].label.set_color(self.lines[0].get_color())
            else:
                axes.axis["left"].label.set_color(self.lines[0].get_color())
        else:
            axes.set_ylabel(axis_label + ' / ' + label)

        for y in self._hlines:
            gr = self._axes.axhline(y, color='r', linestyle='--', linewidth=1.5)
            self.all_artists.append(gr)
        for x in self._vlines:
            gr = self._axes.axvline(x, color='r', linestyle='--', linewidth=1.5)
            self.all_artists.append(gr)
        if self._extra_lines:
            for (x,y) in self._extra_lines:
                gr = axes.plot(x, y, 'r--', linewidth=1.5)
                self.all_artists = self.all_artists + gr
        if self._hlines:
            _add_ticks(self._axes, self._hlines, [str(x) for x in self._hlines], vaxis=True)
        if self._vlines:
            _add_ticks(self._axes, self._vlines, [str(x) for x in self._vlines], vaxis=False)

        if self.x_locator:
            axes.xaxis.set_major_locator(self.x_locator)
        if self.y_locator:
            axes.yaxis.set_major_locator(self.y_locator)
        if self.x_formatter:
            axes.xaxis.set_major_formatter(self.x_formatter)
        if self.y_formatter:
            axes.yaxis.set_major_formatter(self.y_formatter)

        self.update_visibility()
    def update_visibility(self):
        for gr in self.all_artists:
            gr.set_visible(self.visible)

class SubPlot(object):
    _legend_alpha_off = 0.3
    def __init__(self):
        self.axes = None
        self.legend = None
        self.left_graphs = []
        self.right_graphs = []
        self.all_graphs = []
        self.legend_mapping = {}
        self.next_left_offset = 0
        self.next_right_offset = 0
        self.cid = None

    def new_legend(self):
        if self.cid:
            self.axes.figure.canvas.mpl_disconnect(self.cid)
        self.legend = self.axes.legend(loc='upper center', fancybox=True, shadow=True)
        self.legend.get_frame().set_alpha(0.5)
        self.legend_mapping = {}
        #print('number of legend lines: %d  numberof graphs %d' % (len(self.legend.get_lines()), len(self.all_graphs)))
        for llabel, graph in zip(self.legend.get_lines(), self.all_graphs):
            llabel.set_linewidth(3)
            llabel.set_picker(10)  # 5 pts tolerance
            llabel.set_visible(True)
            if not graph.visible:
                llabel.set_alpha(self._legend_alpha_off)
            else:
                llabel.set_alpha(1.0)
            self.legend_mapping[llabel] = graph
        #self.axes.callbacks.connect('pick_event', self.on_legend_pick)
        self.cid = self.axes.figure.canvas.mpl_connect('pick_event', self.on_legend_pick)

    def on_legend_pick(self, event):
        #print("Got picked")
        if event.mouseevent.inaxes != self.axes:
            #print("Not here")
            return
        legend_line = event.artist
        graph = self.legend_mapping[legend_line]
        vis = graph.toggle_visible()
        if vis:
            legend_line.set_alpha(1.0)
        else:
            legend_line.set_alpha(self._legend_alpha_off)

        plt.draw_if_interactive()

class APMlogPlotter(object):
    def __init__(self, logdata, num_subplots = 2, figure = None, title=None, time_based=True):
        # how far apart between parasite axes
        self._axes_offset = 60

        self._logdata = logdata
        self._num_subplots = num_subplots
        self._subplots = []
        self._groups_xdata = {}
        if 'GPS' not in logdata.keys():
            self._time_based = False
        else:
            self._time_based = time_based
        self._max_samples = self._get_max_sample(self._time_based)
        self.figure = figure
        self.title = title

        if num_subplots <= 0:
            raise APMlogPlotLibError('num_subplots must be >= 1 (', num_subplots, 'requested)')

        if self.figure == None:
            # self.figure = plt.figure()   # Create a new figure
            self.figure = plt.gcf()        # Use the current figure

        for k in self._logdata.keys():
            i = self._logdata[k].keys()[0]
            l = len(self._logdata[k][i])
            factor = float(self._max_samples)/l
            self._groups_xdata[k] = arange(0, factor * l, factor)

        self.figure.clear()
        for i in range(self._num_subplots):
            sp = SubPlot()
            if i == 0:
                sp.axes = self._host_subplot(self.figure, self._num_subplots, 1, i+1, axes_class=AA.Axes)
            else:
                sp.axes = self._host_subplot(self.figure, self._num_subplots, 1, i+1, axes_class=AA.Axes, sharex=self._subplots[0].axes)
            sp.axes.set_xlim(0, self._max_samples)
            if self._time_based:
                sp.axes.set_xlabel('time (s)')
            self._subplots.append(sp)
        all_plots = [ x.axes for x in self._subplots ]
        self._multi_cursor = MultiCursor(self.figure.canvas, all_plots, color='r', lw=1)

        self.figure.canvas.draw()

    def show_parameters(self):
        for g in self._logdata.keys():
            print(g, ':', ' ', sep='', end='')
            for parm in self._logdata[g].keys():
                print(parm, end=' ')
            print()

    def add_plot(self, parm_id, subplot=0, right_side=False, own_axes=False,
                            hlines=[], vlines=[], extra_lines=[], redraw=True,
                            inverse=False):
        """Add a graph to sublot sublot that represent parameter parm_id

            parm_id can be in the form 'group.parm' or 'parm'
        """
        if subplot >= self._num_subplots:
            raise APMlogPlotLibError('Invalid subplot', subplot,
                '. Should be in the range  0 to', self._num_subplots,
                'inclusively')
        group = None
        parm = None
        if '.' in parm_id:
            (group, parm) = parm_id.split('.')
        else:
            for g in self._logdata.keys():
                if parm_id in self._logdata[g].keys():
                    group = g
                    parm = parm_id
                    break
            else:
                raise APMlogPlotLibError(
                            '%s does not exist in the log data' % parm_id)
        # some data need some special treatments first
        if group == 'MODE':
            ydata = [flight_mode2num[x] for x in self._logdata[group][parm]]
        else:
            if inverse:
                ydata = [-float(x) for x in self._logdata[group][parm]]
            else:
                ydata = self._logdata[group][parm]
        xdata = self._groups_xdata[group]

        # Some special treatments for some parameters.
        kwargs = {}
        if group == 'MODE' and parm == 'Mode':
            kwargs['y_locator'] = MultipleLocator(1)
            kwargs['y_formatter'] = FuncFormatter(_mode_formatter)

        # TODO: set graph_type
        graph = LogGraph(group, parm, xdata, ydata,
                    hlines=hlines, vlines=vlines, extra_lines=extra_lines,
                    graph_type = 'LINE',
                    right_side = right_side, own_axes = own_axes, **kwargs)
        if right_side:
            self._subplots[subplot].right_graphs.append(graph)
        else:
            self._subplots[subplot].left_graphs.append(graph)
        self._subplots[subplot].all_graphs.append(graph)

        if redraw:
            self.draw()
        #self.draw()
        return graph

    def get_plot(self, parm, group=None):
        for plot in self._subplots:
            for graph in plot.all_graphs:
                if graph.parm == parm:
                    if not group:
                        return graph
                    elif graph.group == group:
                        return graph
        return None

    # Redraw everything for now
    # TODO: only draw new graphs and remove deleted graphs
    def draw(self):
        # self.figure.clear()  # can't clear it here that will wipe out what has been setup in __init__
        for subplot in self._subplots:
            subplot.axes.clear()
            subplot.next_left_offset = 0
            subplot.next_right_offset = 0
            if len(subplot.left_graphs) + len(subplot.right_graphs) <= 0:
                continue
            if len(subplot.left_graphs) > 0:
                # The first plot on left axes, use the default axes.
                graph = subplot.left_graphs[0]
                graph.plot(subplot.axes)
                subplot.next_left_offset -= self._axes_offset
            for graph in subplot.left_graphs[1:]:
                if graph.own_axes:
                    axes = subplot.axes.twinx()
                    new_fixed_axis = axes.get_grid_helper().new_fixed_axis
                    axes.axis["left"] = new_fixed_axis(loc="left",
                                        axes=axes,
                                        offset=(subplot.next_left_offset, 0))
                    subplot.next_left_offset -= self._axes_offset
                    axes.axis["left"].toggle(all=True)
                    axes.axis["right"].toggle(all=False)
                    graph.plot(axes)
                    subplot.next_left_offset -= self._axes_offset
                else:
                    graph.plot(subplot.axes)
            if self._time_based:
                subplot.axes.set_xlabel('time (s)')

            first_right = True
            for graph in subplot.right_graphs:
                if first_right:
                    axes = subplot.axes.twinx()
                    subplot.next_right_offset += self._axes_offset
                    first_right = False
                elif graph.own_axes:
                    axes = subplot.axes.twinx()
                    new_fixed_axis = axes.get_grid_helper().new_fixed_axis
                    axes.axis["right"] = new_fixed_axis(loc="right",
                                        axes=axes,
                                        offset=(subplot.next_right_offset, 0))
                    axes.axis["right"].toggle(all=True)
                    axes.axis["left"].toggle(all=False)
                    subplot.next_right_offset += self._axes_offset
                else:
                    pass
                graph.plot(axes)
            subplot.new_legend()

        self._subplots[0].axes.set_xlim(0, self._max_samples)

        if self.title and len(self._subplots):
            self._subplots[0].axes.set_title(self.title)

        # TODO: need to do self.draw() and tight_layout again everytime
        # the window is resized
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        # plt.draw_if_interactive()
        self.figure.canvas.draw()
        # self.figure.canvas.show()

    def save_script(self, outfile, stand_alone=True, logfilename=None):
        """Write a python script to reconstruct the same graphs later
        When stand_alone is True, the output file can be run as a normal script
        Non stand alone output is meant for intarctive sessions (e.g. using
        ipython)
        To use a non-standalone file, use apmlogplotlib.log_plotter() to create
        a plotteri, the script (using ipython %load or %run) and call
        load_plots() from the generated script.

        file can be a file name or a file object
        """
        if type(outfile) == str:
            fp = open(outfile, 'w')
        else:
            fp = outfile

        def write_args(graph, subplot_num, right_side):
            fp.write('"%s.%s"' % (graph.group, graph.parm))
            fp.write(', subplot=%d' % subplot_num)
            fp.write(', right_side=%s' % right_side)
            fp.write(', own_axes={0}'.format(graph.own_axes))
            fp.write(', hlines=%s' %repr(graph._hlines))
            fp.write(', vlines=%s' %repr(graph._extra_lines))
            fp.write(', vlines=%s' %repr(graph._extra_lines), redraw=False)

        if stand_alone:
            fp.write('#!/usr/bin/env python\n')
            fp.write('# Generated by APMlogPlotter.save_script\n')
            fp.write('import sys\n')
            fp.write('import matplotlib.pyplot as plt\n')
            # Assuming we run the script in the directory where apmlogplotlib is
            # TODO: import using package name when the stuff is install in the
            # right places.
            fp.write('from apmlogplotlib import APMlogPlotter\n')
            fp.write('from apmlogplotlib import log_plotter\n')

        fp.write('def load_plots(plotter):\n')
        for spnum in range(self._num_subplots):
            for graph in self._subplots[spnum].left_graphs:
                fp.write('    plotter.add_plot(')
                write_args(graph, spnum, 'False')
                fp.write(')\n')
            for graph in self._subplots[spnum].right_graphs:
                fp.write('    plotter.add_plot(')
                write_args(graph, spnum, 'True')
                fp.write(')\n')
            fp.write('    plotter.draw()\n')

        if stand_alone:
            fp.write("if __name__ == '__main__':\n")
            fp.write('    if len(sys.argv) > 1:\n')
            fp.write('        plotter = log_plotter(sys.argv[1])\n')
            fp.write('    else:\n')
            if logfilename:
                fp.write('        plotter = log_plotter("%s")\n' % logfilename)
            else:
                fp.write('        print("Usage: %s log_file_name" % sys.argv[0])\n')
                fp.write('        sys.exit(1)\n')

            fp.write('    load_plots(plotter)\n')
            fp.write('    plt.show()\n')

    def _get_max_sample(self, time_based):
        if time_based:
            # Exception will be raised here if GPS data is not logged
            time = self._logdata['GPS']['Time']
            # The UBlox GPS store time in miliseconds since the beginning
            # of the week
            maxval = (int(time[-1]) - int(time[0])) / 1000
        else:
            maxval = max([ len(self._logdata[x][self._logdata[x].keys()[0]]) for x in self._logdata.keys() ])
        return maxval

    def _host_subplot(self, fig, *args, **kwargs):
        from mpl_toolkits.axes_grid1.parasite_axes import host_subplot_class_factory
        axes_class = kwargs.pop("axes_class", None)
        host_subplot_class = host_subplot_class_factory(axes_class)
        ax = host_subplot_class(fig, *args, **kwargs)
        fig.add_subplot(ax)
        plt.draw_if_interactive()
        return ax

if __name__ == '__main__':
    import sys

    if (len(sys.argv) > 1):
        logfile = sys.argv[1]
    else:
        logfile = 'test.log'

    plotter = log_plotter(logfile, title='test')
    plotter.add_plot('IMU.AccX', redraw=False)
    plotter.add_plot('IMU.AccY', redraw=False)
    plotter.add_plot('IMU.AccZ', own_axes=True, redraw=False)
    plotter.add_plot('IMU.AccZ', subplot=1, redraw=False)
    plotter.add_plot('IMU.AccY', subplot=1, right_side=True, redraw=False)
    plotter.add_plot('IMU.AccX', subplot=1, right_side=True, own_axes=True, redraw=False)

    plotter.draw()

    plotter.figure.canvas.draw()
    plotter.figure.canvas.show()
    plt.show()

# vim:set sw=4 softtabstop=4 et syntax=python filetype=python:
