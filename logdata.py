#!/usr/bin/env python
# Copyright 2013 Quy Tonthat <qtonthat@gmail.com>

from __future__ import print_function

import sys
import pickle
import bz2, gzip

def logdata_open(filename):
    data = APlogData()
    if filename.endswith('.bz2'):
        fp = bz2.BZ2File(filename)   # for python3 it can be bz2.open()
    elif filename.endswith('.gz'):
        fp = gzip.GzipFile(filename)
    else:
        fp = open(filename)
    data.parse(fp)
    return data

class APlogError(Exception):
    """Base class for exceptions in this module."""
    pass

# TODO: how to implement 'in' as in 'for k in xxx' ?
# for now use 'for k in xxx.keys() instead
class LogData(object):
    def __init__(self):
        self.data = {}
    def keys(self):
        return(self.data.keys())
    def values(self):
        return(self.data.values())
    def items(self):
        return(self.data.items())
    #def has_key(self, key):
    #   return self.data.has_key(key)
    def get(self, key, default=None):
        return self.data.get(key, default)

    def __getitem__(self, key):
        return(self.data.__getitem__(key))

    def __repr__(self):
        return(self.data.__repr__())

class APlogData(LogData):
    """Dataflash Log data from a file like object.
    """
    def __init__(self):
        self.firmware = None
        self.free_RAM = None
        self.hardware = None
        self.formats = {}
        self.formats_datatypes = {}
        self.parms = {}

        self.parm_events = {}
        # Flight mode won't get logged until changed so we just make up
        # a dummy one to start with
        self.parm_events['MODE'] = ('MODE', 'STABILIZE', 0)

    def parse(self, filefp):
        self.data = {}
        self.formats = {}
        self.parms = {}

        # skip blank lines and junky stuff at the beginning of the file
        # for line in filefp:   # problem with python2
        line_num = 0
        while True:
            line = filefp.readline()
            line_num += 1
            if not line:
                print('Not a log file')
                return
            line = line.strip()
            if line.startswith('Ardu'):
                break

        # The header
        self.firmware = line.strip()
        self.free_RAM = filefp.readline().strip()
        line_num += 1
        self.hardware = filefp.readline().strip()
        line_num += 1

        # Parsing the data
        for line in filefp:
            if line.startswith('logs enabled:'):
                break
            line_num += 1
            tokens = [x.strip() for x in line.split(',')]
            try:
                if tokens[0] == 'FMT':
                    self._set_format(tokens)
                elif tokens[0] == 'PARM':
                    self._set_parm(tokens)
                else:
                    self._set_data(tokens)
            except APlogError as err:
                raise APlogError('Input error on line %d: %s\nLog data may be overwritten by the next log.' % (line_num, err))
            except:
                print("Unexpected error:", sys.exc_info()[0])
                print('Input error on line %d \n %s' % (line_num, line))
                raise

    def save(self, filefp):
        pickle.dump((self.data, self.formats, self.parms, (self.firmware, self.free_RAM, self.hardware)), filefp)

    # TODO: Raise exception on errors
    def load(self, filefp):
        (self.data, self.formats, self.parms, (self.firmware, self.free_RAM, self.hardware)) = pickle.load(filefp)

    # e.g. FMT, 130, 35, GPS, BIBcLLeeEe, Status,Time,NSats,HDop,Lat,Lng,RelAlt,Alt,Spd,GCrs
    def _set_format(self, tokens):
        # Notes: the data types ([4]) are not stored.
        self.formats[tokens[3]] = tokens[5:]
        self.formats_datatypes[tokens[3]] = tokens[4]

    # e.g. PARM, FS_GPS_ENABLE, 1.000000
    def _set_parm(self, tokens):
        self.parms[tokens[1]] = tokens[2]

    # IMU, 0.007462, -0.008702, 0.008119, -0.201464, -0.324597, -10.050831
    def _set_data(self, tokens):
        group = tokens[0]
        if group in self.formats:
            if group == 'ERR' or group == 'CMD' or group == 'MODE':
                # These are events which are not updated frequently.
                # They will be stored and get written when a frequently updated
                # group is encountered.
                self.parm_events[group] = tokens
            elif group not in self.data:
                z = zip(self.formats[group], [[x] for x in tokens[1:]])
                self.data[group] = dict(z)
            else:
                for (k,v) in zip(self.formats[group], tokens[1:]):
                    self.data[group][k].append(v)

            if group == 'ATT':
                # Write an entry for the events to keep them in sync
                for egroup in self.parm_events.keys():
                    if egroup not in self.data.keys():
                        z = zip(self.formats[egroup], [[x] for x in self.parm_events[egroup][1:]])
                        self.data[egroup] = dict(z)
                    else:
                        for (k,v) in zip(self.formats[egroup], self.parm_events[egroup][1:]):
                            self.data[egroup][k].append(v)

        else:
            raise APlogError('Unknown data group %s' % group)


if __name__ == '__main__':
    """Testing logdata.py
    """
    import sys
    import matplotlib.pyplot as plt

    if (len(sys.argv) > 1):
        logfile = sys.argv[1]
    else:
        logfile = 'test.log'

    print('logfile = %s' % logfile)
    filefp = open(logfile)
    data = APlogData()
    data.parse(filefp)
    # No need to convert to float, plot can take strings
    # plt.plot([float(x) for x in data['IMU']['AccZ']])
    plt.plot(data['IMU']['AccZ'])
    plt.show()


# vim:set sw=4 softtabstop=4 et syntax=python filetype=python:
