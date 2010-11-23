#!/usr/bin/python
"""
   http://stackoverflow.com/questions/2207670/date-versus-time-interval-plotting-in-matplotlib

"""

import datetime
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# dates for xaxis
from env.mysql_np import DB
db = DB()
a = db("select * from  CalibFeeSpecVld limit 10")


# event times
event_date = a["INSERTDATE"].astype(datetime.datetime) 
event_start  = a["TIMESTART"].astype(datetime.datetime)
event_finish = a["TIMEEND"].astype(datetime.datetime)

# translate times and dates lists into matplotlib date format numpy arrays
start = np.fromiter((mdates.date2num(event) for event in event_start), dtype = 'float', count = len(event_start))
finish = np.fromiter((mdates.date2num(event) for event in event_finish), dtype = 'float', count = len(event_finish))
date = mdates.date2num(event_date)

# calculate events durations
duration = finish - start

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# use errorbar to represent event duration
ax.errorbar(date, start, [np.zeros(len(duration)), duration], linestyle = '-')
# make matplotlib treat both axis as times
ax.xaxis_date()
ax.yaxis_date()

fig.autofmt_xdate()
fig.show()
