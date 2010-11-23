"""
   http://matplotlib.sourceforge.net/examples/pylab_examples/date_demo1.html
   http://matplotlib.sourceforge.net/examples/pylab_examples/date_demo2.html


"""
from pylab import figure, show

import  matplotlib.dates as mpld
import datetime
from env.mysql_np import DB

a = DB()("select * from CalibFeeSpecVld") 

start   = a['TIMESTART'].astype(datetime.datetime)
end     = a['TIMEEND'].astype(datetime.datetime)
insert  = a['INSERTDATE'].astype(datetime.datetime)
version = a['VERSIONDATE'].astype(datetime.datetime)

vals  = a['SEQNO']

fig = figure()
ax = fig.add_subplot(111)
ax.grid(True)

ax.plot_date( mpld.date2num(insert), mpld.date2num(start) ,   '-' )
ax.plot_date( mpld.date2num(insert), mpld.date2num(end) ,     '-' )
ax.plot_date( mpld.date2num(insert), mpld.date2num(version) , '-' )

#ax.plot_date( mpld.date2num(start),  vals , '-' )
#ax.plot_date( mpld.date2num(end),    vals , '-' )
#ax.plot_date( mpld.date2num(version), vals , '-' )

years    = mpld.YearLocator()   # every year
months   = mpld.MonthLocator()  # every month
monthsFmt = mpld.DateFormatter('%Y-%m')

mondays   = mpld.WeekdayLocator(mpld.MONDAY)

# format the ticks
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.xaxis.set_minor_locator(mondays)
ax.xaxis.grid(False,'major')
ax.xaxis.grid(True,'minor')

ax.yaxis.set_major_locator(months)
ax.yaxis.set_major_formatter(monthsFmt)
ax.yaxis.set_minor_locator(mondays)
ax.yaxis.grid(False,'major')
ax.yaxis.grid(True,'minor')


ax.fmt_xdata = mpld.DateFormatter('%Y-%m-%d')
ax.fmt_ydata = mpld.DateFormatter('%Y-%m-%d')

ax.autoscale_view()



fig.autofmt_xdate()
show()

