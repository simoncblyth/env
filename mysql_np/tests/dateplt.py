"""
   http://matplotlib.sourceforge.net/api/dates_api.html


"""

import datetime
from env.mysql_np import DB

db = DB()
a = db("select * from  CalibFeeSpecVld limit 10")

opens = a["SEQNO"]
dates = a["TIMESTART"].astype(datetime.datetime)

from pylab import figure, show
import datetime

from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
years    = YearLocator()   # every year
months   = MonthLocator()  # every month
yearsFmt = DateFormatter('%Y')

fig = figure()
ax = fig.add_subplot(111)
ax.plot_date(dates, opens, '-')

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)
ax.autoscale_view()
ax.grid(True)

# format the coords message box
def price(x): return '$%1.2f'%x
ax.fmt_xdata = DateFormatter('%Y-%m-%d')
ax.fmt_ydata = price

fig.autofmt_xdate()

show()



