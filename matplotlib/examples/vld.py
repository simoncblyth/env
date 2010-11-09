"""

   Need good way of loading numpy structures from DB queries ...
      google:"numpy database"

          http://mail.scipy.org/pipermail/numpy-discussion/2010-September/052599.html


      http://stackoverflow.com/questions/762750/defining-a-table-with-sqlalchemy-with-a-mysql-unix-timestamp



"""

from datetime import datetime
from env.sa import Session, DBISOUP
from sqlalchemy import select, func

tabname, limit = 'CalibFeeSpecVld', 25

kls = DBISOUP.get(tabname, None)
tc = kls._table.c

count,seqmin,seqmax,tmin,tmax = select([func.count(),func.min(tc.SEQNO),func.max(tc.SEQNO),func.min(tc.TIMESTART),func.max(tc.TIMEEND),]).where(tc.TIMESTART>datetime(2000, 1, 1, 0, 0, 0)).execute().fetchone()

session = Session()
objs = session.query(kls).order_by(kls.SEQNO).all()[0:limit]

for obj in objs:
    print obj.TIMESTART


rp = select( [tc.SEQNO,tc.TIMESTART,tc.TIMEEND] ).execute()






from pylab import figure, show
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
years    = YearLocator()   # every year
months   = MonthLocator()  # every month
yearsFmt = DateFormatter('%Y')

dates = [row[1] for row in rp]

fig = figure()
ax = fig.add_subplot(111)
ax.plot_date( dates, opens, '-')
#
## format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)
ax.autoscale_view()
ax.grid(True)

fig.autofmt_xdate()
show()
