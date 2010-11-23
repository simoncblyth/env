"""
 
   Visualizing DBI dates using numpy 2.0... + matplotlib 1.0.0

   http://matplotlib.sourceforge.net/api/dates_api.html

   Because sometimes/always NULL : 
          coalesce(INSERTDATE,TIMESTART) as INSERTDATE 
      
   but coalesce result type coming in as strings so convert ...
          convert(coalesce(INSERTDATE,TIMESTART),DATETIME) as INSERTDATE

"""

from datetime import datetime, timedelta 
from env.mysql_np import DB
from pylab import figure, show
import matplotlib.dates as mpld

db = DB()
t = db("show tables")
for _ in t:    
    print _[0]

a = db("select SEQNO,TIMESTART,TIMEEND,convert(coalesce(INSERTDATE,TIMESTART),DATETIME) as INSERTDATE,convert(coalesce(VERSIONDATE,TIMESTART),DATETIME) as VERSIONDATE from DaqRunInfoVld")

seqno   = a["SEQNO"]
start   = a["TIMESTART"].astype(datetime)
end     = a["TIMEEND"].astype(datetime)
insert  = a["INSERTDATE"].astype(datetime)
version = a["VERSIONDATE"].astype(datetime)

fig = figure()
ax = fig.add_subplot(111)

# use of timedelta with elementwise numpy array behavior
ax.plot_date( start                        , seqno , 'ro')
ax.plot_date( end     + timedelta(days=5)  , seqno , 'go')
ax.plot_date( insert  - timedelta(days=5)  , seqno , 'bo')
ax.plot_date( version + timedelta(days=10) , seqno , 'r^')

locator = mpld.AutoDateLocator()
formatter = mpld.AutoDateFormatter(locator)

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

#ax.yaxis.set_major_locator(locator)
#ax.yaxis.set_major_formatter(formatter)

ax.autoscale_view()
ax.grid(True)
fig.autofmt_xdate()

show()



