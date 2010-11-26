"""
   Visualizing DBI dates using numpy 2.0... + matplotlib 1.0.0
   http://matplotlib.sourceforge.net/api/dates_api.html

      python $(env-home)/mysql_np/tests/dateplt_auto.py DaqRunInfoVld


"""

from datetime import datetime, timedelta 

import matplotlib.pyplot as plt
import matplotlib.dates as mpld

from env.mysql_np import DB
from env.dyb.db import Qry


def vld_( ax ,  a ):
    seqno   = a["SEQNO"]
    start   = a["TIMESTART"].astype(datetime)
    end     = a["TIMEEND"].astype(datetime)
    insert  = a["INSERTDATE"].astype(datetime)
    version = a["VERSIONDATE"].astype(datetime)

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



def fig_matrix():
    """
        presents Vld plots of all tables  
    """
    db = DB()
    ts = db("show tables")

    vld = filter( lambda _:_[0].upper().endswith('VLD'), ts  )
    vxs = np.array( vld ).reshape( 3, -1 )

    fig, axs = plt.subplots( vxs.shape[0] , vxs.shape[1] , sharex=False, sharey=False)
    for ax,vx in zip(axs.flat, vxs.flat):
        q = Qry( vx[0] , limit=10)
        a = db(str(q))
        vld_( ax , a)

    fig.autofmt_xdate()
    


if __name__ == '__main__':
    pass
    import sys
    vx = len(sys.argv) > 1 and sys.argv[1] or "CalibPmtSpecVld"
   
    fig, ax = plt.subplots( 1 , 1 , sharex=False, sharey=False)
    plt.title(vx) 

    db = DB()
    q = Qry( vx , limit=-1 )

    print str(q) 
    a = db(str(q))
    vld_( ax , a)

    fig.autofmt_xdate()


    plt.show()   ## when not using ipython, need this to start mainloop



