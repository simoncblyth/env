"""
  For interactive testing of specific figs use the below __main__ and invoke ...
      [blyth@cms01 db]$ ipython -pylab figs.py
      
   No worries concerning the warning about matplotlib.use('Agg') doing nothing 
   ... for interactive use you want TkAgg anyhow.

"""
import matplotlib 
#matplotlib.use('Agg')      #  non-interactive/web server usage
matplotlib.use('GTkAgg')  # interactive plot showing etc... 
import matplotlib.pyplot as plt

#from env.offline.dbn import DBn

#from env.mysql_np import SlowDB as DB
from env.mysql_np import DB     ## this relies on patch to 1.2.3 or use of un-released 1.3.0

from env.dyb.db import Qry


def demo_fig():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([1,2,3])
    return fig

def one_column_hist( dbname, tabname, colname ):

    db = DB(dbname)
    q = Qry(tabname)
    a = db(str(q))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist( a[colname] )

    return fig


def multi_column_hist( dbname, tabname ):
    """
                   
    """

    db = DB(dbname)
    q = Qry(tabname)
    a = db(str(q))
    cols = a.dtype.names
    
    if len(cols)<2:
        n = 1 
    if len(cols)<5:
        n = 2
    elif len(cols)<10:
        n = 3 
    elif len(cols)<17:
        n = 4
    else:
        n = 5

    fig = plt.figure()
    for i, colname in enumerate(cols):
        ax = fig.add_subplot(n,n,i + 1)
        col = a[colname]
        if col.dtype.kind == 'S':    ## for string entries ... could add bar graph which frequency of each string 
            pass
        else: 
            ax.hist( col )
        #plt.title( colname )
        plt.xlabel( colname )

    return fig


def make_fig( **kwargs ):
    """
         TODO:
             dispatching in Fig classes rather than here
    """
    dbname = kwargs.get('dbname','client' )   
    tabname = kwargs.get('tabname', None )
    colname = kwargs.get('colname', None )
    type = kwargs.get('type', None )

    if tabname and colname:
        return one_column_hist( dbname, tabname, colname )
    elif tabname:
        return multi_column_hist( dbname, tabname )
    else:
        return demo_fig()
 
     




if __name__ == '__main__':
    pass


    fig = make_fig("client","CalibPmtSpec","SEQNO")
    fig.show()




