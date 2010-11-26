"""
  For interactive testing of specific figs use the below __main__ and invoke ...
      [blyth@cms01 db]$ ipython -pylab figs.py
      
   No worries concerning the warning about matplotlib.use('Agg') doing nothing 
   ... for interactive use you want TkAgg anyhow.

"""
import matplotlib 
matplotlib.use('Agg')      #  non-interactive/web server usage
#matplotlib.use('GTkAgg')  # interactive plot showing etc... 
import matplotlib.pyplot as plt

from env.offline.dbn import DBn


from env.mysql_np import DB
from env.dyb.db import Qry


def demo_fig():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([1,2,3])
    return fig

def column_fig( dbname, tabname, colname ):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    db = DBn( dbname )
    db.tab = tabname
    db()
    nd = db.numpy 

    col = nd[colname]

    ## for string entries ... could add bar graph which frequency of each string 
    if col.dtype.kind == 'S':
        pass 
    else:
        ax.hist( col )
    return fig


def column_fig2( dbname, tabname, colname ):
    fig = plt.figure()
    ax = fig.add_subplot(111)


    q = Qry(tabname)
    db = DB(dbname)
    #a = db(q.sql) 
     




if __name__ == '__main__':
    pass
    #fig = column_fig( "prior","CalibFeeSpec", "SEQNO" )
    #fig = column_fig( "prior","SimPmtSpec", "PMTDESCRIB" )
    #fig = column_fig( "prior","SimPmtSpec", "PMTSIGMAG" )
    #fig.show()

    db = DBn("prior", tab="SimPmtSpec")()  
    nd = db.numpy 






