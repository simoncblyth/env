"""
        sqlalchemy - numpy - matplotlib - django
           MySQLdb - numpy - matplotlib - django 

    http://matplotlib.sourceforge.net/ 

    http://matplotlib.sourceforge.net/faq/howto_faq.html#matplotlib-in-a-web-application-server

    http://matplotlib.sourceforge.net/users/artists.html#figure-container 
        line drawing example can be the basis of the validity viz 
          help(matplotlib.dates) 
          help(matplotlib.ticker)

    http://code.creativecommons.org/svnroot/stats/reports/temp/date_demo.py
        sqlalchemy with numpy/matplotlib

     http://www.sqlalchemy.org/trac/ticket/1572
          suggestion to subclass SA query to provide numpy arrays
     
    https://github.com/dalloliogm/sqlalchemy-recarray
          only a stub, no implementation 

"""
import matplotlib 
matplotlib.use('Agg')    # Agg      non-interactive/web server usage
#matplotlib.use('TkAgg')  # TkAgg    interactive plot showing etc... 
import matplotlib.pyplot as plt

from env.offline.dbn import DBn


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

    ax.hist( nd[colname] )
    return fig


if __name__ == '__main__':
    pass
    fig = column_fig( "prior","CalibFeeSpec", "SEQNO" )
    fig.show()




