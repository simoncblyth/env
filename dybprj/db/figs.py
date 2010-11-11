"""
  For interactive testing of specific figs use the below __main__ and invoke ...
      [blyth@cms01 db]$ ipython -pylab figs.py
      
   No worries concerning the warning about matplotlib.use('Agg') doing nothing 
   ... for interactive use you want TkAgg anyhow.


   Access the numpy arrays like below, remember to call with the () ... which does the DB query 

         db = DBn("prior", tab="SimPmtSpec" )()
         nd = db.numpy 

   switch tables by assignment :
         db.tab = "CalibPmtSpecVld"


   read from csv into numpy arrays ...


   import numpy as np

In [45]: c = np.genfromtxt("/tmp/cat/offline_db/CalibPmtSpec/CalibPmtSpec.csv", skip_header=1 ,delimiter=",",dtype=None)

In [46]: c[0]
Out[46]: (17, 1, 536936705, '"SABAD1-ring01-column01"', 1, 19.521999999999998, 0.0, 1.952, -6.3140000000000001, 0, 0.79300000000000004, 0, 0, 0)

In [47]: c
Out[47]: 
array([ (17, 1, 536936705, '"SABAD1-ring01-column01"', 1, 19.521999999999998, 0.0, 1.952, -6.3140000000000001, 0, 0.79300000000000004, 0, 0, 0),
       (17, 2, 536936961, '"SABAD1-ring02-column01"', 1, 17.565999999999999, 0.0, 1.7569999999999999, -4.3449999999999998, 0, 0.84099999999999997, 0, 0, 0),
       (17, 3, 536937217, '"SABAD1-ring03-column01"', 1, 18.324999999999999, 0.0, 1.833, -5.4020000000000001, 0, 0.876, 0, 0, 0),
       ...,
       (25, 414, 537001998, '"SABAD2-ring00-column14"', 1, 20.0, 0.0, 2.0, 0.0, 0, 1.0, 0, 0, 0),
       (25, 415, 537001999, '"SABAD2-ring00-column15"', 1, 20.0, 0.0, 2.0, 0.0, 0, 1.0, 0, 0, 0),
       (25, 416, 537002000, '"SABAD2-ring00-column16"', 1, 20.0, 0.0, 2.0, 0.0, 0, 1.0, 0, 0, 0)], 
      dtype=[('f0', '<i4'), ('f1', '<i4'), ('f2', '<i4'), ('f3', '|S24'), ('f4', '<i4'), ('f5', '<f8'), ('f6', '<f8'), ('f7', '<f8'), ('f8', '<f8'), ('f9', '<i4'), ('f10', '<f8'), ('f11', '<i4'), ('f12', '<i4'), ('f13', '<i4')])








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

    col = nd[colname]

    ## for string entries ... could add bar graph which frequency of each string 
    if col.dtype.kind == 'S':
        pass 
    else:
        ax.hist( col )
    return fig


if __name__ == '__main__':
    pass
    #fig = column_fig( "prior","CalibFeeSpec", "SEQNO" )
    #fig = column_fig( "prior","SimPmtSpec", "PMTDESCRIB" )
    #fig = column_fig( "prior","SimPmtSpec", "PMTSIGMAG" )
    #fig.show()

    db = DBn("prior", tab="SimPmtSpec")()  
    nd = db.numpy 







