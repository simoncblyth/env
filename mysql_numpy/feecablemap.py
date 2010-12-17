"""

In [1]: a
Out[1]: 
rec.array([ (1, 1, 16843009, 'DayaBayAD1-board01-connector01', 67109121, 'Fee-1', 16843009, 'DayaBayAD1-ring01-column01', 16777217, 'Pmt8inch-1'),
       (1, 2, 16843010, 'DayaBayAD1-board01-connector02', 67109122, 'Fee-1', 16843010, 'DayaBayAD1-ring01-column02', 16777218, 'Pmt8inch-2'),
       (1, 3, 16843011, 'DayaBayAD1-board01-connector03', 67109123, 'Fee-1', 16843011, 'DayaBayAD1-ring01-column03', 16777219, 'Pmt8inch-3'),
       ...,
       (3, 414, 537006350, 'SABAD2-board17-connector14', 0, 'NULL', 537001998, 'SABAD2-ring00-column14', 0, 'NULL'),
       (3, 415, 537006351, 'SABAD2-board17-connector15', 0, 'NULL', 537001999, 'SABAD2-ring00-column15', 0, 'NULL'),
       (3, 416, 537006352, 'SABAD2-board17-connector16', 0, 'NULL', 537002000, 'SABAD2-ring00-column16', 0, 'NULL')], 
      dtype=[('SEQNO', '>i4'), ('ROW_COUNTER', '>i4'), ('FEECHANNELID', '>i4'), ('FEECHANNELDESC', '|S30'), ('FEEHARDWAREID', '>i4'), ('CHANHRDWDESC', '|S30'), ('SENSORID', '>i4'), ('SENSORDESC', '|S30'), ('PMTHARDWAREID', '>i4'), ('PMTHRDWDESC', '|S30')])

In [2]: a.SEQNO
Out[2]: array([1, 1, 1, ..., 3, 3, 3])

In [3]: a.ROW_COUNTER
Out[3]: array([  1,   2,   3, ..., 414, 415, 416])

In [4]: a.ROW_COUNTER.max()
Out[4]: 2656

In [5]: a.FEECHANNELID
Out[5]: 
array([ 16843009,  16843010,  16843011, ..., 537006350, 537006351,
       537006352])

In [6]: a.FEECHANNELDESC
Out[6]: 
chararray(['DayaBayAD1-board01-connector01', 'DayaBayAD1-board01-connector02',
       'DayaBayAD1-board01-connector03', ..., 'SABAD2-board17-connector14',
       'SABAD2-board17-connector15', 'SABAD2-board17-connector16'], 
      dtype='|S30')

In [7]: a.FEEHARDWAREID
Out[7]: array([67109121, 67109122, 67109123, ...,        0,        0,        0])

In [8]: a.SENSORID
Out[8]: 
array([ 16843009,  16843010,  16843011, ..., 537001998, 537001999,
       537002000])

In [9]: a.SENSORDESC
Out[9]: 
chararray(['DayaBayAD1-ring01-column01', 'DayaBayAD1-ring01-column02',
       'DayaBayAD1-ring01-column03', ..., 'SABAD2-ring00-column14',
       'SABAD2-ring00-column15', 'SABAD2-ring00-column16'], 
      dtype='|S30')

In [10]: a.PMTHARDWAREID
Out[10]: array([16777217, 16777218, 16777219, ...,        0,        0,        0])


"""
import matplotlib.pyplot as plt

import numpy as np
from env.mysql_numpy import DB
db = DB()
a_ = db("select * from FeeCableMap")
a = a_.view(np.recarray)


## for interactive plotting need matplotlib and  use :    ipython feecablemap.py -pylab   
## see help(pylab)  
plt.plot(  a.SEQNO*3000 + a.ROW_COUNTER ,  a.PMTHARDWAREID , "o" )


