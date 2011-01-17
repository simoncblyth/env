"""
mysql> describe DcsPmtHv ;
+-------------+--------------+------+-----+---------+----------------+
| Field       | Type         | Null | Key | Default | Extra          |
+-------------+--------------+------+-----+---------+----------------+
| SEQNO       | int(11)      |      | PRI | 0       |                |
| ROW_COUNTER | int(11)      |      | PRI | NULL    | auto_increment |
| ladder      | tinyint(4)   | YES  |     | NULL    |                |
| col         | tinyint(4)   | YES  |     | NULL    |                |
| ring        | tinyint(4)   | YES  |     | NULL    |                |
| voltage     | decimal(6,2) | YES  |     | NULL    |                |
| pw          | tinyint(4)   | YES  |     | NULL    |                |
+-------------+--------------+------+-----+---------+----------------+
7 rows in set (0.04 sec)

"""
import numpy as np
from env.mysql_numpy import DB
db = DB()
#a = db("select voltage from DcsPmtHv where ladder=2 and col=2 and ring=2 and voltage is not null order by seqno")
#a = db("select voltage from DcsPmtHv where voltage is not null order by seqno limit 192000")
#a = db("select voltage from DcsPmtHv where voltage is not null order by seqno limit 192000")


## 1..94463      2009-12-09 10:43:33 ... 2010-11-15 18:16:38 
n = (1, 10001,)
nseq = n[1] - n[0] + 1

a = db("select SEQNO, ROW_COUNTER, ladder, col, ring, coalesce(voltage, -10) as voltage, coalesce(pw, -10) as pw from DcsPmtHv where %s <= SEQNO and SEQNO <= %s order by SEQNO, ROW_COUNTER" % n )
b = a.reshape((nseq,-1))                ## (192*nseq,) --> (nseq,192,)
d = np.absolute(np.diff( b['voltage'] , axis=0 ))    ## (nseq-1,192,)
x = np.max( d , axis=1 )    ## (nseq-1,)
l = len(x[x>0.25])


#c = np.transpose(b)        ## (nseq,192,) --> (192,nseq,)   make last axis seqno to allow diffing between seqno  
#d = np.diff(c['voltage'])  ## (192,nseq-1,) 

print b

def check_delta(thresh=0.25):
    """
        numpy operates elementwise 
        ... this provides the maximum absolute voltage shift in any of the 192 between 
        subsequent seqno

        only 259 of first 10000 seqno show a difference of any of the 192 voltages  > 0.25 V  
        ... and half of those are the first seqnos

    """
    j = 0 
    for i in range(0,len(b)-1):
        absd = np.absolute(b[i+1]['voltage'] - b[i]['voltage'])
        maxd = np.max(absd)
        if maxd>thresh:
            j += 1
            print j, i, maxd
           

check_delta()
