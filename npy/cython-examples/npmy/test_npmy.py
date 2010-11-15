"""
mysql> select * from DcsPmtHv limit 10 ;
+-------+-------------+--------+------+------+---------+------+
| SEQNO | ROW_COUNTER | ladder | col  | ring | voltage | pw   |
+-------+-------------+--------+------+------+---------+------+
|     1 |           1 |      0 |    2 |    6 | 2256.32 |    1 | 
|     1 |           2 |      1 |    0 |    3 | 2255.87 |    1 | 
|     1 |           3 |      0 |    2 |    4 | 2267.80 |    1 | 
|     1 |           4 |      1 |    1 |    0 | 2302.50 |    1 | 

mysql> select distinct(pw) from DcsPmtHv  ;
+------+
| pw   |
+------+
|    1 | 
|    0 | 
| NULL | 
+------+
3 rows in set (28.72 sec)

mysql> select distinct(ladder) from DcsPmtHv  ;
+--------+
| ladder |
+--------+
|      0 | 
|      1 | 
|      7 | 
|      5 | 
|      4 | 
|      2 | 
|      3 | 
|      6 | 
|      8 | 
+--------+
9 rows in set (11.15 sec)


mysql> select distinct(col) from DcsPmtHv  ;
+------+
| col  |
+------+
|    2 | 
|    0 | 
|    1 | 
|    3 | 
+------+
4 rows in set (11.25 sec)


mysql> select distinct(ring) from DcsPmtHv  ;
+------+
| ring |
+------+
|    6 | 
|    3 | 
|    4 | 
|    0 | 
|    2 | 
|    1 | 
|    5 | 
|    7 | 
|    8 | 
+------+
9 rows in set (11.20 sec)


mysql> select count(*) from DcsPmtHv where voltage is null ;
+----------+
| count(*) |
+----------+
|  3857113 | 
+----------+
1 row in set (2.84 sec)

mysql> select count(*) from DcsPmtHv where pw is null ;
+----------+
| count(*) |
+----------+
|  4963815 | 
+----------+
1 row in set (2.86 sec)


mysql> select count(*) from DcsPmtHv where ROW_COUNTER is null ;
+----------+
| count(*) |
+----------+
|        0 | 
+----------+
1 row in set (0.00 sec)

mysql> select count(*) from DcsPmtHv where SEQNO is null ;
+----------+
| count(*) |
+----------+
|        0 | 
+----------+
1 row in set (0.00 sec)



"""

import _mysql
conn = _mysql.connect( read_default_file="~/.my.cnf", read_default_group="client" )

import numpy as np


## tinyints could be used for ladder/col/ring ... but leave at i4 initially 
## voltage and pw go null quite a lot ... avoid for now by "not null" exclusion  
dt = np.dtype([('SEQNO', 'i4'), ('ROW_COUNTER', 'i4'), ('ladder', 'i4'),('col','i4'),('ring','i4'),('voltage','f4'),('pw','i4')])

sql = "select %(cols)s from %(tab)s limit %(limit)s " % { 'cols':",".join(dt.names), 'tab':"DcsPmtHv", 'limit':10 }

conn.query( sql )

# store_result  : results stored in client
# use_result    : results stay on server, until fetched ..... for HUGE resultsets
#result = conn.store_result()    ... this is 1.2.2 API

result = conn.get_result()
n = result.num_rows()   ##  NB will not be valid for "use_result" until all rows are fetched
print n

from npmy import look, fetch_rows_into_array
look(result)

qa = np.zeros(10, dt)
print qa
fetch_rows_into_array( result, qa )
print qa


