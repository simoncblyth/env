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

from env.mysql_numpy import DB
db = DB()
#a = db("select voltage from DcsPmtHv where ladder=2 and col=2 and ring=2 and voltage is not null order by seqno")
#a = db("select voltage from DcsPmtHv where voltage is not null order by seqno limit 192000")
#a = db("select voltage from DcsPmtHv where voltage is not null order by seqno limit 192000")

n = 1000
a = db("select SEQNO, ROW_COUNTER, ladder, col, ring, coalesce(voltage, -10), coalesce(pw, -10) from DcsPmtHv where SEQNO <= %s order by SEQNO, ROW_COUNTER" % n )
b = a.reshape( (n,-1) )
print b


