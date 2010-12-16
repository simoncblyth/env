
from env.mysql_numpy import DB
db = DB()
a = db("select voltage from DcsPmtHv where ladder=2 and col=2 and ring=2 and voltage is not null order by seqno")

print a


