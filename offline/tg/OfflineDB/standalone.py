
from offlinedb.model import init_model
from env.base.private import Private
p = Private()
from sqlalchemy import create_engine
engine = create_engine( p('DATABASE_URL') )
init_model(engine)


from offlinedb.model.dbi import dbi_ 

sps = dbi_.pair('SimPmtSpec')
print sps

## use composite pk to get a row
row = sps.query.get((1,100))
assert row.ROW_COUNTER == 100 and row.SEQNO == 1


print row



