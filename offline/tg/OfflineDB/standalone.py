
from offlinedb.model import init_model
from env.base.private import Private
p = Private()
from sqlalchemy import create_engine
engine = create_engine( p('DATABASE_URL') )
init_model(engine)
from offlinedb.model.dbi import soup

print soup.entity('SimPmtSpec')

