"""
   http://www.sqlalchemy.org/docs/orm/session.html#vertical-partitioning

  



"""

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import scoped_session, sessionmaker
from env.offline.dbconf import DBConf

mydb,dbi = 'mydb','prior'

ENGINES = dict((cnf, create_engine(DBConf(cnf).sqlalchemy, echo=True)) for cnf in (mydb,dbi))

Session = scoped_session(sessionmaker(autoflush=False, expire_on_commit=False, autocommit=True))  ## needed for the soup ?
__all__ = "Session ENGINES get_or_create".split()


## declarative base tables 
from models import Qry, Base, get_or_create
Base.metadata.create_all( ENGINES[mydb] )   # create tables if needed
decls = Base.metadata.tables.values()

binds = {}
binds.update(dict((table, ENGINES[mydb]) for table in decls))
__all__ += map(lambda u:str(u),decls) 


## soup tables
from sqlalchemy.engine.reflection import Inspector
insp = Inspector.from_engine(ENGINES[dbi])

soupls = insp.get_table_names()
binds.update(dict((table, ENGINES[dbi]) for table in soupls))
__all__ += map(lambda u:str(u),soupls) 


## table/engine partitioning 
Session.configure(binds=binds)

from sqlalchemy.ext.sqlsoup import SqlSoup
meta = MetaData( ENGINES[dbi] )
soup = SqlSoup( meta , session = Session )   # introspect DB and mapem 
from dbi import DbiSoup 
dbisoup = DbiSoup(soup)
locals().update(dbisoup)





if __name__ == '__main__':
    session = Session()
    print session.query(Qry).count()
    assert SimPmtSpec.count() == 2546 
    assert CalibPmtSpec.count() == 4160 
    from datetime import datetime
    assert SimPmtSpec.get((1,100)).VERSIONDATE == datetime(2010, 1, 20, 0, 0)   ## CPK get 
    assert CalibFeeSpecVld.count() == 111
    for v in CalibPmtSpecVld.all():
        print v 


