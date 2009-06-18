
from env.base.private import Private
p = Private()
from sqlalchemy import create_engine, MetaData
e = create_engine( p('DATABASE_URL') )
from sqlalchemy.ext.sqlsoup import SqlSoup
db = SqlSoup( MetaData(e) )
print db.SimPmtSpec.all()
