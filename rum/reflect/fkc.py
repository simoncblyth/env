import sys
from sqlalchemy import create_engine, MetaData
from sqlalchemy import Table, ForeignKeyConstraint

from env.base.private import Private
p = Private()
dburl = p('DATABASE_URL')  
engine = create_engine( dburl )

from sqlalchemy.orm import create_session
#session = create_session(bind=engine)

## see rumalchemy/test_repository.py 
from sqlalchemy.orm import scoped_session, sessionmaker
Session = scoped_session(sessionmaker(autocommit=False, autoflush=True))
Session.configure(bind=engine)

metadata = MetaData( engine )
metadata.reflect()   

## avoid unicode hassles
for name, table in metadata.tables.iteritems():
    table.name = str(name)
    metadata.tables[table.name] = metadata.tables.pop(name)


def dbi_fkc( metadata ):
    """
       Append FK constraints to the SA tables as MySQL dont know em 

       Problem here is that souping seems to ignore these and introspects 
       its own tables ?   Maybe need to plant in session 

    """
    from sqlalchemy import ForeignKeyConstraint
    pay_tables = [n[0:-3] for n,t in metadata.tables.items() if n.endswith('Vld')]
    vld_tables = ["%sVld" % n for n in pay_tables]    
    for p,v in zip(pay_tables,vld_tables): 
        pay = metadata.tables.get(p, None )
        vld = metadata.tables.get(v, None )
        if not(pay) or not(vld):
            print "skipping tables %s " % n
            continue
        pay.append_constraint( ForeignKeyConstraint( ['SEQNO'] , ['%s.SEQNO' % v ] ) )
    return pay_tables + vld_tables 


t = dbi_fkc(metadata)


#from sqlalchemy.ext.soup import SqlSoup
#
from rumalchemy.sqlsoup import SqlSoup
soup = SqlSoup( metadata , Session  )

sps = soup.entity('SimPmtSpec')
spv = soup.entity('SimPmtSpecVld')


