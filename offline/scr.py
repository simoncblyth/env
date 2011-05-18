"""
dcs tables have nasty habit of encoding content (stuff that should be in rows) into 
table and column names 


Scraper code not matching current offline_db table


mysql> describe DcsPmtHv  ;
+-------------+--------------+------+-----+---------+----------------+
| Field       | Type         | Null | Key | Default | Extra          |
+-------------+--------------+------+-----+---------+----------------+
| SEQNO       | int(11)      | NO   | PRI |         |                | 
| ROW_COUNTER | int(11)      | NO   | PRI | NULL    | auto_increment | 
| ladder      | tinyint(4)   | YES  |     | NULL    |                | 
| col         | tinyint(4)   | YES  |     | NULL    |                | 
| ring        | tinyint(4)   | YES  |     | NULL    |                | 
| voltage     | decimal(6,2) | YES  |     | NULL    |                | 
| pw          | tinyint(4)   | YES  |     | NULL    |                | 
+-------------+--------------+------+-----+---------+----------------+
7 rows in set (0.14 sec)



"""
import os
from sqlalchemy import MetaData, create_engine


def sa_url( sect , path="~/.my.cnf" ):
    """
    Provide SQLAlchemy URL for the config file section `sect`
    """ 
    from ConfigParser import ConfigParser
    cfp = ConfigParser()
    cfp.read( map(lambda _:os.path.expanduser(_), [path] ))
    cfg = dict(cfp.items(sect))
    return "mysql://%(user)s:%(password)s@%(host)s/%(database)s" % cfg

def sa_meta( url ):
    """
    Reflect schema of all tables in database at `url`

    How to handle connections to multiple DB ? 
        http://www.sqlalchemy.org/docs/07/core/schema.html?highlight=multiple%20metadata#binding-metadata-to-an-engine-or-connection

    adopt simple approach of binding the engine to the metadata
    """      
    m = MetaData()
    e = create_engine( url , echo=True )
    m.reflect(bind=e)       
    return m


m_dcs = sa_meta(sa_url("dcs"))
m_off = sa_meta(sa_url("tmp_offline_db"))

print "\n".join([t.name for t in m_dcs.sorted_tables])
print "\n".join([t.name for t in m_off.sorted_tables])


## NB non-uniform table naming
pairs = [('DBNS_AD1_HV','DBNS_AD1_HVPw',),('DBNS_AD2_HV','DBNS_AD2_HV_Pw',)]  

from sqlalchemy.sql import join
from sqlalchemy.orm import mapper

for hv_,pw_ in pairs:
    hv,pw = m_dcs.tables[hv_],m_dcs.tables[pw_]
    j = join( hv, pw , hv.c.id == pw.c.id )   ## no FK so must specify the onclause
    #mapper(AD, j )



## accessing column objects
#print t.c.id
#print t.c.date_time







