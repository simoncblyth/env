"""
Generic Scraping Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~

Goals
  * eliminate duplication in scraper code
  * reduce scraper code to managable levels, that do not scale with the tables scraped
  
GOAL OF SA MAPPING IS NOT A WHOLE-IN-ONE, as need to use DybDbi on the destination side anyhow
  * source mapped instances are an intermediary 
  * ensemble intermediary needed ... for delta logic

MAYBE EASIER NOT TO USE SA MAPPING   

Boundary conditions
  * 2 databases : source and destination
  * sa mapping of source tables (or any selectable eg with join), to instances  
  * propagate such instances into DybDbi rows   
  * ``.spec``ify destination DBI table pairs  
     *  ? mapping specifics in the .spec ? 


dcs tables have nasty habit of encoding content (stuff that should be in rows) into table and column names 
this requires very non-standard mapping ... one row of source dcs table corresponds to many rows 
of destination table (actually ... need to map a subset ) 

  * http://www.sqlalchemy.org/docs/07/orm/mapper_config.html#sql-expressions-as-mapped-attributes
      combining columns into attributes
  * http://sqlalchemy.readthedocs.org/en/latest/orm/mapper_config.html#mapping-a-subset-of-table-columns  

   mapper(User, users_table, include_properties=['user_id', 'user_name'])

  * google:"sqlalchemy mapping multiple instances from single row"
  * http://stackoverflow.com/questions/1300433/how-to-map-one-class-against-multiple-tables-with-sqlalchemy


Are mapping a row to a collection of instances 
  * http://www.sqlalchemy.org/docs/orm/collections.html

  * http://stackoverflow.com/questions/780774/sqlalchemy-dictionary-of-tags
 

Note that scraper code is not matching current offline_db table

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

    Paraphrasing 
        http://www.sqlalchemy.org/docs/07/core/schema.html?highlight=multiple%20metadata#binding-metadata-to-an-engine-or-connection

    Application has multiple schemas that correspond to different engines. 
    Using one MetaData for each schema, bound to each engine, provides a decent place to delineate between the schemas. 
    The ORM will also integrate with this approach, where the Session will naturally use the engine that is bound to each 
    table via its metadata (provided the Session itself has no bind configured.).

    Adopt simple approach of binding the engine to the metadata
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
pair = pairs[0]

from sqlalchemy.sql import join
from sqlalchemy.orm import mapper

class ADHV(object):
    pass

hv_, pw_ = pair
hv, pw = m_dcs.tables[hv_],m_dcs.tables[pw_]
hp = join( hv, pw , hv.c.id == pw.c.id )   ## no FK so must specify the onclause

import re
kptn = re.compile("^L(?P<ladder>\d)C(?P<col>\d)R(?P<ring>\d)$")

for k in hv.c.keys():
    m = kptn.match(k)
    if m:
        print m.groupdict()
    #mapper(AD, hp , include_properties={ 'ladder':k[1],'ring':k )


"""
In [5]: hv.c.keys()
Out[5]: 
[u'id',
 u'date_time',
 u'L8C3R8',
 u'L8C3R7',
 u'L8C3R6',
 u'L8C3R5',
 u'L8C3R4',
 u'L8C3R3',
...


    



mapper complaining regards same named columns

/home/blyth/v/scr/lib/python2.4/site-packages/sqlalchemy/orm/mapper.py:585: SAWarning: Implicitly combining column DBNS_AD1_HV.id with column DBNS_AD1_HVPw.id under attribute 'id'.  This usage will be prohibited in 0.7.  Please configure one or more attributes for these same-named columns explicitly.
  setparent=True)


"""

## accessing column objects
#print t.c.id
#print t.c.date_time







