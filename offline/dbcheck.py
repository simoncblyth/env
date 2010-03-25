import os
from mysqldb import DB
from dbconf import DBConf
from datetime import datetime
import re
from env.structure import Persdict

class DBTableCounts(Persdict):

    _patn = re.compile("(?P<sect>\S*)_(?P<stamp>\d*)")
    def _id( cls, *args, **kwa ):
        return "%s_%s" % ( kwa.get('sect','nosect'), kwa.get('stamp','nostamp') )
    _id = classmethod( _id )

    def __init__(self, sect=None, stamp=None ):
        print "__init__ called  "
        ini = os.path.expanduser("~/.dybdb.ini")
        cfg = DBConf( path=ini , sect=sect )  
        db = DB( **cfg )
        self.table_counts( db )
        db.close()

    def table_counts(self, db ):
        """
            Populate self with table counts 
        """ 
        rec = db.fetchone("SELECT VERSION()")  
        for rec in db("SHOW TABLES"):
            tab = rec.values()[0]
            cnt = db.fetchone("SELECT COUNT(*) FROM  %s" % tab )
            n = cnt.values()[0]
            self[tab] = n
 

def today():
    """ create a new instance for today if not existing already  """
    sect = os.environ.get("DB_SECT","local")
    stamp = datetime.strftime( datetime.now() , "%Y%m%d" )
    
    dbtc = DBTableCounts( sect=sect, stamp=stamp )
    print dbtc


if __name__=='__main__':

    ## over the persisted instances
    for i in DBTableCounts._instances():
        print i


    



