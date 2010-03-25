import os
from mysqldb import DB
from dbconf import DBConf
from datetime import datetime
import re
from env.structure import Pers

class DBTableCounts(Pers):

    _patn = re.compile("(?P<sect>\S*)_(?P<stamp>\d*)")
    def _id( cls, *args, **kwa ):
        return "%s_%s" % ( kwa.get('sect','nosect'), kwa.get('stamp','nostamp') )
    _id = classmethod( _id )

    def _parse(cls, name ):
        m = cls._patn.match( name )
        if m:
            return m.groupdict()
        return None   
    _parse = classmethod( _parse )    

    def __init__(self, sect=None, stamp=None ):
        ini = os.path.expanduser("~/.dybdb.ini")
        cfg = DBConf( path=ini , sect=sect )  
        db = DB( **cfg )
        self.table_counts( db )
        db.close()

    def _instances(cls):        
        for iname in os.listdir( DBTableCounts._dir() ):
           d = cls._parse( iname )
           if d:
               yield cls( **d )
    _instances = classmethod( _instances )

    def table_counts(self, db ):
        rec = db.fetchone("SELECT VERSION()")  ## check connection with query that should always succeed
        for rec in db("SHOW TABLES"):
            tab = rec.values()[0]
            cnt = db.fetchone("SELECT COUNT(*) FROM  %s" % tab )
            n = cnt.values()[0]
            self[tab] = n
 
if __name__=='__main__':

    sect = os.environ.get("DB_SECT","local")
    stamp = datetime.strftime( datetime.now() , "%Y%m%d" )
    
    ## create a new instance for today if not existing already 
    dbtc = DBTableCounts( sect=sect, stamp=stamp )
    print dbtc

    ## over the persisted instances
    for i in DBTableCounts._instances():
        print i


    



