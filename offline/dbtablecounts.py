import os
import re
from env.structure import Persdict
import os
from mysqldb import DB
from dbconf import DBConf
import re


class DBTableCounts(Persdict):

    _dbg = 2 
    _patn = re.compile("(?P<sect>\S*)_(?P<stamp>\d*)")
    def _id( cls, *args, **kwa ):
        return "%s_%s" % ( kwa.get('sect','nosect'), kwa.get('stamp','nostamp') )
    _id = classmethod( _id )

    def __init__(self, sect=None, stamp=None ):
        print "(client)%s.__init__ called ... %s  " % ( self.__class__.__name__ , self )

        ini = os.path.expanduser("~/.dybdb.ini")
        cfg = DBConf( path=ini , sect=sect )  
        db = DB( **cfg )
        self.table_counts( db )

        db.close()
        
        DBTableCounts._save( self, sect=sect , stamp=stamp )
        print "(client)%s.__init__ done ... %s  " % ( self.__class__.__name__ , self )
        

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
 
