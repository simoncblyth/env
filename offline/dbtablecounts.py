import os
from env.structure import Persdict
from mysqldb import DB

class DBTableCounts(Persdict):
    """
         Next comparing instances with diff reporting ...
           http://code.activestate.com/recipes/576644-diff-two-dictionaries/
    """
    _dbg = 0 
    def populate(self, *args, **kwa ):
        """
              Parameters other than 'stamp' are used for the DB connection
        """
        stamp = kwa.pop('stamp',None)  
        db = DB( **kwa )
        rec = db.fetchone("SELECT VERSION()")  
        for rec in db("SHOW TABLES"):
            tab = rec.values()[0]
            cnt = db.fetchone("SELECT COUNT(*) FROM  %s" % tab )
            n = cnt.values()[0]
            self[tab] = n        
        db.close()



