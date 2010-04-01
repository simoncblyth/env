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
        group = kwa.get('group','client')
        stamp = kwa.get('stamp',None)  
        db = DB( read_default_file=os.path.expanduser("~/.my.cnf"), read_default_group=group )
        rec = db.fetchone("SELECT VERSION()")  
        for rec in db("SHOW TABLES"):
            tab = rec.values()[0]
            cnt = db.fetchone("SELECT COUNT(*) FROM  %s" % tab )
            n = cnt.values()[0]
            self[tab] = n        
        db.close()

    #def __init__(self, *args, **kwa ):
    #    print "(client)%s.__init__ " % ( self.__class__.__name__ )



