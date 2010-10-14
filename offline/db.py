#!/usr/bin/env python
import os
import MySQLdb
from dbconf import DBConf

class DB:
    def __init__(self, sect=None ,  **kwa ):
        """
             Connection parameters obtained from DBConf for consistency 
             with other DB usage.   

             A successful connection to "sectname" requires the config file 
             (default ~/.my.cnf) named section to provide the below keys :  

             [sectname]
             host = 203....
             user = ...
             password = ...
             database = ...

        """
        try:  
            dbc = DBConf(sect=sect, **kwa)
            self.conn = MySQLdb.connect( **dbc.mysqldb_parameters() ) 
            self.dbc = dbc
        except MySQLdb.Error, e: 
            print "Error %d: %s " % ( e.args[0], e.args[1] )
         
    def close(self):
        self.conn.close()

    def execute_(self, cmd):
        cursor = self.conn.cursor(MySQLdb.cursors.DictCursor)
        #cursor = self.conn.cursor()
        cursor.execute( cmd )
        return cursor

    def fetchone(self, cmd ): 
        cursor = self.execute_(cmd)
        row = cursor.fetchone()
        cursor.close()
        return row

    def fetchcount(self, cmd ): 
        row = self.fetchone(cmd)
        assert len(row) == 1
        return row.values()[0]

    def fetchall(self, cmd ): 
        cursor = self.execute_(cmd)
        rows = cursor.fetchall()
        self.count = cursor.rowcount
        cursor.close()
        return rows

    def __call__(self, cmd):return self.fetchall(cmd)



def main():
    import sys
    sect = len(sys.argv)>1 and sys.argv[1] or "offline_db"
    print "connecting to \"%s\" and listing table counts " % sect 
    db = DB(sect, verbose=True )   
    rec = db.fetchone("SELECT VERSION()")
    print rec
    for rec in db("SHOW TABLES"):
        tab = rec.values()[0]
        cnt = db.fetchone("SELECT COUNT(*) FROM  %s" % tab )
        n = cnt.values()[0]
        print "%-30s : %s " % ( tab , n )
    db.close()


if __name__=='__main__':
    main()
    
