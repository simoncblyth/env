
import MySQLdb

class DB:
    def __init__(self, **cfg ):
        try:  
            self.conn = MySQLdb.connect( **cfg ) 
            self.cfg = cfg
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

    def fetchall(self, cmd ): 
        cursor = self.execute_(cmd)
        rows = cursor.fetchall()
        self.count = cursor.rowcount
        cursor.close()
        return rows

    def __call__(self, cmd):return self.fetchall(cmd)


if __name__=='__main__':
   
    from dbconf import DBConf 
    cfg = DBConf( sect="testdb" )  
    print cfg

    db = DB( **cfg )
    rec = db.fetchone("SELECT VERSION()")
    print rec
    for rec in db("SHOW TABLES"):
        print rec 
        tab = rec.values()[0]
        print db("DESCRIBE %s" % tab )
    db.close()

    
