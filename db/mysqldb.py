import os
import MySQLdb

class DB:
    def __init__(self, **cfg ):
        """
             Connection parameters are passed to MySQLdb as is, except group which 
             is used to signal reading from the ~/.my.cnf defaults file
        """
        try:  
            group = cfg.pop('group',None)
            if group:cfg.update(  read_default_file=os.path.expanduser("~/.my.cnf"), read_default_group=group )
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


if __name__=='__main__':

    cnf = os.path.expanduser('~/.my.cnf')
    db = DB( read_default_file=cnf )
    #db = DB( read_default_file=cnf , read_default_group='test' )  ## pick alternate to "client" group in the ~/.my.cnf

    rec = db.fetchone("SELECT VERSION()")
    print rec
    for rec in db("SHOW TABLES"):
        tab = rec.values()[0]
        cnt = db.fetchone("SELECT COUNT(*) FROM  %s" % tab )
        n = cnt.values()[0]
        print "%-30s : %s " % ( tab , n )
    db.close()

    
