import MySQLdb

class DBP(dict):
    def __init__(self):
        from env.base.private import Private
        p = Private()
        self['host'] = p('DATABASE_HOST') 
        self['db'] = p('DATABASE_NAME')  
        self['user'] = p('DATABASE_USER')
        self['passwd'] = p('DATABASE_PASSWORD')
 
class DB:
    """
      Best description :
         http://www.kitebird.com/articles/pydbapi.html

         http://www.devshed.com/c/a/Python/MySQL-Connectivity-With-Python/
         http://mysql-python.sourceforge.net/MySQLdb.html#mysqldb


    """
    def __init__(self):
        try:  
            self.conn = MySQLdb.connect( **DBP() ) 
        except MySQLdb.Error, e:
            print dbp
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
    db = DB()
    rec = db.fetchone("SELECT VERSION()")
    print rec
    for rec in db("SHOW TABLES"):
        print rec 
    db.close()

    
