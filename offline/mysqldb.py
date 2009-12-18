import MySQLdb

class DBP(dict):
    def __init__(self):
        from private import Private
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

      Usage example :

     In [1]: from env.offline.mysqldb import DB
     In [2]: db = DB()
     In [3]: db("SHOW TABLES")
     Out[3]: 
     ({'Tables_in_mydb': 'SimPmtSpec'},
      {'Tables_in_mydb': 'SimPmtSpecVld'},
      {'Tables_in_mydb': 'auth_group'},
       ...

     In [5]: db("describe SimPmtSpecVld")
     Out[5]: 
     ({'Default': None,
       'Extra': 'auto_increment',
       'Field': 'SEQNO',
       'Key': 'PRI',
       'Null': '',
       'Type': 'int(11)'},
      {'Default': '0000-00-00 00:00:00',
      ...


    """
    def __init__(self):
        try:  
            self.conn = MySQLdb.connect( **DBP() ) 
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
    db = DB()
    rec = db.fetchone("SELECT VERSION()")
    print rec
    for rec in db("SHOW TABLES"):
        print rec 
    db.close()

    
