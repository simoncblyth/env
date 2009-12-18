import MySQLdb

class DBP(dict):
    def __init__(self):
        from private import Private
        p = Private()
        self['host'] = p('DATABASE_HOST') 
        self['db'] = p('DATABASE_NAME')  
        self['user'] = p('DATABASE_USER')
        self['passwd'] = p('DATABASE_PASSWORD')

if __name__=='__main__':
    con = MySQLdb.connect( **DBP() )
    cur = con.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT VERSION()")
    rec = cur.fetchone()
    print rec 

    cur.close()
    con.commit()

