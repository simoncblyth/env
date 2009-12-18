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
    conn = MySQLdb.connect( **DBP() )
    cursor = conn.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("SELECT VERSION()")
    rec = cursor.fetchone()
    cursor.close()
    print rec 

    cur.close()
    conn.commit()

