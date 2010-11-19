
import env.mysql_np.npy as npy
import _mysql

conn = None
 


def setup():
    global conn
    conn = _mysql.connect(  read_default_file="~/.my.cnf", read_default_group="client" )   # _mysql.connection  instance 

def teardown():
    global conn
    conn.close()

def fetch():
    """
        _mysql.connect in instance of CPython implemented class 
         ... so cannot add methods OR inherit from it ???
        
    """
    global conn 
    return npy.fetch( conn )


def test_look():
    conn.query( "select * from DcsPmtHv limit 10" )
    result = conn.get_result()      ## _mysql.result object    (slow)
    npy.look( result )
    result.clear()     ## ESSENTIAL MEMORY CLEANUP
    result = None 


def test_fetch():
    conn.query( "select * from DcsPmtHv limit 10" )
    a = npy.fetch(conn)
    print  a
 

if __name__ == '__main__':
    setup()
    test_fetch()
    #teardown()

    conn.query("select * from DcsPmtHv limit 10000")
    a = fetch()



