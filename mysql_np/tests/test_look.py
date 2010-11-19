
import env.mysql_np.npy as npy
import _mysql

conn = None

 


def setup():
    global conn
    conn = _mysql.connect(  read_default_file="~/.my.cnf", read_default_group="client" )   # _mysql.connection  instance 

def teardown():
    global conn
    conn.close()

def fetch_np():
    """
        _mysql.connect in instance of CPython implemented class 
         ... so cannot add methods OR inherit from it ???
        
    """
    global conn 
    return npy.fetch_np( conn )


def test_look():
    conn.query( "select * from DcsPmtHv limit 10" )
    result = conn.get_result()      ## _mysql.result object    (slow)
    npy.look( result )
    result.clear()     ## ESSENTIAL MEMORY CLEANUP
    result = None 


def test_fetch():
    """ Messy interface   """
    conn.query( "select * from DcsPmtHv limit 10" )
    result = conn.get_result()      ## _mysql.result object    (slow)
    npy.look( result )
    a = npy.fetch( result )
    print  a
    result.clear()     ## ESSENTIAL MEMORY CLEANUP
    result = None 


def test_fetch_np():
    """ this way the cleanup is taken care off """
    conn.query( "select * from DcsPmtHv limit 10" )
    a = npy.fetch_np(conn)
    print  a
 

if __name__ == '__main__':
    setup()
    test_fetch()
    test_fetch_np()
    #teardown()

    conn.query("select * from DcsPmtHv limit 10000")
    a = fetch_np()



