import _mysql
import env.mysql_np.npy as npy

conn = None

def setup():
    global conn
    conn = _mysql.connect(  read_default_file="~/.my.cnf", read_default_group="client" )   # _mysql.connection  instance 

def teardown():
    global conn
    conn.close()


def test_look():
    conn.query( "select * from DcsPmtHv limit 10" )
    result = conn.get_result()    
    npy.look( result )
    result.clear()     ## ESSENTIAL MEMORY CLEANUP
    result = None 

def test_datetime():
    conn.query( "select * from DcsPmtHvVld limit 10" )
    result = conn.get_result()   
    npy.look( result )
    result.clear()     ## ESSENTIAL MEMORY CLEANUP
    result = None 

def test_fetch():
    conn.query( "select * from DcsPmtHv limit 10" )
    a = npy.fetch(conn)
    print  a
 

if __name__ == '__main__':
    setup()

    test_look()
    test_fetch()
    test_datetime()

    #teardown()




