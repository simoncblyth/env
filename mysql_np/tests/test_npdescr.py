import _mysql

def test_npdescr():
    conn = _mysql.connect(  read_default_file="~/.my.cnf", read_default_group="client" ) 
    conn.query("show tables")
    r = conn.store_result()
    d = r.npdescr()
    assert d[0][1] == "|S64",  d
    print repr(d) 


def test_nparray_1():
    conn = _mysql.connect(  read_default_file="~/.my.cnf", read_default_group="client" ) 
    conn.query("show tables")
    r = conn.store_result()
    a = r.fetch_nparray()
    print repr(a) 
    conn.close()

def test_nparray_2():
    conn = _mysql.connect(  read_default_file="~/.my.cnf", read_default_group="client" ) 
    conn.query("select * from CalibPmtSpec limit 100")
    r = conn.store_result()
    a = r.fetch_nparray()
    print repr(a) 
    conn.close()



if __name__ == '__main__':
    test_npdescr()
    test_nparray_1()
    test_nparray_2()

