import _mysql

#if 1:
def test_npdescr_0():
    conn = _mysql.connect(  read_default_file="~/.my.cnf", read_default_group="client" ) 
    conn.query("select * from CalibPmtSpec limit 10")
    r = conn.store_result()
    d = r.npdescr()
    print repr(d) 
    assert len(d.names) == 14 , repr(d)

"""
   looks like numpy behavior change in recent numpy ...
       dtype([('Tables_in_offline_db_20101227', '|S64')])
          'There are no fields in dtype |S64.'
"""
def test_npdescr_1():
    conn = _mysql.connect(  read_default_file="~/.my.cnf", read_default_group="client" ) 
    conn.query("show tables")
    r = conn.store_result()
    d = r.npdescr()
    print repr(d) 
    assert len(d.names) == 1, d
    assert d[d.names[0]].name == 'string512'  


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
    pass
    test_npdescr_0()
    test_npdescr_1()
    test_nparray_1()
    test_nparray_2()

