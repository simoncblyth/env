import _mysql

def test_npdescr():
    conn = _mysql.connect(  read_default_file="~/.my.cnf", read_default_group="client" ) 
    conn.query("show tables")
    r = conn.store_result()
    d = r.npdescr()
    assert d[0][1] == "|S64",  d

if __name__ == '__main__':
    test_npdescr()

