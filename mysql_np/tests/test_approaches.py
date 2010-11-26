import numpy as np
import MySQLdb
import _mysql
from env.dyb.db import Qry

def test_fromiter():
    conn = _mysql.connect(  read_default_file="~/.my.cnf", read_default_group="client" )   # _mysql.connection  instance 
    q = Qry("DcsPmtHv", limit=100 )
    conn.query( str(q) )
    result = conn.get_result()      ## _mysql.result object    (slow)
    a = np.fromiter( result , dtype=np.dtype(q.descr), count = q.limit )
    result.clear()     ## ESSENTIAL MEMORY CLEANUP
    result = None 
    conn.close()
    print repr(a)
    n = len(a)
    assert n == q.limit , ( n , q.limit ) 

def test_transitional():
    conn = MySQLdb.connect(  read_default_file="~/.my.cnf", read_default_group="client" )
    cursor = conn.cursor()
    q = Qry("DcsPmtHv", limit=100 )

    # transitioning from cursors to lower level
    #
    # cursor.execute( q )   
    # does 
    #   _clear 
    #   _query  ... 
    #         _flush 
    #         cursor._result = Result(cursor) 
    #      BUT instantiation of the Result object 
    #         ... pulls the rows and flushes the result
    #             setting result to None
    #      SO NO GOOD    

    cursor._clear()
    # adapt from  from cursor._query 
    db = cursor._get_db()    ## the  _mysql.connection  instance 
    cursor._flush()          ## clears result 
    cursor._executed = str(q)
    db.query(str(q))

    # adapt from  Result.__init__
    result = db.get_result(cursor.use_result)   ## _mysql.result object
    a = np.fromiter( result  , dtype=np.dtype(q.descr) )

    # after usage 
    result.clear()
    result = None
    print repr(a)

    n = len(a)
    assert n == q.limit , ( n , q.limit ) 



if __name__ == '__main__':
    test_fromiter()
    test_transitional()

