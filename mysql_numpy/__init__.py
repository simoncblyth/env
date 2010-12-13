import _mysql 


class DB(object):
    """   _mysql.connect in instance of CPython implemented class 
          ... apparently cannot add methods OR inherit from it ???
          hence use composition with DB class having _mysql.connect as member
    """
    def __init__(self, *args, **kwargs ):
        if args:
            kwargs.update( read_default_group=args[0] )         # some sugar 
        kwargs.setdefault( "read_default_file", "~/.my.cnf" ) 
        kwargs.setdefault( "read_default_group",  "client" )   # _mysql.connection  instance 
        conn = _mysql.connection( **kwargs )
        self.conn = conn
        self.q = None

    def __call__(self, q , **kwargs):
        self.q = q 
        fast = kwargs.pop('kwargs',False)
        conn = self.conn
        conn.query( str(q) )
        r = conn.store_result()
        if fast: 
            return r.fetch_nparrayfast(**kwargs)
        else:
            return r.fetch_nparray(**kwargs)

    def close(self):
        self.conn.close()


