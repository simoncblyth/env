
import numpy as np


class Qry(dict):
    _sql = "select %(cols)s from %(tab)s limit %(limit)s  "
    _qry = {
       'DcsPmtHv':{
             'descr':[('SEQNO', 'i4'), ('ROW_COUNTER', 'i4'), ('ladder', 'i4'),('col','i4'),('ring','i4'),('voltage','f4'),('pw','i4')],
               'tab':'DcsPmtHv',
             'limit':10000,
             'voltage':lambda _:"coalesce(%s,0.)" % _,
                  'pw':lambda _:"coalesce(%s,0)" % _,
                  } 
            }

    def _cols(self):
        """ special coalese handling for some columns with NULLs that cause issue at MySQLdb level  """
        cols = []
        for name in self.colnames:
            cols.append( name in self and self[name](name) or name )
        return cols

    cols = property(_cols) 
    colnames = property(lambda self:map(lambda _:_[0], self['descr'])) 
    dtype = property(lambda self:np.dtype(self['descr']))
    sql  = property(lambda self:self._sql % self )
    connargs = property(lambda self:dict(read_default_file="~/.my.cnf", read_default_group=self['read_default_group'])) 
    limit = property( lambda self:int(self['limit']) )
    method = property( lambda self:int(self.get('method',0)) )

    def __repr__(self):
        return "%s(\"%s\",%s)" % ( self.__class__.__name__, self.name , ",".join(["%s=\"%s\"" % _ for _ in self.kwargs.items() ] ) ) 

    def __str__(self):
        return self.sql 

    def __init__(self, name, **kwargs):
        assert name in self._qry
        self.name = name
        self.kwargs = kwargs 
        self.update(self._qry[name]) 
        self['cols'] = ",".join(self.cols)
        self.update(kwargs)




