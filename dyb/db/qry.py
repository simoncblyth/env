
class Qry(dict):
    """
          Parameterizs SQL queries against DBI tables 
           ... avoiding duplication in application of common fixups 
          and collecting those fixups for future applications at source.
   
          Because sometimes/always NULL : 
                 coalesce(INSERTDATE,TIMESTART) as INSERTDATE 
             
          but coalesce result type coming in as strings so convert ...
                 convert(coalesce(INSERTDATE,TIMESTART),DATETIME) as INSERTDATE

           Defining the fields ahead of the query is only needed for 
           cython optimizations using fixed/compiled types.

    """
    _vld = "SEQNO TIMESTART TIMEEND SITEMASK SIMMASK SUBSITE TASK AGGREGATENO VERSIONDATE INSERTDATE".split()
    _sql = "select %(_cols)s from %(_tab)s %(_limoff)s  "
    _qry = {
        'DEFAULT':{
                'cols':'*',
               'INSERTDATE':"convert(coalesce(INSERTDATE,TIMESTART),DATETIME) as INSERTDATE",
               'VERSIONDATE':"convert(coalesce(VERSIONDATE,TIMESTART),DATETIME) as VERSIONDATE",
               'AGGREGATENO':"coalesce(AGGREGATENO,-1) as AGGREGATENO",
                  },
       'DcsPmtHv':{
             '_descr':[('SEQNO', 'i4'), ('ROW_COUNTER', 'i4'), ('ladder', 'i4'),('col','i4'),('ring','i4'),('voltage','f4'),('pw','i4')],
               'table':'DcsPmtHv',
              'limit':10000,
            'voltage':"coalesce(voltage,0.) as voltage",
                 'pw':"coalesce(pw,0) as pw",
                     }, 
            }

    def _colnames(self):
        """
            List column names without fixes applied
        """ 
        if 'cols' in self.kwargs:
            return self.kwargs['cols']
        elif self.get('_descr',None):
            return map(lambda _:_[0], self['_descr'])
        elif self.is_vld:
            return self._vld 
        else:
            return []
    colnames = property(_colnames)
    
    def _cols(self):
        """ 
            Apply column wrappers where defined 
              eg : use coalesce to avoid NULLs
        """
        cols = []
        for name in self.colnames:
            if name in self:
                wrapr= self[name]
                cols.append( wrapr )
            else:
                cols.append(name) 
        return cols and ",".join(cols) or "*"
    cols = property(_cols) 

    def _tab(self):
        return self.get('table',self.name)
    tab = property( _tab )       

    def _limit(self):
        i = int(self.get('limit',1000000)) 
        if i < 0: 
            return "" 
        return "limit %d" % i
    limit  = property( _limit )

    def _offset(self):
        i = int(self.get('offset',0)) 
        return i < 0 and "" or "offset %d" % i
    offset  = property( _offset )

    def _limoff(self):
        lim = self.limit
        off = self.offset
        return lim and "%s %s " % ( lim , off ) or "" 
    limoff  = property( _limoff )


    def _pars(self):
        return dict(self, _cols=self.cols, _tab=self.tab, _limoff=self.limoff )
    pars = property( _pars ) 


    sql  = property(lambda self:self._sql % self.pars )

    connargs = property(lambda self:dict(read_default_file="~/.my.cnf", read_default_group=self['read_default_group'])) 
    descr  = property( lambda self:self['_descr'])
    method = property( lambda self:int(self.get('_method',0)) )
    is_special = property( lambda self:self.name in self._qry)
    is_vld = property( lambda self:self.name.upper().endswith('VLD'))

    def __repr__(self):
        return "%s(\"%s\",%s)" % ( self.__class__.__name__, self.name , ",".join(["%s=\"%s\"" % _ for _ in self.kwargs.items() ] ) ) 

    def __str__(self):
        return self.sql 

    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs 
        
        settings = self._qry['DEFAULT']
        if self.is_special:
            settings.update( self._qry[name] )

        self.update(settings) 
        self.update(kwargs)



def test_vld():
    q = Qry("DummyVld")
    assert q._vld == q.colnames , q 


if __name__=='__main__':
    pass
    test_vld()
    q = Qry("DummyVld")

    q = Qry("SimPmtSpecVld")
    print q.sql


