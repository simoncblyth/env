"""

  Following first instantiation of the Dbi singleton 
  with the db engine metadata, 
  (usually done by TG2 in  model/__init__.py::init_model)
  
     from offlinedb.model.dbi import Dbi 
     Dbi( metadata )

  The mapped table classes are available in this modules scope
  the ones with _ are joins to the validity tables

    from offlinedb.model.dbi import SimPmtSpec, SimPmtSpec_
    from offlinedb.model.dbi import soup, dbi_

"""

class _Dbi(object):
    head = """
# expose the instance as the only dynamic access
global dbi_ 
dbi_ = self

## exposing soup is a whopping great security risk ... 
# global soup
# soup = self.soup
"""
    tmpl = """
# exports to module level ... per table exports useful for interactive usage
global %(t)s
%(t)s = self.payload( "%(t)s" )

global %(t)sVld
%(t)s%(VLD)s = self.validity( "%(t)s" )

global %(t)s_
%(t)s_ = self.pair( "%(t)s" )
"""
    FK = 'SEQNO'
    VLD = 'Vld'

    def __init__(self, *args, **kwa):
        from sqlalchemy.ext.sqlsoup import SqlSoup
        self.soup = SqlSoup(*args, **kwa)
        self.exec_()
   
    def payload( self, t ):return getattr( self.soup , t )
    def validity(self, t ):return getattr( self.soup , "%s%s" % (t, _Dbi.VLD) )
    def pair(self, t):
        pay = self.payload(t)
        vld = self.validity(t)
        return self.soup.join( pay , vld , getattr( pay, _Dbi.FK ) == getattr( vld, _Dbi.FK ), isouter=False )

    def table_names(self):
        return [n[0:-3] for n in self.soup.engine.table_names() if n.endswith(_Dbi.VLD)]
    def __repr__(self):
        return "<Dbi singleton instance for %s at %s >" % ( repr(self.soup) , self.__hash__() )
    def __str__(self):
        return "\n".join( [_Dbi.head] + [ _Dbi.tmpl % {'t':t , 'VLD':_Dbi.VLD } for t in self.table_names()]) 
    def exec_(self):         
       exec str(self)

## singleton 
_instance = None 
def Dbi(*args, **kwa):
    global _instance
    if not(_instance): 
        _instance = _Dbi(*args, **kwa) 
    return _instance


if __name__=='__main__':

    from env.base.private import Private
    p = Private()
    from sqlalchemy import create_engine, MetaData
    engine = create_engine( p('DATABASE_URL') )
    metadata = MetaData( engine )
    Dbi( metadata )
    print dbi_ 
    #print repr(soup)
    print SimPmtSpec.first()
    print SimPmtSpecVld.first()
    print SimPmtSpec_.first()


